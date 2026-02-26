import os
import glob
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import PolygonSelector, RectangleSelector
from matplotlib.patches import Ellipse
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from density_calculation_functions import get_raw_densities, calculate_densities_batch, estimate_grid_batches
from gmm_sorting_functions import (
    gm_seed_local_max, fit_gaussian_model_from_seed, _create_init_cluster_from_seed, _fit_model_size,
    make_gaussian_model_from_cluster, iterate_GM_model, GaussianModel, apply_bic_refinement,
    prob_density_from_curve_or_formula,
)
from unit_classes import Bound, Unit


class GMCSorter:
    def __init__(self, root):
        self.root = root
        self.root.title("GMC Sorter")
        self.root.geometry("1400x900")
        try:
            self.root.update_idletasks()
            w = self.root.winfo_screenwidth()
            h = int(self.root.winfo_screenheight() * 0.9)
            self.root.geometry(f"{w}x{h}+0+0")
        except tk.TclError:
            pass
        
        # Data storage
        self.all_properties = None
        self.prop_titles = None
        self.sort_feature_idx = 0  # Global feature index to sort data by (load-time setting)
        self.clust_ind = None
        self.cluster_den = None
        self.background_den = None
        self.included_feature_indexes = None
        self.seeds = []
        self.gaussian_models = []
        self.data_folder = None
        
        # Bounds storage
        self.bounds = []  # List of Bound objects
        self.cluster_seeds = []  # List of init_cluster dicts: {'bounds': list(Bound), 'spike_ids': ndarray}
        self.next_unit_label = 0
        
        # UI state
        self.sorting_type = tk.StringVar(value="Single seed")
        self.dot_density = tk.DoubleVar(value=1.0)
        self.dot_size = tk.DoubleVar(value=1.0)
        self.max_dots = tk.IntVar(value=30000)
        self.density_hue_min = tk.DoubleVar(value=1.0)
        self.density_hue_max = tk.DoubleVar(value=6.0)
        self.density_hue_source = tk.StringVar(value="Cluster index")  # "Cluster density", "Background density", "Cluster index"
        self.unit_id = None
        self.unit_labels = []
        self.unit_info = {}  # Dictionary mapping unit_label (int) -> Unit object
        self.curr_max_prob = None  # Normalized mah dist per spike (mah/threshold); in-cluster when current model gives smaller value
        self.selected_unit = tk.StringVar(value="new_unit")
        self.compare_unit = tk.StringVar(value="new_unit")
        self.selected_cluster_seed = tk.StringVar(value="none")
        self.plot_sorted = tk.BooleanVar(value=True)
        
        # Display ranges
        self.global_display_ranges = {}
        self.updating_programmatically = False
        
        # Feature selections for 4 plots
        self.feature_selections = []
        self.saved_feature_selections = None  # For loading saved state
        
        # Add seed mode (Gaussian model): plot_idx to add seed on next click, or None
        self.add_seed_plot_idx = None
        self._add_seed_cid = None
        self._add_seed_last_click_time = 0
        self._add_seed_last_click_xy = None
        self._add_seed_pending_after_id = None
        # Single source of truth for all settings; populated by load data (last_settings.npz) or Load settings.
        # Keys: 'grid', 'seeds', 'gm' -> dict of section-specific keys (name-based where applicable).
        self.settings = {'grid': {}, 'seeds': {}, 'gm': {}}
        # Polygon selectors for bounds
        self.polygon_selectors = [None] * 4
        self.focus_polygons = [None] * 4  # For highlighting selected bounds
        self.selected_bounds = [None] * 4  # Currently selected bound for each plot
        self.delete_bound_buttons = [None] * 4  # Delete bound buttons per plot
        self.make_bound_buttons = [None] * 4  # Make bound buttons per plot
        self.rectangle_selectors = [None] * 4  # Rectangle selectors for zoom
        
        # Grid visualization (for density calculation)
        self.grid_window = None
        self.grid_range_vars = {}
        self.grid_step_vars = {}
        
        # Store references to UI widgets for enable/disable (will be set in create_ui)
        self.sorting_combo = None
        
        # Density ranges and grid steps (for density calculation)
        self.density_ranges = None
        self.grid_steps = None
        
        # Create UI
        self.create_ui()
        
    def create_ui(self):
        """Create the main UI"""
        # Top frame - path and load/save
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Path field
        ttk.Label(top_frame, text="Data Folder:").pack(side=tk.LEFT, padx=5)
        self.path_var = tk.StringVar()
        path_entry = ttk.Entry(top_frame, textvariable=self.path_var, width=50)
        path_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Browse button
        browse_btn = ttk.Button(top_frame, text="Browse", command=self.browse_folder)
        browse_btn.pack(side=tk.LEFT, padx=5)
        
        # Load button
        load_btn = ttk.Button(top_frame, text="Load", command=self.load_data)
        load_btn.pack(side=tk.LEFT, padx=5)
        # Load settings (from defaults folder; optional name)
        load_settings_btn = ttk.Button(top_frame, text="Load settings", command=self._load_settings_dialog)
        load_settings_btn.pack(side=tk.LEFT, padx=2)
        # Save settings (to last_settings.npz or named file in defaults folder)
        save_settings_btn = ttk.Button(top_frame, text="Save settings", command=self._save_settings_dialog)
        save_settings_btn.pack(side=tk.LEFT, padx=2)

        # Sort by feature (load-time setting; used when loading and for Gaussian fit)
        ttk.Label(top_frame, text="Sort by:").pack(side=tk.LEFT, padx=(15, 2))
        self.sort_feature_combo = ttk.Combobox(top_frame, width=14, state='readonly')
        self.sort_feature_combo.pack(side=tk.LEFT, padx=2)
        self.sort_feature_combo.bind('<<ComboboxSelected>>', self._on_sort_feature_changed)

        # Save button
        save_btn = ttk.Button(top_frame, text="Save", command=self.save_data)
        save_btn.pack(side=tk.LEFT, padx=5)
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - scrollable toolbar
        left_panel = ttk.Frame(main_frame, width=250)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        left_panel.pack_propagate(False)
        
        left_canvas = tk.Canvas(left_panel, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=left_canvas.yview)
        scrollable_toolbar = ttk.Frame(left_canvas)
        scrollable_toolbar.bind("<Configure>", lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.create_window((0, 0), window=scrollable_toolbar, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        def _on_left_toolbar_mousewheel(e):
            left_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        left_canvas.bind("<MouseWheel>", _on_left_toolbar_mousewheel)
        scrollable_toolbar.bind("<MouseWheel>", _on_left_toolbar_mousewheel)
        
        # Dot density control (at the top)
        density_frame = ttk.LabelFrame(scrollable_toolbar, text="Dot Density")
        density_frame.pack(pady=5, fill=tk.X, padx=5)
        density_scale = ttk.Scale(density_frame, from_=0.0, to=1.0, variable=self.dot_density, 
                                 orient=tk.HORIZONTAL)
        density_scale.pack(fill=tk.X, padx=5, pady=5)
        # Update plots only on slider release
        density_scale.bind("<ButtonRelease-1>", lambda e: self.update_all_plots())
        self.density_label = ttk.Label(density_frame, text="1.00")
        self.density_label.pack()
        
        # Max dots control
        max_dots_frame = ttk.Frame(density_frame)
        max_dots_frame.pack(fill=tk.X, padx=5, pady=2)
        self.max_dots_label = ttk.Label(max_dots_frame, text="Max dots:")
        self.max_dots_label.pack(side=tk.LEFT, padx=2)
        max_dots_entry = ttk.Entry(max_dots_frame, textvariable=self.max_dots, width=10)
        max_dots_entry.pack(side=tk.LEFT, padx=2)
        
        def validate_max_dots(e=None):
            try:
                val = int(self.max_dots.get())
                if val < 1:
                    self.max_dots.set(1)
                self.update_all_plots()
            except (ValueError, tk.TclError):
                # Invalid value, reset to previous or default
                self.max_dots.set(30000)
                self.update_all_plots()
        
        max_dots_entry.bind("<Return>", validate_max_dots)
        max_dots_entry.bind("<FocusOut>", validate_max_dots)
        
        # Density hue: source (cluster density / background density / cluster index) and range (min=black, max=red)
        density_hue_frame = ttk.Frame(density_frame)
        density_hue_frame.pack(fill=tk.X, padx=5, pady=5)
        hue_row1 = ttk.Frame(density_hue_frame)
        hue_row1.pack(fill=tk.X)
        ttk.Label(hue_row1, text="Hue:").pack(side=tk.LEFT, padx=2)
        self.density_hue_source_combo = ttk.Combobox(hue_row1, textvariable=self.density_hue_source, width=16, state="readonly")
        self.density_hue_source_combo['values'] = ("Cluster density", "Background density", "Cluster index")
        self.density_hue_source_combo.pack(side=tk.LEFT, padx=2)
        self.density_hue_source_combo.bind("<<ComboboxSelected>>", lambda e: self.update_all_plots())
        hue_row2 = ttk.Frame(density_hue_frame)
        hue_row2.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(hue_row2, text="min:").pack(side=tk.LEFT, padx=2)
        density_hue_min_entry = ttk.Entry(hue_row2, textvariable=self.density_hue_min, width=6)
        density_hue_min_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(hue_row2, text="max:").pack(side=tk.LEFT, padx=2)
        density_hue_max_entry = ttk.Entry(hue_row2, textvariable=self.density_hue_max, width=6)
        density_hue_max_entry.pack(side=tk.LEFT, padx=2)
        def on_density_hue_change(e=None):
            try:
                self.update_all_plots()
            except (ValueError, tk.TclError):
                pass
        density_hue_min_entry.bind("<Return>", on_density_hue_change)
        density_hue_min_entry.bind("<FocusOut>", on_density_hue_change)
        density_hue_max_entry.bind("<Return>", on_density_hue_change)
        density_hue_max_entry.bind("<FocusOut>", on_density_hue_change)
        
        # Dot size control (below dot density)
        dot_size_frame = ttk.LabelFrame(scrollable_toolbar, text="Dot Size")
        dot_size_frame.pack(pady=5, fill=tk.X, padx=5)
        dot_size_scale = ttk.Scale(dot_size_frame, from_=0.25, to=5.0, variable=self.dot_size,
                                  orient=tk.HORIZONTAL)
        dot_size_scale.pack(fill=tk.X, padx=5, pady=5)
        dot_size_scale.bind("<ButtonRelease-1>", lambda e: self.update_all_plots())
        self.dot_size_label = ttk.Label(dot_size_frame, text="1.00")
        self.dot_size_label.pack()
        ttk.Button(dot_size_frame, text="Home", command=self.reset_all_zoom).pack(pady=2)
        
        # Calculate Densities (visible in both Single seed and Cluster)
        calc_dens_frame = ttk.Frame(scrollable_toolbar)
        calc_dens_frame.pack(pady=5, fill=tk.X, padx=5)
        self.calc_densities_btn = ttk.Button(calc_dens_frame, text="Calculate Densities",
                                            command=self.calculate_densities,
                                            state=tk.NORMAL if self.all_properties is not None else tk.DISABLED)
        self.calc_densities_btn.pack(fill=tk.X)
        
        # Seed type dropdown
        sorting_frame = ttk.LabelFrame(scrollable_toolbar, text="Seed Type")
        sorting_frame.pack(pady=5, fill=tk.X, padx=5)
        self.sorting_combo = ttk.Combobox(sorting_frame, textvariable=self.sorting_type, 
                                    values=["Single seed", "Cluster"], state="readonly", width=20)
        self.sorting_combo.pack(pady=5, padx=5)
        self.sorting_combo.bind("<<ComboboxSelected>>", lambda e: self.on_sorting_type_changed())
        
        # Buttons frame (will show different buttons based on sorting type)
        self.buttons_frame = ttk.Frame(scrollable_toolbar)
        self.buttons_frame.pack(pady=5, fill=tk.X, padx=5)
        
        # Initialize button references (will be created in on_sorting_type_changed; calc_densities_btn already created above)
        self.find_seeds_btn = None
        self.compute_gaussian_models_button = None
        self.define_bounds_btn = None
        self.clear_bounds_btn = None
        self.assign_units_btn = None
        
        # Plot sorted checkbox (for Gaussian model mode)
        self.plot_sorted_check = ttk.Checkbutton(scrollable_toolbar, text="Plot sorted", 
                                                 variable=self.plot_sorted, 
                                                 command=lambda: self.update_all_plots())
        self.plot_sorted_check.pack(pady=5)
        
        # Unit selection (for viewing assigned units)
        unit_frame = ttk.LabelFrame(scrollable_toolbar, text="Selected Unit")
        unit_frame.pack(pady=5, fill=tk.X, padx=5)
        self.resort_units_btn = ttk.Button(unit_frame, text="Resort labels by position", command=self.resort_unit_labels_by_position)
        self.resort_units_btn.pack(pady=(5, 2), fill=tk.X)
        unit_combo_row = ttk.Frame(unit_frame)
        unit_combo_row.pack(pady=5, padx=5, fill=tk.X)
        ttk.Button(unit_combo_row, text="<", width=3, command=self.go_prev_unit).pack(side=tk.LEFT, padx=(0, 2))
        self.unit_combo = ttk.Combobox(unit_combo_row, textvariable=self.selected_unit, 
                                 state="readonly", width=18)
        self.unit_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(unit_combo_row, text=">", width=3, command=self.go_next_unit).pack(side=tk.LEFT, padx=(2, 0))
        self.unit_combo.bind("<<ComboboxSelected>>", lambda e: self.on_unit_selection_changed())
        self.unit_combo['values'] = ["new_unit"]
        
        self.refresh_gm_assignment_btn = ttk.Button(unit_frame, text="Refresh unit assignment", command=self.refresh_gm_unit_assignment)
        self.refresh_gm_assignment_btn.pack(pady=2, fill=tk.X)
        self.refresh_gm_assignment_btn.pack_forget()
        
        # Delete unit button (shown when a unit is selected)
        self.delete_unit_btn = ttk.Button(unit_frame, text="Delete Unit", 
                                         command=self.delete_selected_unit, state=tk.DISABLED)
        self.delete_unit_btn.pack(pady=2, fill=tk.X)
        
        # Edit unit button (shown when a unit is selected)
        self.edit_unit_btn = ttk.Button(unit_frame, text="Edit Unit", 
                                        command=self.edit_selected_unit, state=tk.DISABLED)
        self.edit_unit_btn.pack(pady=2, fill=tk.X)
        
        # ISI analysis button (below selected unit menu)
        self.isi_analysis_btn = ttk.Button(unit_frame, text="ISI analysis", 
                                          command=self.show_isi_analysis, state=tk.DISABLED)
        self.isi_analysis_btn.pack(pady=2, fill=tk.X)
        
        # Compare Unit (scrollable, same units as Selected Unit)
        compare_frame = ttk.LabelFrame(scrollable_toolbar, text="Compare Unit")
        compare_frame.pack(pady=5, fill=tk.X, padx=5)
        compare_combo_row = ttk.Frame(compare_frame)
        compare_combo_row.pack(pady=5, padx=5, fill=tk.X)
        ttk.Button(compare_combo_row, text="<", width=3, command=self.go_prev_compare_unit).pack(side=tk.LEFT, padx=(0, 2))
        self.compare_combo = ttk.Combobox(compare_combo_row, textvariable=self.compare_unit, 
                                         state="readonly", width=18)
        self.compare_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(compare_combo_row, text=">", width=3, command=self.go_next_compare_unit).pack(side=tk.LEFT, padx=(2, 0))
        self.compare_combo.bind("<<ComboboxSelected>>", lambda e: self._update_combined_isi_button_visibility())
        
        # Cross ISI button (visible only when both Selected Unit and Compare Unit have a unit selected)
        self.combined_isi_btn = ttk.Button(compare_frame, text="Cross ISI", 
                                           command=self.show_cross_isi_analysis)
        self.combined_isi_btn.pack(pady=2, fill=tk.X)
        self.combined_isi_btn.pack_forget()  # hidden until both units selected
        self.compare_combo['values'] = list(self.unit_combo['values']) if self.unit_combo['values'] else ["new_unit"]
        
        # Right panel - plots with controls
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create 4 plots (2x2 grid) with controls
        self.figures = []
        self.canvases = []
        self.axes = []
        self.plot_frames = []
        
        for i in range(4):
            # Frame for each plot with controls
            plot_frame = ttk.LabelFrame(right_panel, text=f"Plot {i+1}")
            plot_frame.grid(row=i//2, column=i%2, padx=5, pady=5, sticky="nsew")
            right_panel.grid_rowconfigure(i//2, weight=1)
            right_panel.grid_columnconfigure(i%2, weight=1)
            
            self.plot_frames.append(plot_frame)
            
            # Feature selection frame
            feat_frame = ttk.Frame(plot_frame)
            feat_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Store feature selections (create StringVars first)
            x_var = tk.StringVar()
            y_var = tk.StringVar()
            
            ttk.Label(feat_frame, text="X:").pack(side=tk.LEFT, padx=2)
            x_combo = ttk.Combobox(feat_frame, textvariable=x_var, width=10, state="readonly")
            x_combo.pack(side=tk.LEFT, padx=2)
            
            ttk.Label(feat_frame, text="Y:").pack(side=tk.LEFT, padx=2)
            y_combo = ttk.Combobox(feat_frame, textvariable=y_var, width=10, state="readonly")
            y_combo.pack(side=tk.LEFT, padx=2)
            
            # Bound buttons (shown when Bounds sorting mode is active)
            make_bound_btn = ttk.Button(feat_frame, text="Make Bound", 
                                       command=lambda idx=i: self.make_bound_for_plot(idx))
            make_bound_btn.pack(side=tk.LEFT, padx=2)
            self.make_bound_buttons[i] = make_bound_btn
            
            delete_bound_btn = ttk.Button(feat_frame, text="Delete Bound", 
                                         command=lambda idx=i: self.delete_bound_for_plot(idx))
            delete_bound_btn.pack(side=tk.LEFT, padx=2)
            
            # Store reference to delete bound button for show/hide
            if not hasattr(self, 'delete_bound_buttons'):
                self.delete_bound_buttons = [None] * 4
            self.delete_bound_buttons[i] = delete_bound_btn
            
            # Add/remove seed button (Gaussian model only): single-click adds, double-click removes closest seed
            add_seed_btn = ttk.Button(feat_frame, text="Add/remove seed", 
                                     command=lambda idx=i: self.enable_add_seed_mode(idx))
            add_seed_btn.pack(side=tk.LEFT, padx=2)
            if not hasattr(self, 'add_seed_buttons'):
                self.add_seed_buttons = [None] * 4
            self.add_seed_buttons[i] = add_seed_btn
            add_seed_btn.pack_forget()
            
            # Zoom, Zoom out, and Home buttons
            zoom_home_frame = ttk.Frame(feat_frame)
            zoom_home_frame.pack(side=tk.RIGHT, padx=5)
            zoom_btn = ttk.Button(zoom_home_frame, text="Zoom", 
                                 command=lambda idx=i: self.enable_zoom(idx))
            zoom_btn.pack(side=tk.LEFT, padx=2)
            zoom_out_btn = ttk.Button(zoom_home_frame, text="Zoom out", 
                                     command=lambda idx=i: self.zoom_out(idx))
            zoom_out_btn.pack(side=tk.LEFT, padx=2)
            home_btn = ttk.Button(zoom_home_frame, text="Home", 
                                 command=lambda idx=i: self.reset_zoom(idx))
            home_btn.pack(side=tk.LEFT, padx=2)
            
            # Store feature selections
            self.feature_selections.append({
                'x': x_var,
                'y': y_var,
                'x_combo': x_combo,
                'y_combo': y_combo
            })
            
            x_combo.bind("<<ComboboxSelected>>", lambda e, idx=i: self.on_feature_change(idx))
            y_combo.bind("<<ComboboxSelected>>", lambda e, idx=i: self.on_feature_change(idx))
            
            # Navigation buttons frame for X axis (below plot) - pack first to reserve space
            x_nav_frame = ttk.Frame(plot_frame)
            x_nav_frame.pack(fill=tk.X, padx=5, pady=2, side=tk.BOTTOM)
            
            # Add spacer on left to center buttons
            left_spacer = ttk.Frame(x_nav_frame, width=1)
            left_spacer.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # X axis navigation buttons: << < > >>
            x_fast_left_btn = ttk.Button(x_nav_frame, text="<<", width=3,
                                        command=lambda idx=i: self.move_x_range(idx, -1.0))
            x_fast_left_btn.pack(side=tk.LEFT, padx=1)
            
            x_left_btn = ttk.Button(x_nav_frame, text="<", width=3,
                                    command=lambda idx=i: self.move_x_range(idx, -0.25))
            x_left_btn.pack(side=tk.LEFT, padx=1)
            
            x_right_btn = ttk.Button(x_nav_frame, text=">", width=3,
                                     command=lambda idx=i: self.move_x_range(idx, 0.25))
            x_right_btn.pack(side=tk.LEFT, padx=1)
            
            x_fast_right_btn = ttk.Button(x_nav_frame, text=">>", width=3,
                                         command=lambda idx=i: self.move_x_range(idx, 1.0))
            x_fast_right_btn.pack(side=tk.LEFT, padx=1)
            
            # Add spacer on right to center buttons
            right_spacer = ttk.Frame(x_nav_frame, width=1)
            right_spacer.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Frame for plot content (Y nav buttons + canvas)
            plot_content_frame = ttk.Frame(plot_frame)
            plot_content_frame.pack(fill=tk.BOTH, expand=True)
            
            # Navigation buttons frame for Y axis (left of plot)
            y_nav_frame = ttk.Frame(plot_content_frame)
            y_nav_frame.pack(side=tk.LEFT, padx=2, pady=5)
            
            # Add spacer at top to center buttons vertically
            top_spacer = ttk.Frame(y_nav_frame, height=1)
            top_spacer.pack(side=tk.TOP, fill=tk.Y, expand=True)
            
            # Y axis navigation buttons: ↑↑ ↑ ↓ ↓↓ (vertical)
            y_fast_up_btn = ttk.Button(y_nav_frame, text="↑↑", width=3,
                                      command=lambda idx=i: self.move_y_range(idx, 1.0))
            y_fast_up_btn.pack(side=tk.TOP, padx=1, pady=1)
            
            y_up_btn = ttk.Button(y_nav_frame, text="↑", width=3,
                                 command=lambda idx=i: self.move_y_range(idx, 0.25))
            y_up_btn.pack(side=tk.TOP, padx=1, pady=1)
            
            y_down_btn = ttk.Button(y_nav_frame, text="↓", width=3,
                                    command=lambda idx=i: self.move_y_range(idx, -0.25))
            y_down_btn.pack(side=tk.TOP, padx=1, pady=1)
            
            y_fast_down_btn = ttk.Button(y_nav_frame, text="↓↓", width=3,
                                        command=lambda idx=i: self.move_y_range(idx, -1.0))
            y_fast_down_btn.pack(side=tk.TOP, padx=1, pady=1)
            
            # Add spacer at bottom to center buttons vertically
            bottom_spacer = ttk.Frame(y_nav_frame, height=1)
            bottom_spacer.pack(side=tk.TOP, fill=tk.Y, expand=True)
            
            # Matplotlib figure
            fig = Figure(figsize=(6, 6))
            ax = fig.add_subplot(111)
            canvas = FigureCanvasTkAgg(fig, master=plot_content_frame)
            
            self.figures.append(fig)
            self.axes.append(ax)
            self.canvases.append(canvas)
            
            canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Initialize sorting type UI
        self.on_sorting_type_changed()
        
        # Load saved state
        self.load_state()
        
        # Load consolidated settings (grid, seeds, GM) from last_settings.npz if present
        self._load_consolidated_settings(self._last_settings_path())
        
        # Save state when window closes
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _get_defaults_dir(self):
        """Path to defaults folder for settings files."""
        return os.path.join(os.path.dirname(__file__), "defaults")

    def _last_settings_path(self):
        """Path to consolidated last_settings.npz."""
        return os.path.join(self._get_defaults_dir(), "last_settings.npz")

    def _build_consolidated_save_dict(self):
        """Build a single dict with grid_, seeds_, gm_ prefixes for saving to one npz from self.settings."""
        out = {}
        for k, v in self.settings.get('grid', {}).items():
            out[f'grid_{k}'] = v
        for k, v in self.settings.get('seeds', {}).items():
            out[f'seeds_{k}'] = v
        gm = self.settings.get('gm') or {}
        skip = {'init_mah_d', 'min_stds', 'max_stds', 'unit_id', 'curr_max_prob',
                'debug_init_callback', 'debug_after_com_callback', 'debug_after_gm_callback'}
        for k, v in gm.items():
            if k in skip:
                continue
            out[f'gm_{k}'] = v
        return out

    def _load_consolidated_settings(self, filepath):
        """Load a consolidated settings npz into self.settings (grid, seeds, gm)."""
        if not filepath or not os.path.exists(filepath):
            return
        try:
            data = np.load(filepath, allow_pickle=True)
            keys = list(data.files)
            grid_dict = {}
            seeds_dict = {}
            gm_dict = {}
            for k in keys:
                if k.startswith('grid_'):
                    grid_dict[k[5:]] = data[k]
                elif k.startswith('seeds_'):
                    seeds_dict[k[6:]] = data[k]
                elif k.startswith('gm_'):
                    gm_dict[k[3:]] = data[k]
            if grid_dict:
                self.settings['grid'] = dict(grid_dict)
            if seeds_dict:
                self.settings['seeds'] = dict(seeds_dict)
            if gm_dict:
                self.settings['gm'] = dict(gm_dict)
        except Exception:
            pass

    def _load_settings_dialog(self):
        """Let user select a settings file from the defaults folder (or path) and load it."""
        defaults_dir = self._get_defaults_dir()
        os.makedirs(defaults_dir, exist_ok=True)
        filepath = filedialog.askopenfilename(
            title="Load settings",
            initialdir=defaults_dir,
            filetypes=[("NumPy settings", "*.npz"), ("All files", "*.*")]
        )
        if filepath:
            self._load_consolidated_settings(filepath)

    def _save_last_settings(self):
        """Save current consolidated settings to last_settings.npz (no dialog). Called on window close."""
        save_dict = self._build_consolidated_save_dict()
        if not save_dict:
            return
        try:
            path = self._last_settings_path()
            os.makedirs(self._get_defaults_dir(), exist_ok=True)
            np.savez(path, **save_dict)
        except Exception:
            pass

    def _save_settings_dialog(self):
        """Let user save consolidated settings as last_settings.npz or a named file in defaults folder."""
        defaults_dir = self._get_defaults_dir()
        os.makedirs(defaults_dir, exist_ok=True)
        filepath = filedialog.asksaveasfilename(
            title="Save settings",
            initialdir=defaults_dir,
            initialfile="last_settings.npz",
            defaultextension=".npz",
            filetypes=[("NumPy settings", "*.npz"), ("All files", "*.*")]
        )
        if filepath:
            try:
                save_dict = self._build_consolidated_save_dict()
                if not save_dict:
                    messagebox.showwarning("Save settings", "No settings to save. Use Define Grid, Find Seeds, or Gaussian Parameters first, then Save settings.")
                    return
                np.savez(filepath, **save_dict)
                messagebox.showinfo("Settings", "Settings saved.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {str(e)}")

    def get_spike_ids_for_unit(self, unit_label):
        """Return spike indices assigned to this unit (from unit_id)."""
        if self.unit_id is None:
            return np.array([], dtype=int)
        return np.where(self.unit_id == unit_label)[0]
    
    def _remove_empty_units(self):
        """Remove any unit that has no spikes assigned (e.g. after reassignment to another unit)."""
        to_remove = [ul for ul in list(self.unit_labels) if len(self.get_spike_ids_for_unit(ul)) == 0]
        for ul in to_remove:
            if ul in self.unit_info:
                del self.unit_info[ul]
            if ul in self.unit_labels:
                self.unit_labels.remove(ul)
        if to_remove and self.unit_combo is not None:
            self.unit_combo['values'] = ["new_unit"] + [f"Unit {l}" for l in self.unit_labels]
            self._sync_compare_combo_values()
        if to_remove:
            self.resort_unit_labels_by_position()
    
    def _normalized_dist_for_spike_and_unit(self, spike_id, unit_label):
        """Return (in_bounds, prob_density) for spike vs unit. prob_density from model for gaussian; 0 when in bounds for bounded."""
        if unit_label not in self.unit_info or self.all_properties is None:
            return False, 0.0
        unit = self.unit_info[unit_label]
        row = self.all_properties[spike_id, :].astype(float)
        if unit.is_gaussian() and unit.unit_variables:
            gv = unit.unit_variables[0]
            center = np.asarray(gv.get('center'), dtype=float)
            cov = np.asarray(gv.get('covariance'), dtype=float)
            mah_th = float(gv.get('mah_th', 2.0))
            feat_idx = list(gv.get('feature_indices', []))
            if not feat_idx or center is None or cov is None:
                return False, 0.0
            pts = row[np.asarray(feat_idx)].reshape(1, -1)
            diff = pts - center
            inv_cov = np.linalg.pinv(cov)
            mah_sq = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)[0]
            mahal_d = np.sqrt(max(0, mah_sq))
            density_curve = gv.get('density_curve')
            prob_density = prob_density_from_curve_or_formula(mahal_d, mah_th, density_curve=density_curve)
            prob_density = max(0.0, float(prob_density))
            # Full rule: in_bounds iff inside ellipse and prob_density > current_max_prob (no fallback)
            if self.curr_max_prob is None or spike_id >= len(self.curr_max_prob):
                return False, prob_density
            cmp = float(self.curr_max_prob[spike_id])
            in_bounds = (mahal_d <= mah_th) and (prob_density > cmp)
            return in_bounds, prob_density
        if unit.is_bounded() and unit.unit_variables:
            for bound in unit.unit_variables:
                x_idx, y_idx = bound.property_indices
                pt = np.array([[row[x_idx], row[y_idx]]])
                if not bound.path.contains_points(pt)[0]:
                    return False, 0.0
            return True, np.inf  # inf so assignment (second > curr_max_prob) is always taken
        return False, 0.0

    def _get_hue_array_for_display(self):
        """Return the array to use for density hue coloring based on density_hue_source, or None if not available."""
        if self.all_properties is None:
            return None
        n = self.all_properties.shape[0]
        src = (self.density_hue_source.get() or "Cluster index").strip()
        if src == "Cluster density" and self.cluster_den is not None and len(self.cluster_den) == n:
            return np.asarray(self.cluster_den, dtype=float)
        if src == "Background density" and self.background_den is not None and len(self.background_den) == n:
            return np.asarray(self.background_den, dtype=float)
        if src == "Cluster index" and self.clust_ind is not None and len(self.clust_ind) == n:
            return np.asarray(self.clust_ind, dtype=float)
        return None

    def _require_curr_max_prob(self, n_spikes):
        """Require curr_max_prob exists, has same length as spikes, and valid values (non-negative finite or inf). Raise otherwise."""
        if self.curr_max_prob is None:
            raise ValueError("curr_max_prob is required (same length as spikes, valid values or inf)")
        if len(self.curr_max_prob) != n_spikes:
            raise ValueError(f"curr_max_prob length ({len(self.curr_max_prob)}) must match number of spikes ({n_spikes})")
        arr = np.asarray(self.curr_max_prob, dtype=float)
        if np.any(np.isnan(arr)) or np.any((arr < 0) & np.isfinite(arr)):
            raise ValueError("curr_max_prob must contain only valid (non-negative finite) values or inf")

    def _require_background_den(self, n_spikes):
        """Require background_den exists, has same length as spikes, and valid values (non-negative finite or inf). Raise otherwise."""
        if self.background_den is None:
            raise ValueError("background_den is required (same length as spikes, valid values or inf)")
        if len(self.background_den) != n_spikes:
            raise ValueError(f"background_den length ({len(self.background_den)}) must match number of spikes ({n_spikes})")
        arr = np.asarray(self.background_den, dtype=float)
        if np.any(np.isnan(arr)) or np.any((arr < 0) & np.isfinite(arr)):
            raise ValueError("background_den must contain only valid (non-negative finite) values or inf")

    def _reassign_orphan_spikes_to_other_units(self, orphan_spike_ids, exclude_unit_label):
        """Assign orphan spikes to remaining units: vectorized for Gaussian (by sorted-dim range + batched mahal), fallback loop for bounded."""
        if not len(orphan_spike_ids) or self.unit_id is None or self.all_properties is None:
            return
        orphan_spike_ids = np.asarray(orphan_spike_ids)
        n_spikes = self.all_properties.shape[0]
        self._require_curr_max_prob(n_spikes)
        other_labels = [l for l in self.unit_labels if l != exclude_unit_label]
        if not other_labels:
            return
        n_orphan = len(orphan_spike_ids)
        X_orphan = self.all_properties[orphan_spike_ids, :].astype(float)
        max_prob = self.curr_max_prob[orphan_spike_ids].copy()
        assigned = np.zeros(n_orphan, dtype=bool)
        sorted_global = None
        if self.included_feature_indexes is not None and len(self.included_feature_indexes) > 0:
            incl = list(self.included_feature_indexes)
            settings = self._get_gaussian_model_settings()
            if settings is not None and 0 <= settings.get('sorted_feature_idx', -1) < len(incl):
                sorted_global = int(incl[int(settings['sorted_feature_idx'])])
            elif self.prop_titles and 'y_pos' in self.prop_titles:
                sorted_global = self.prop_titles.index('y_pos')
            else:
                sorted_global = int(incl[0]) if incl else None
        if sorted_global is None and self.prop_titles and 'y_pos' in self.prop_titles:
            sorted_global = self.prop_titles.index('y_pos')
        gaussian_units = [(ul, self.unit_info[ul]) for ul in other_labels if self.unit_info[ul].is_gaussian() and self.unit_info[ul].unit_variables]
        bounded_units = [(ul, self.unit_info[ul]) for ul in other_labels if self.unit_info[ul].is_bounded() and self.unit_info[ul].unit_variables]
        if sorted_global is not None and gaussian_units:
            pos_sorted = X_orphan[:, sorted_global]
            pos_min, pos_max = float(np.min(pos_sorted)), float(np.max(pos_sorted))
            max_std_sorted = 0.0
            for ul, unit in gaussian_units:
                gv = unit.unit_variables[0]
                feat_idx = list(gv.get('feature_indices', []))
                cov = np.asarray(gv.get('covariance'), dtype=float)
                if sorted_global in feat_idx and cov.size > 0:
                    i = feat_idx.index(sorted_global)
                    if i < cov.shape[0]:
                        std = np.sqrt(max(0, float(cov[i, i])))
                        max_std_sorted = max(max_std_sorted, std)
            if max_std_sorted <= 0:
                max_std_sorted = 1.0
            margin = 3.0 * max_std_sorted
            range_lo, range_hi = pos_min - margin, pos_max + margin
            for ul, unit in gaussian_units:
                gv = unit.unit_variables[0]
                center = np.asarray(gv.get('center'), dtype=float)
                cov = np.asarray(gv.get('covariance'), dtype=float)
                mah_th = float(gv.get('mah_th', 2.0))
                feat_idx = list(gv.get('feature_indices', []))
                if not feat_idx or center is None or cov is None:
                    continue
                if sorted_global not in feat_idx:
                    continue
                center_sorted = float(center[feat_idx.index(sorted_global)])
                if center_sorted < range_lo or center_sorted > range_hi:
                    continue
                X = X_orphan[:, np.asarray(feat_idx)]
                diff = X - center
                inv_cov = np.linalg.pinv(cov)
                mah_sq = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
                mah_sq = np.maximum(mah_sq, 0.0)
                mah_dist = np.sqrt(mah_sq)
                density_curve = gv.get('density_curve')
                prob_density = prob_density_from_curve_or_formula(mah_dist, mah_th, density_curve=density_curve)
                in_bounds = mah_dist <= mah_th
                improves = in_bounds & (prob_density > max_prob)
                if np.any(improves):
                    max_prob[improves] = prob_density[improves]
                    assigned[improves] = True
                    self.unit_id[orphan_spike_ids[improves]] = ul
        for ul, unit in bounded_units:
            for i, sid in enumerate(orphan_spike_ids):
                if assigned[i]:
                    continue
                in_bounds, prob_density = self._normalized_dist_for_spike_and_unit(int(sid), ul)
                if in_bounds and prob_density > max_prob[i]:
                    max_prob[i] = prob_density
                    assigned[i] = True
                    self.unit_id[sid] = ul
        self.curr_max_prob[orphan_spike_ids] = max_prob
    
    def resort_unit_labels_by_position(self):
        """Renumber unit labels so that label 0 has smallest mean_y_pos, label 1 next, etc."""
        if not self.unit_labels or self.unit_id is None:
            return
        ordered = sorted(self.unit_labels, key=lambda l: self.unit_info[l].mean_y_pos)
        old_to_new = {old: i for i, old in enumerate(ordered)}
        for sid in range(len(self.unit_id)):
            old_l = self.unit_id[sid]
            if old_l in old_to_new:
                self.unit_id[sid] = old_to_new[old_l]
        new_info = {old_to_new[old]: self.unit_info[old] for old in ordered}
        self.unit_info.clear()
        self.unit_info.update(new_info)
        self.unit_labels = list(range(len(ordered)))
        if self.unit_combo is not None:
            self.unit_combo['values'] = ["new_unit"] + [f"Unit {l}" for l in self.unit_labels]
            self._sync_compare_combo_values()
        self.selected_unit.set("new_unit")
        self.compare_unit.set("new_unit")
        self.on_unit_selection_changed()
        self.update_all_plots()
    
    def get_state_file_path(self):
        """Get path to state file"""
        defaults_dir = os.path.join(os.path.dirname(__file__), "defaults")
        os.makedirs(defaults_dir, exist_ok=True)
        return os.path.join(defaults_dir, "visualizer_state.npz")
    
    def save_state(self):
        """Save current state (folder path and feature selections)"""
        try:
            state_file = self.get_state_file_path()
            state_dict = {}
            
            # Save last folder
            if self.data_folder:
                state_dict['last_folder'] = self.data_folder
            
            # Save feature selections for each plot
            # Save as separate arrays for easier loading
            x_features = []
            y_features = []
            for feat_sel in self.feature_selections:
                x_val = feat_sel['x'].get()
                y_val = feat_sel['y'].get()
                x_features.append(x_val if x_val else '')
                y_features.append(y_val if y_val else '')
            
            state_dict['x_features'] = np.array(x_features, dtype=object)
            state_dict['y_features'] = np.array(y_features, dtype=object)
            state_dict['sort_feature_idx'] = int(self.sort_feature_idx)
            np.savez(state_file, **state_dict, allow_pickle=True)
        except Exception as e:
            # Silently fail - state saving is not critical
            pass
    
    def load_state(self):
        """Load saved state"""
        try:
            state_file = self.get_state_file_path()
            if not os.path.exists(state_file):
                return
            
            state_data = np.load(state_file, allow_pickle=True)
            
            # Load last folder
            if 'last_folder' in state_data:
                last_folder = str(state_data['last_folder'])
                if os.path.exists(last_folder):
                    self.path_var.set(last_folder)
                    self.data_folder = last_folder
            if 'sort_feature_idx' in state_data:
                self.sort_feature_idx = int(state_data['sort_feature_idx'])

            # Load feature selections (will be applied when data is loaded)
            # Try new format first (separate x/y arrays)
            if 'x_features' in state_data and 'y_features' in state_data:
                x_features = state_data['x_features']
                y_features = state_data['y_features']
                # Convert to list
                if hasattr(x_features, 'tolist'):
                    x_features = x_features.tolist()
                if hasattr(y_features, 'tolist'):
                    y_features = y_features.tolist()
                
                # Convert to list of dicts
                self.saved_feature_selections = []
                for i in range(len(x_features)):
                    self.saved_feature_selections.append({
                        'x': str(x_features[i]) if i < len(x_features) else '',
                        'y': str(y_features[i]) if i < len(y_features) else ''
                    })
            # Fallback to old format
            elif 'feature_selections' in state_data:
                saved_selections = state_data['feature_selections']
                # Handle numpy array/object array conversion
                if hasattr(saved_selections, 'item'):
                    saved_selections = saved_selections.item()
                elif hasattr(saved_selections, 'tolist'):
                    saved_selections = saved_selections.tolist()
                
                # Convert to list of dicts if needed
                if isinstance(saved_selections, (list, np.ndarray)):
                    converted_selections = []
                    for sel in saved_selections:
                        if isinstance(sel, dict):
                            converted_selections.append(sel)
                        elif hasattr(sel, 'item'):
                            converted_selections.append(sel.item())
                        else:
                            # Try to convert numpy array element
                            converted_selections.append({
                                'x': str(sel.get('x', '')) if hasattr(sel, 'get') else '',
                                'y': str(sel.get('y', '')) if hasattr(sel, 'get') else ''
                            })
                    self.saved_feature_selections = converted_selections
                else:
                    self.saved_feature_selections = saved_selections
            else:
                self.saved_feature_selections = None
                
        except Exception as e:
            # Silently fail - state loading is not critical
            self.saved_feature_selections = None
    
    def on_closing(self):
        """Handle window closing - save state and exit cleanly so process terminates."""
        self.save_state()
        self._save_last_settings()
        try:
            plt.close('all')
        except Exception:
            pass
        self.root.quit()
        self.root.destroy()
    
    def _recompute_gm_units_data_range_for_sort(self):
        """If any GM unit's sort_feature_idx != current sort dimension, recompute its data_range and sort_range for the new sort feature."""
        if self.all_properties is None or not self.unit_labels:
            return
        current_sort = self.sort_feature_idx
        sorted_col = self.all_properties[:, current_sort].astype(float)
        for ul in self.unit_labels:
            unit = self.unit_info.get(ul)
            if unit is None or not unit.is_gaussian() or not unit.unit_variables:
                continue
            gv = unit.unit_variables[0]
            unit_sort_idx = gv.get('sort_feature_idx')
            if unit_sort_idx is None:
                continue
            if int(unit_sort_idx) == current_sort:
                continue
            center = np.asarray(gv.get('center'), dtype=float)
            cov = np.asarray(gv.get('covariance'), dtype=float)
            mah_th = float(gv.get('mah_th', 2.0))
            feat_idx = list(gv.get('feature_indices', []))
            if not feat_idx or current_sort not in feat_idx:
                continue
            j = feat_idx.index(current_sort)
            center_sorted = float(center[j])
            std = np.sqrt(max(0.0, float(cov[j, j])))
            if std <= 0:
                std = 1.0
            sort_lo = center_sorted - std * mah_th
            sort_hi = center_sorted + std * mah_th
            first_idx = np.searchsorted(sorted_col, sort_lo, side='left')
            last_idx = np.searchsorted(sorted_col, sort_hi, side='right') - 1
            last_idx = min(max(last_idx, first_idx), len(sorted_col) - 1)
            first_idx = min(first_idx, last_idx)
            gv['data_range'] = (int(first_idx), int(last_idx))
            gv['sort_range'] = (sort_lo, sort_hi)
            gv['sort_feature_idx'] = current_sort

    def _on_sort_feature_changed(self, event=None):
        """Re-sort data by the newly selected feature."""
        if self.all_properties is None or self.prop_titles is None or self.sort_feature_combo is None:
            return
        sel = self.sort_feature_combo.get()
        if sel not in self.prop_titles:
            sorted_name = self._sorted_feature_name()
            if sorted_name:
                self.sort_feature_combo.set(sorted_name)
            return
        new_idx = self.prop_titles.index(sel)
        if new_idx == self.sort_feature_idx:
            return
        self.sort_feature_idx = new_idx
        sort_indices = np.argsort(self.all_properties[:, new_idx])
        self.all_properties = self.all_properties[sort_indices]
        if self.clust_ind is not None and len(self.clust_ind) == len(sort_indices):
            self.clust_ind = self.clust_ind[sort_indices]
        if self.unit_id is not None and len(self.unit_id) == len(sort_indices):
            self.unit_id = self.unit_id[sort_indices]
        n_spikes = len(sort_indices)
        self._require_curr_max_prob(n_spikes)
        self.curr_max_prob = self.curr_max_prob[sort_indices]
        self._recompute_gm_units_data_range_for_sort()
        self.update_all_plots()

    def _sorted_feature_name(self):
        """Return the sorted feature name (default: index 0 in properties tensor)."""
        if self.prop_titles is None or len(self.prop_titles) == 0:
            return None
        idx = getattr(self, 'sort_feature_idx', 0)
        if idx < 0 or idx >= len(self.prop_titles):
            idx = 0
        return self.prop_titles[idx]

    def browse_folder(self):
        """Browse for data folder; start two levels above current path. On selection, load data."""
        current = (self.path_var.get() or "").strip()
        if current and os.path.exists(current):
            initial = os.path.normpath(current)
            for _ in range(2):
                parent = os.path.dirname(initial)
                if not parent or parent == initial:
                    break
                initial = parent
            initialdir = initial
        else:
            initialdir = os.getcwd()
        folder = filedialog.askdirectory(title="Select folder with spike properties", initialdir=initialdir)
        if folder:
            self.path_var.set(folder)
            self.data_folder = folder
            self.save_state()
            self.load_data()
    
    def load_data(self):
        """Load spike properties data"""
        folder = self.path_var.get()
        if not folder or not os.path.exists(folder):
            messagebox.showerror("Error", "Please select a valid folder")
            return
        
        spike_prop_dir = os.path.join(folder, 'spike_prop')
        if not os.path.exists(spike_prop_dir):
            messagebox.showerror("Error", "spike_prop directory not found")
            return
        
        # Load properties files
        spike_prop_files = sorted(glob.glob(os.path.join(spike_prop_dir, 'batch_*_spike_properties.npz')))
        if len(spike_prop_files) == 0:
            messagebox.showerror("Error", "No spike property files found")
            return
        
        # Load and concatenate
        all_properties_list = []
        for prop_file in spike_prop_files:
            data = np.load(prop_file)
            properties = data['Properties']
            all_properties_list.append(properties)
        
        self.all_properties = np.concatenate(all_properties_list, axis=0)
        n_features = self.all_properties.shape[1]

        # Load prop titles (needed before sort so we can show sort combo)
        if 'PropTitles' in data:
            self.prop_titles = list(data['PropTitles'])
        else:
            self.prop_titles = [f"Feature {i}" for i in range(n_features)]

        # Sort by the chosen feature (load-time setting). Default to index 0 if out of range.
        self.sort_feature_idx = min(max(0, self.sort_feature_idx), n_features - 1) if n_features > 0 else 0
        sort_col = self.sort_feature_idx
        sort_indices = np.argsort(self.all_properties[:, sort_col])
        self.all_properties = self.all_properties[sort_indices]

        # Populate sort-by combo
        if self.sort_feature_combo is not None and self.prop_titles:
            self.sort_feature_combo['values'] = self.prop_titles
            self.sort_feature_combo.set(self.prop_titles[self.sort_feature_idx])

        # Reset experiment-specific state so a second load doesn't keep old experiment's data
        self.cluster_den = None
        self.background_den = None
        self.clust_ind = None
        self.included_feature_indexes = None
        self.bounds = []
        self.seeds = []
        self.gaussian_models = []
        
        # Update feature dropdowns
        for plot_idx, feat_sel in enumerate(self.feature_selections):
            feat_sel['x_combo']['values'] = self.prop_titles
            feat_sel['y_combo']['values'] = self.prop_titles
            
            # Try to restore saved feature selections
            restored_x = False
            restored_y = False
            if self.saved_feature_selections is not None and plot_idx < len(self.saved_feature_selections):
                try:
                    sel_dict = self.saved_feature_selections[plot_idx]
                    # Handle different data types
                    if isinstance(sel_dict, dict):
                        saved_x = sel_dict.get('x', '')
                        saved_y = sel_dict.get('y', '')
                    elif hasattr(sel_dict, 'item'):
                        sel_dict = sel_dict.item()
                        saved_x = sel_dict.get('x', '') if isinstance(sel_dict, dict) else ''
                        saved_y = sel_dict.get('y', '') if isinstance(sel_dict, dict) else ''
                    else:
                        saved_x = ''
                        saved_y = ''
                    
                    # Convert to string and restore if valid; if not present use sorted feature (index 0 default)
                    saved_x = str(saved_x).strip() if saved_x else ''
                    saved_y = str(saved_y).strip() if saved_y else ''
                    x_to_set = saved_x if (saved_x and saved_x in self.prop_titles) else self._sorted_feature_name()
                    y_to_set = saved_y if (saved_y and saved_y in self.prop_titles) else self._sorted_feature_name()
                    if x_to_set:
                        feat_sel['x'].set(x_to_set)
                        restored_x = True
                    if y_to_set:
                        feat_sel['y'].set(y_to_set)
                        restored_y = True
                except Exception as e:
                    # If restoration fails, continue with defaults
                    pass
            
            restored = restored_x or restored_y
            
            # Set default values if not restored and not already set (use sorted feature when missing)
            if len(self.prop_titles) > 0 and not restored:
                current_x = feat_sel['x'].get()
                current_y = feat_sel['y'].get()
                sorted_name = self._sorted_feature_name()
                if not current_x or current_x not in self.prop_titles:
                    feat_sel['x'].set(sorted_name or self.prop_titles[min(plot_idx, len(self.prop_titles) - 1)])
                if not current_y or current_y not in self.prop_titles:
                    x_val = feat_sel['x'].get()
                    x_idx = self.prop_titles.index(x_val) if x_val in self.prop_titles else 0
                    y_idx = min((x_idx + 1) % len(self.prop_titles), len(self.prop_titles) - 1)
                    if y_idx == x_idx and len(self.prop_titles) > 1:
                        y_idx = (y_idx + 1) % len(self.prop_titles)
                    feat_sel['y'].set(self.prop_titles[y_idx])
        
        # Initialize display ranges for restored features (without saving state)
        for plot_idx in range(4):
            if plot_idx < len(self.feature_selections):
                x_feat = self.feature_selections[plot_idx]['x'].get()
                y_feat = self.feature_selections[plot_idx]['y'].get()
                if x_feat and y_feat and x_feat in self.prop_titles and y_feat in self.prop_titles:
                    x_min, x_max = self.get_feature_range(x_feat)
                    y_min, y_max = self.get_feature_range(y_feat)
                    if x_min is not None and y_min is not None:
                        x_range = x_max - x_min
                        y_range = y_max - y_min
                        # Initialize global display ranges if not exist
                        if x_feat not in self.global_display_ranges:
                            self.global_display_ranges[x_feat] = {
                                'center': tk.DoubleVar(),
                                'range': tk.DoubleVar()
                            }
                            center = (x_min + x_max) / 2.0
                            range_val = x_range * 1.1
                            self.global_display_ranges[x_feat]['center'].set(center)
                            self.global_display_ranges[x_feat]['range'].set(range_val)
                        if y_feat not in self.global_display_ranges:
                            self.global_display_ranges[y_feat] = {
                                'center': tk.DoubleVar(),
                                'range': tk.DoubleVar()
                            }
                            center = (y_min + y_max) / 2.0
                            range_val = y_range * 1.1
                            self.global_display_ranges[y_feat]['center'].set(center)
                            self.global_display_ranges[y_feat]['range'].set(range_val)
        
        # Save state after loading data (saves the restored selections)
        self.save_state()
        
        # Try to load densities
        densities_file = os.path.join(folder, 'densities.npz')
        if os.path.exists(densities_file):
            densities_data = np.load(densities_file, allow_pickle=True)
            
            # Load density arrays WITHOUT sorting - keep original order
            if 'cluster_den' in densities_data:
                self.cluster_den = np.array(densities_data['cluster_den'], copy=False)
            if 'background_den' in densities_data:
                self.background_den = np.array(densities_data['background_den'], copy=False)
            if 'clust_ind' in densities_data:
                # Load clust_ind without sorting - preserve original order
                self.clust_ind = np.array(densities_data['clust_ind'], copy=False)
                # Ensure it has the same length as all_properties
                if self.clust_ind.shape[0] != self.all_properties.shape[0]:
                    messagebox.showwarning("Warning", 
                        f"clust_ind length ({self.clust_ind.shape[0]}) doesn't match properties length ({self.all_properties.shape[0]}). "
                        "clust_ind will be truncated or padded to match.")
                    if self.clust_ind.shape[0] > self.all_properties.shape[0]:
                        self.clust_ind = self.clust_ind[:self.all_properties.shape[0]]
                    else:
                        # Pad with zeros if shorter
                        padding = np.zeros(self.all_properties.shape[0] - self.clust_ind.shape[0])
                        self.clust_ind = np.concatenate([self.clust_ind, padding])
            if 'included_feature_indexes' in densities_data:
                self.included_feature_indexes = densities_data['included_feature_indexes'].tolist()
        
        # Try to load bounds
        bounds_file = os.path.join(folder, 'bounds.npz')
        if os.path.exists(bounds_file):
            bounds_data = np.load(bounds_file, allow_pickle=True)
            if 'bounds' in bounds_data:
                bounds_list = bounds_data['bounds'].item() if hasattr(bounds_data['bounds'], 'item') else bounds_data['bounds']
                self.bounds = []
                for b in bounds_list:
                    bound = Bound(b['property_indices'], b['vertices'], b['unit_label'])
                    self.bounds.append(bound)
                if len(self.bounds) > 0:
                    self.next_unit_label = max(b.unit_label for b in self.bounds) + 1
        
        # Try to load unit assignments from sorting.npz
        sorting_file = os.path.join(folder, 'sorting.npz')
        if os.path.exists(sorting_file):
            sorting_data = np.load(sorting_file, allow_pickle=True)
            if 'unit_id' in sorting_data:
                # Load unit_id WITHOUT sorting - keep original order
                unit_id_loaded = np.array(sorting_data['unit_id'], copy=False)
                if unit_id_loaded.shape[0] == self.all_properties.shape[0]:
                    self.unit_id = unit_id_loaded
                else:
                    messagebox.showwarning("Warning", 
                        f"unit_id length ({unit_id_loaded.shape[0]}) doesn't match properties length ({self.all_properties.shape[0]}). "
                        "unit_id will be initialized to -1.")
                    self.unit_id = np.full(self.all_properties.shape[0], -1, dtype=int)
                curr_max_prob_key = 'curr_max_prob' if 'curr_max_prob' in sorting_data else 'dist_to_center'
                if curr_max_prob_key in sorting_data:
                    cmp = np.array(sorting_data[curr_max_prob_key], dtype=float, copy=True)
                    if cmp.shape[0] != self.all_properties.shape[0]:
                        raise ValueError(f"curr_max_prob length ({cmp.shape[0]}) must match properties length ({self.all_properties.shape[0]})")
                    if np.any(np.isnan(cmp)) or np.any((cmp < 0) & np.isfinite(cmp)):
                        raise ValueError("curr_max_prob must contain only valid (non-negative finite) values or inf")
                    self.curr_max_prob = cmp
                elif self.background_den is not None and len(self.background_den) == self.all_properties.shape[0]:
                    # Fallback: sorting file has no curr_max_prob (e.g. old save); init from loaded densities
                    self.curr_max_prob = np.where(
                        self.background_den > 0,
                        self.background_den.astype(float),
                        np.inf
                    )
                else:
                    raise ValueError("sorting file must contain curr_max_prob (or dist_to_center), or load densities first so curr_max_prob can be initialized from background_den")
            else:
                # Initialize unit_id if not found; curr_max_prob must be set by running density before using GM
                self.unit_id = np.full(self.all_properties.shape[0], -1, dtype=int)
                self.curr_max_prob = None
            
            if 'unit_labels' in sorting_data:
                self.unit_labels = list(sorting_data['unit_labels'])
                # Convert to int if needed (might be float64 from npz)
                self.unit_labels = [int(l) for l in self.unit_labels]
            else:
                self.unit_labels = []
            
            # Initialize unit_info
            self.unit_info = {}
            if 'unit_info' in sorting_data:
                unit_info_loaded = sorting_data['unit_info'].item() if hasattr(sorting_data['unit_info'], 'item') else sorting_data['unit_info']
                if isinstance(unit_info_loaded, dict):
                    for unit_label, unit_data in unit_info_loaded.items():
                        # Reconstruct Unit objects from saved data
                        if isinstance(unit_data, dict):
                            utype = unit_data.get('unit_type', 'bounded')
                            if utype == 'gaussian' and 'unit_variables' in unit_data:
                                unit_variables = list(unit_data['unit_variables'])
                            else:
                                unit_variables = []
                                if 'unit_variables' in unit_data:
                                    for b_data in unit_data['unit_variables']:
                                        bound = Bound(b_data['property_indices'], b_data['vertices'], b_data['unit_label'])
                                        unit_variables.append(bound)
                            unit = Unit(
                                unit_variables=unit_variables,
                                mean_y_pos=unit_data.get('mean_y_pos', 0.0)
                            )
                            self.unit_info[int(unit_label)] = unit
                        elif hasattr(unit_data, 'unit_type'):  # Already a Unit object
                            self.unit_info[int(unit_label)] = unit_data
            
            # Update dropdown
            if len(self.unit_labels) > 0:
                self.unit_combo['values'] = ["new_unit"] + [f"Unit {l}" for l in self.unit_labels]
            else:
                self.unit_combo['values'] = ["new_unit"]
            self._sync_compare_combo_values()
            # Set to new_unit
            self.selected_unit.set("new_unit")
            self.compare_unit.set("new_unit")
        else:
            # Initialize empty unit structures; curr_max_prob and background_den set when density is run
            self.unit_id = np.full(self.all_properties.shape[0], -1, dtype=int)
            self.curr_max_prob = None
            self.unit_info = {}
            self.unit_labels = []
            self.unit_combo['values'] = ["new_unit"]
            self._sync_compare_combo_values()
            self.selected_unit.set("new_unit")
            self.compare_unit.set("new_unit")
        
        # If we have densities but curr_max_prob was never set (e.g. no sorting file), initialize from background_den
        n_spikes = self.all_properties.shape[0]
        if self.background_den is not None and len(self.background_den) == n_spikes and self.curr_max_prob is None:
            self.curr_max_prob = np.where(
                self.background_den > 0,
                self.background_den.astype(float),
                np.inf
            )
        
        # Initialize button states
        self.on_unit_selection_changed()
        
        # Update UI (this will recreate buttons if needed)
        self.on_sorting_type_changed()
        
        # Enable plot sorted checkbox if units exist
        if len(self.unit_labels) > 0 and self.sorting_type.get() == "Cluster":
            self.plot_sorted_check.config(state=tk.NORMAL)
        
        # Enable buttons if they exist
        if self.calc_densities_btn is not None:
            self.calc_densities_btn.config(state=tk.NORMAL)
        if self.find_seeds_btn is not None and self.cluster_den is not None:
            self.find_seeds_btn.config(state=tk.NORMAL)
        # Auto-load all settings from last_settings.npz so grid/seeds/GM match this dataset
        self._load_consolidated_settings(self._last_settings_path())
        # Data loaded successfully: reset all graphs to home (full range)
        self.reset_all_zoom()
    
    def save_data(self):
        """Save current state"""
        if self.data_folder is None:
            folder = filedialog.askdirectory(title="Select folder to save data")
            if not folder:
                return
            self.data_folder = folder
            self.path_var.set(folder)
        
        # Save bounds
        if len(self.bounds) > 0:
            bounds_data = []
            for bound in self.bounds:
                bounds_data.append({
                    'property_indices': bound.property_indices,
                    'vertices': bound.vertices,
                    'unit_label': bound.unit_label
                })
            np.savez(os.path.join(self.data_folder, 'bounds.npz'), bounds=bounds_data)
        
        # Save unit assignments to sorting.npz
        if self.unit_id is not None:
            # Prepare unit_info for saving (convert Unit objects to dictionaries)
            unit_info_to_save = {}
            for unit_label, unit in self.unit_info.items():
                if unit.is_gaussian():
                    unit_variables_dict = list(unit.unit_variables)
                else:
                    unit_variables_dict = []
                    for bound in unit.unit_variables:
                        unit_variables_dict.append({
                            'property_indices': bound.property_indices,
                            'vertices': bound.vertices,
                            'unit_label': bound.unit_label
                        })
                unit_info_to_save[unit_label] = {
                    'unit_type': 'gaussian' if unit.is_gaussian() else 'bounded',
                    'unit_variables': unit_variables_dict,
                    'mean_y_pos': unit.mean_y_pos
                }
            
            # Reorganize unit_labels to be incremental by mean_y_pos (0, 1, 2...)
            # This was requested in the conversation summary
            if len(self.unit_labels) > 0:
                # Sort by mean_y_pos
                sorted_labels = sorted(self.unit_labels, key=lambda l: self.unit_info[l].mean_y_pos)
                # Create mapping from old labels to new incremental labels
                label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}
                
                # Reorganize unit_id
                unit_id_reorganized = self.unit_id.copy()
                for old_label, new_label in label_mapping.items():
                    unit_id_reorganized[self.unit_id == old_label] = new_label
                
                # Reorganize unit_info
                unit_info_reorganized = {}
                for old_label, new_label in label_mapping.items():
                    unit_info_reorganized[new_label] = unit_info_to_save[old_label]
                
                # New unit_labels are just 0, 1, 2, ...
                unit_labels_reorganized = list(range(len(sorted_labels)))
            else:
                unit_id_reorganized = self.unit_id
                unit_info_reorganized = unit_info_to_save
                unit_labels_reorganized = self.unit_labels
            
            save_dict = {
                'unit_id': unit_id_reorganized,
                'unit_info': unit_info_reorganized,
                'unit_labels': np.array(unit_labels_reorganized),
            }
            n_save = len(unit_id_reorganized)
            self._require_curr_max_prob(n_save)
            save_dict['curr_max_prob'] = self.curr_max_prob
            np.savez(os.path.join(self.data_folder, 'sorting.npz'), **save_dict)
        
        # Data saved successfully
    
    def on_sorting_type_changed(self):
        """Handle sorting type change - show/hide appropriate buttons"""
        # Clear buttons frame
        for widget in self.buttons_frame.winfo_children():
            widget.destroy()
        
        # Reset button references (calc_densities_btn is fixed in toolbar, not recreated)
        self.find_seeds_btn = None
        self.gaussian_model_settings_btn = None
        self.compute_gaussian_models_button = None
        self.define_bounds_btn = None
        self.clear_bounds_btn = None
        self.assign_units_btn = None
        self.compute_gaussian_models_cluster_btn = None
        self.cluster_seeds_frame = None
        self.cluster_seeds_combo = None
        self.delete_cluster_seed_btn = None
        
        sorting_type = self.sorting_type.get()
        
        if sorting_type == "Cluster":
            self.calc_densities_btn.config(state=tk.NORMAL if self.all_properties is not None else tk.DISABLED)
            # Cluster seed mode buttons (Clear, Define seed cluster, Compute Gaussian Models)
            self.clear_bounds_btn = ttk.Button(self.buttons_frame, text="Clear Bounds", 
                                              command=self.clear_bounds, state=tk.NORMAL if len(self.bounds) > 0 else tk.DISABLED)
            self.clear_bounds_btn.pack(pady=2, fill=tk.X)
            
            self.assign_units_btn = ttk.Button(self.buttons_frame, text="Define seed cluster", 
                                               command=self.define_seed_cluster, state=tk.NORMAL if len(self.bounds) > 0 else tk.DISABLED)
            self.assign_units_btn.pack(pady=2, fill=tk.X)
            
            self.gaussian_model_settings_btn = ttk.Button(self.buttons_frame, text="Gaussian Model\nParameters", 
                                                         command=self._open_gaussian_model_settings_window,
                                                         state=tk.NORMAL if self.all_properties is not None else tk.DISABLED)
            self.gaussian_model_settings_btn.pack(pady=2, fill=tk.X)
            
            self.compute_gaussian_models_cluster_btn = ttk.Button(self.buttons_frame, text="Compute Gaussian\nModels", 
                                                                  command=self.compute_gaussian_models_from_cluster_seeds,
                                                                  state=tk.NORMAL if getattr(self, 'cluster_seeds', None) and len(self.cluster_seeds) > 0 else tk.DISABLED)
            self.compute_gaussian_models_cluster_btn.pack(pady=2, fill=tk.X)
            
            # Cluster seeds dropdown and Delete (only in Cluster mode)
            self.cluster_seeds_frame = ttk.LabelFrame(self.buttons_frame, text="Cluster seeds")
            self.cluster_seeds_frame.pack(pady=5, fill=tk.X)
            self.cluster_seeds_combo = ttk.Combobox(self.cluster_seeds_frame, textvariable=self.selected_cluster_seed,
                                                    state="readonly", width=18)
            self.cluster_seeds_combo.pack(pady=2, padx=5, fill=tk.X)
            self.cluster_seeds_combo.bind("<<ComboboxSelected>>", lambda e: self.on_cluster_seed_selection_changed())
            self.delete_cluster_seed_btn = ttk.Button(self.cluster_seeds_frame, text="Delete", 
                                                     command=self.delete_selected_cluster_seed, state=tk.DISABLED)
            self.delete_cluster_seed_btn.pack(pady=2, fill=tk.X)
            self._refresh_cluster_seeds_ui()
            
            # Enable plot sorted if units exist, otherwise disable
            if len(self.unit_labels) > 0:
                self.plot_sorted_check.config(state=tk.NORMAL)
            else:
                self.plot_sorted_check.config(state=tk.DISABLED)
            if getattr(self, 'refresh_gm_assignment_btn', None) is not None:
                self.refresh_gm_assignment_btn.pack_forget()
            
            # Show bound buttons, hide Add seed buttons in feature frame
            for plot_idx in range(4):
                if self.make_bound_buttons[plot_idx] is not None:
                    self.make_bound_buttons[plot_idx].pack(side=tk.LEFT, padx=2)
                if self.delete_bound_buttons[plot_idx] is not None:
                    self.delete_bound_buttons[plot_idx].pack(side=tk.LEFT, padx=2)
                if getattr(self, 'add_seed_buttons', None) and self.add_seed_buttons[plot_idx] is not None:
                    self.add_seed_buttons[plot_idx].pack_forget()
            self._exit_add_seed_mode()
            
        elif sorting_type == "Single seed":
            # Gaussian model mode buttons
            self.calc_densities_btn.config(state=tk.NORMAL if self.all_properties is not None else tk.DISABLED)
            self.find_seeds_btn = ttk.Button(self.buttons_frame, text="Find Local\nMaxima Seeds", 
                                            command=self.find_local_maxima_seeds, 
                                            state=tk.NORMAL if self.cluster_den is not None else tk.DISABLED)
            self.find_seeds_btn.pack(pady=2, fill=tk.X)
            
            self.gaussian_model_settings_btn = ttk.Button(self.buttons_frame, text="Gaussian Model\nParameters", 
                                                         command=self._open_gaussian_model_settings_window,
                                                         state=tk.NORMAL if self.all_properties is not None else tk.DISABLED)
            self.gaussian_model_settings_btn.pack(pady=2, fill=tk.X)
            
            self.compute_gaussian_models_button = ttk.Button(self.buttons_frame, text="Compute Gaussian\nModels", 
                                                            command=self.compute_gaussian_models, 
                                                            state=tk.NORMAL if (self.seeds is not None and len(self.seeds) > 0) else tk.DISABLED)
            self.compute_gaussian_models_button.pack(pady=2, fill=tk.X)
            
            self.plot_sorted_check.config(state=tk.NORMAL)
            if getattr(self, 'refresh_gm_assignment_btn', None) is not None:
                self.refresh_gm_assignment_btn.pack(pady=2, fill=tk.X)
            
            # Hide bound buttons, show Add seed buttons in feature frame
            for plot_idx in range(4):
                if self.make_bound_buttons[plot_idx] is not None:
                    self.make_bound_buttons[plot_idx].pack_forget()
                if self.delete_bound_buttons[plot_idx] is not None:
                    self.delete_bound_buttons[plot_idx].pack_forget()
                if getattr(self, 'add_seed_buttons', None) and self.add_seed_buttons[plot_idx] is not None:
                    self.add_seed_buttons[plot_idx].pack(side=tk.LEFT, padx=2)
        
        # Update plots
        for plot_idx in range(4):
            self.update_plot(plot_idx)
    
    def on_feature_change(self, plot_idx):
        """Handle feature selection change"""
        # Clear any existing polygon selector
        if self.polygon_selectors[plot_idx] is not None:
            self.polygon_selectors[plot_idx].set_active(False)
            self.polygon_selectors[plot_idx] = None
        
        # Clear focus polygon
        self.focus_polygons[plot_idx] = None
        
        # Initialize display ranges for new features
        x_feat = self.feature_selections[plot_idx]['x'].get()
        y_feat = self.feature_selections[plot_idx]['y'].get()
        if x_feat and y_feat and x_feat in self.prop_titles and y_feat in self.prop_titles:
            x_min, x_max = self.get_feature_range(x_feat)
            y_min, y_max = self.get_feature_range(y_feat)
            
            if x_min is not None and y_min is not None:
                x_range = x_max - x_min
                y_range = y_max - y_min
                
                # Initialize global display ranges if not exist
                if x_feat not in self.global_display_ranges:
                    self.global_display_ranges[x_feat] = {
                        'center': tk.DoubleVar(),
                        'range': tk.DoubleVar()
                    }
                    center = (x_min + x_max) / 2.0
                    range_val = x_range * 1.1
                    self.global_display_ranges[x_feat]['center'].set(center)
                    self.global_display_ranges[x_feat]['range'].set(range_val)
                
                if y_feat not in self.global_display_ranges:
                    self.global_display_ranges[y_feat] = {
                        'center': tk.DoubleVar(),
                        'range': tk.DoubleVar()
                    }
                    center = (y_min + y_max) / 2.0
                    range_val = y_range * 1.1
                    self.global_display_ranges[y_feat]['center'].set(center)
                    self.global_display_ranges[y_feat]['range'].set(range_val)
        
        self.update_plot(plot_idx)
        
        # Save state when features change
        self.save_state()
    
    def get_feature_range(self, feat_name):
        """Get min and max values for a feature"""
        if self.all_properties is None or feat_name not in self.prop_titles:
            return None, None
        feat_idx = self.prop_titles.index(feat_name)
        feat_data = self.all_properties[:, feat_idx]
        return np.min(feat_data), np.max(feat_data)
    
    def get_dots_in_display_range(self):
        """Get indices of dots within current display ranges"""
        if self.all_properties is None:
            return None
        
        mask = np.ones(self.all_properties.shape[0], dtype=bool)
        
        for feat_name, range_dict in self.global_display_ranges.items():
            if feat_name not in self.prop_titles:
                continue
            
            feat_idx = self.prop_titles.index(feat_name)
            center = range_dict['center'].get()
            range_val = range_dict['range'].get()
            
            feat_min = center - range_val / 2.0
            feat_max = center + range_val / 2.0
            
            feat_data = self.all_properties[:, feat_idx]
            mask &= (feat_data >= feat_min) & (feat_data <= feat_max)
        
        return np.where(mask)[0]
    
    def get_focus_mask(self):
        """Get mask for focused (bounded) spikes - must be in ALL boundaries"""
        if len(self.bounds) == 0:
            return None
        
        focus_mask = np.ones(self.all_properties.shape[0], dtype=bool)
        
        for bound in self.bounds:
            x_idx, y_idx = bound.property_indices
            x_data = self.all_properties[:, x_idx]
            y_data = self.all_properties[:, y_idx]
            
            points = np.column_stack([x_data, y_data])
            in_bound = bound.path.contains_points(points)
            focus_mask &= in_bound  # Must be in ALL bounds
        
        return focus_mask
    
    def update_plot(self, plot_idx):
        """Update a single plot"""
        if self.all_properties is None:
            return
        
        # Don't clear axes if a rectangle selector is active (prevents disconnecting it)
        if self.rectangle_selectors[plot_idx] is not None:
            if self.rectangle_selectors[plot_idx].active:
                return
        
        x_feat = self.feature_selections[plot_idx]['x'].get()
        y_feat = self.feature_selections[plot_idx]['y'].get()
        if not x_feat or not y_feat or x_feat not in self.prop_titles or y_feat not in self.prop_titles:
            return
        x_idx = self.prop_titles.index(x_feat)
        y_idx = self.prop_titles.index(y_feat)
        
        # Get dots within global display ranges
        in_range_indices = self.get_dots_in_display_range()
        if in_range_indices is None or len(in_range_indices) == 0:
            ax = self.axes[plot_idx]
            ax.clear()
            ax.set_xlabel(x_feat)
            ax.set_ylabel(y_feat)
            ax.grid(True, alpha=0.3)
            self.figures[plot_idx].tight_layout(pad=0.5)
            self.canvases[plot_idx].draw()
            return
        
        # Get data for dots in range
        x_data_full = self.all_properties[in_range_indices, x_idx]
        y_data_full = self.all_properties[in_range_indices, y_idx]
        
        # Get focus mask
        focus_mask_full = self.get_focus_mask()
        if focus_mask_full is not None:
            focus_mask_full = focus_mask_full[in_range_indices]
        
        # Apply dot density
        n_total = len(x_data_full)
        n_requested = max(1, int(n_total * self.dot_density.get()))
        max_dots_val = self.max_dots.get()
        n_plot = min(n_requested, max_dots_val)
        
        if n_plot < n_total:
            indices = np.random.choice(n_total, n_plot, replace=False)
            x_data = x_data_full[indices]
            y_data = y_data_full[indices]
            if focus_mask_full is not None:
                focus_mask = focus_mask_full[indices]
            else:
                focus_mask = None
        else:
            x_data = x_data_full
            y_data = y_data_full
            focus_mask = focus_mask_full
        
        # Clear and redraw
        ax = self.axes[plot_idx]
        ax.clear()
        
        # Get unit_id for displayed spikes
        if self.unit_id is not None:
            if n_plot < n_total:
                unit_ids_displayed = self.unit_id[in_range_indices[indices]]
            else:
                unit_ids_displayed = self.unit_id[in_range_indices]
        else:
            unit_ids_displayed = np.full(len(x_data), -1, dtype=int)
        
        # Get selected unit label
        selected_unit_label = None
        selected = self.selected_unit.get()
        if selected and selected != "None" and selected != "new_unit":
            try:
                label_str = selected.split()[1]
                selected_unit_label = int(label_str)
            except:
                pass
        
        # Unit colors by sorted position (mean_y_pos): [yellow, blue, green, purple, brown, cyan], then loop
        _SU_COLORS = ['yellow', 'blue', 'green', 'purple', 'brown', 'cyan']
        def color_for_unit(ul):
            if ul not in self.unit_labels:
                return 'gray'
            return _SU_COLORS[self.unit_labels.index(ul) % len(_SU_COLORS)]
        
        # Separate points by unit assignment and focus
        unassigned_mask = (unit_ids_displayed == -1)
        assigned_mask = ~unassigned_mask
        
        # Separate bounded (focused) from non-bounded
        bounded_mask = focus_mask if focus_mask is not None else np.zeros(len(x_data), dtype=bool)
        non_bounded_mask = ~bounded_mask
        
        try:
            s_dot = max(0.25, min(5.0, float(self.dot_size.get())))
        except (ValueError, tk.TclError, AttributeError):
            s_dot = 1.0
        
        # LAYER 1 (BACK): Plot unassigned non-bounded points
        if np.any(unassigned_mask & non_bounded_mask):
            unassigned_non_bounded = unassigned_mask & non_bounded_mask
            unassigned_x = x_data[unassigned_non_bounded]
            unassigned_y = y_data[unassigned_non_bounded]
            
            # Use selected hue source (cluster density / background density / cluster index) for coloring in Gaussian model mode
            hue_array = self._get_hue_array_for_display()
            if hue_array is not None:
                if n_plot < n_total:
                    hue_displayed = hue_array[in_range_indices[indices]]
                else:
                    hue_displayed = hue_array[in_range_indices]
                hue_unassigned = hue_displayed[unassigned_non_bounded]
                if len(hue_unassigned) > 0:
                    try:
                        vmin = float(self.density_hue_min.get())
                        vmax = float(self.density_hue_max.get())
                    except (ValueError, tk.TclError):
                        vmin, vmax = 1.0, 6.0
                    if vmax <= vmin:
                        vmax = vmin + 1.0
                    hue_norm = (hue_unassigned - vmin) / (vmax - vmin)
                    hue_norm = np.clip(hue_norm, 0.0, 1.0)
                    alpha_values = 0.1 + 0.9 * hue_norm
                    colors = np.zeros((len(hue_norm), 3))
                    colors[:, 0] = hue_norm
                    sort_ind = np.argsort(hue_unassigned)
                    ax.scatter(unassigned_x[sort_ind], unassigned_y[sort_ind], c=colors[sort_ind], alpha=alpha_values[sort_ind], s=s_dot, zorder=1, edgecolors='none')
                else:
                    ax.scatter(unassigned_x, unassigned_y, c='black', alpha=0.1, s=s_dot, zorder=1, edgecolors='none')
            else:
                ax.scatter(unassigned_x, unassigned_y, c='black', alpha=0.1, s=s_dot, zorder=1, edgecolors='none')
        
        # LAYER 2 (MIDDLE): Plot assigned non-bounded points
        if self.plot_sorted.get() and np.any(assigned_mask & non_bounded_mask):
            assigned_non_bounded = assigned_mask & non_bounded_mask
            assigned_x_nb = x_data[assigned_non_bounded]
            assigned_y_nb = y_data[assigned_non_bounded]
            assigned_unit_ids_nb = unit_ids_displayed[assigned_non_bounded]
            
            for unit_label in self.unit_labels:
                unit_mask_nb = (assigned_unit_ids_nb == unit_label)
                if np.any(unit_mask_nb):
                    color = color_for_unit(unit_label)
                    ax.scatter(assigned_x_nb[unit_mask_nb], assigned_y_nb[unit_mask_nb], 
                             c=color, alpha=0.8, s=s_dot, zorder=2)
        
        # LAYER 3 (FRONT): Plot bounded (focused) points
        if np.any(bounded_mask):
            bounded_x = x_data[bounded_mask]
            bounded_y = y_data[bounded_mask]
            bounded_unit_ids = unit_ids_displayed[bounded_mask]
            
            # Plot unassigned bounded points (red, alpha=0.5, no edges)
            unassigned_bounded = (bounded_unit_ids == -1)
            if np.any(unassigned_bounded):
                ax.scatter(bounded_x[unassigned_bounded], bounded_y[unassigned_bounded], 
                         c='red', alpha=0.5, s=s_dot * 3, zorder=3, edgecolors='none')
            
            # Plot assigned bounded points
            if self.plot_sorted.get():
                for unit_label in self.unit_labels:
                    unit_mask_b = (bounded_unit_ids == unit_label)
                    if np.any(unit_mask_b):
                        color = color_for_unit(unit_label)
                        ax.scatter(bounded_x[unit_mask_b], bounded_y[unit_mask_b], 
                                 c=color, alpha=0.5, s=s_dot * 5, zorder=4, edgecolors='none')
        
        # LAYER 4 (TOP): Plot selected unit's spikes in red
        if selected_unit_label is not None and selected_unit_label in self.unit_info:
            unit = self.unit_info[selected_unit_label]
            unit_spike_ids = self.get_spike_ids_for_unit(selected_unit_label)
            unit_spike_mask = np.isin(in_range_indices, unit_spike_ids)
            if np.any(unit_spike_mask):
                unit_spike_indices = np.where(unit_spike_mask)[0]
                if n_plot < n_total:
                    # Check which of the displayed spikes are from the unit
                    displayed_unit_mask = np.isin(indices, unit_spike_indices)
                    unit_x = x_data[displayed_unit_mask]
                    unit_y = y_data[displayed_unit_mask]
                else:
                    unit_x = x_data[unit_spike_indices]
                    unit_y = y_data[unit_spike_indices]
                
                if len(unit_x) > 0:
                    ax.scatter(unit_x, unit_y, c='red', alpha=0.8, s=s_dot * 5, zorder=5, edgecolors='none')
        
        # LAYER: Cluster seed points (orange)
        if getattr(self, 'cluster_seeds', None) and len(self.cluster_seeds) > 0:
            displayed_global_ids = in_range_indices[indices] if n_plot < n_total else in_range_indices
            all_cs_ids = np.concatenate([np.asarray(cs['spike_ids']).ravel() for cs in self.cluster_seeds])
            cs_mask = np.isin(displayed_global_ids, all_cs_ids)
            if np.any(cs_mask):
                ax.scatter(x_data[cs_mask], y_data[cs_mask], c='orange', alpha=0.7, s=s_dot, zorder=4, edgecolors='none')
        
        # Plot bounds polygons with click detection
        # Plot each bounded unit's bounds in the unit's color only if unit has dots in this plot's range
        for unit_label in self.unit_labels:
            if unit_label not in self.unit_info:
                continue
            unit = self.unit_info[unit_label]
            if not unit.is_bounded():
                continue
            if not np.any(np.isin(self.get_spike_ids_for_unit(unit_label), in_range_indices)):
                continue
            bnd_color = color_for_unit(unit_label)
            for bound in unit.unit_variables:
                if bound.property_indices == (x_idx, y_idx):
                    vertices = np.array(bound.vertices)
                    ax.plot(np.append(vertices[:, 0], vertices[0, 0]),
                            np.append(vertices[:, 1], vertices[0, 1]),
                            color=bnd_color, linewidth=2, zorder=6, linestyle='-')
        # Selected unit's boundaries (thicker/dashed for emphasis), only if selected unit has dots in range
        selected_unit_bounds = []
        if selected_unit_label is not None and selected_unit_label in self.unit_info:
            unit = self.unit_info[selected_unit_label]
            if unit.is_bounded() and np.any(np.isin(self.get_spike_ids_for_unit(selected_unit_label), in_range_indices)):
                selected_unit_bounds = unit.unit_variables
        for bound in selected_unit_bounds:
            if bound.property_indices == (x_idx, y_idx):
                vertices = np.array(bound.vertices)
                ax.plot(np.append(vertices[:, 0], vertices[0, 0]), 
                       np.append(vertices[:, 1], vertices[0, 1]), 
                       color='red', linewidth=2.5, zorder=7, linestyle='--')
        # Cluster seed boundaries (orange)
        for cluster_seed in getattr(self, 'cluster_seeds', []):
            for bound in cluster_seed.get('bounds', []):
                if bound.property_indices == (x_idx, y_idx):
                    vertices = np.array(bound.vertices)
                    ax.plot(np.append(vertices[:, 0], vertices[0, 0]),
                            np.append(vertices[:, 1], vertices[0, 1]),
                            color='orange', linewidth=2, zorder=6, linestyle='-')
        
        # Plot Gaussian units: dashed ellipse at mah_th boundary in unit color (only if unit has dots in range)
        for unit_label in self.unit_labels:
            if unit_label not in self.unit_info:
                continue
            unit = self.unit_info[unit_label]
            if not unit.is_gaussian() or not unit.unit_variables:
                continue
            if not np.any(np.isin(self.get_spike_ids_for_unit(unit_label), in_range_indices)):
                continue
            gv = unit.unit_variables[0]
            if not isinstance(gv, dict) or not gv.get('gaussian'):
                continue
            feat_idxs = gv.get('feature_indices', [])
            if x_idx not in feat_idxs or y_idx not in feat_idxs:
                continue
            center = np.asarray(gv.get('center'))
            cov = np.asarray(gv.get('covariance'))
            mah_th = gv.get('mah_th')
            if center is None or cov is None or mah_th is None:
                continue
            ix, iy = feat_idxs.index(x_idx), feat_idxs.index(y_idx)
            center_2d = np.array([center[ix], center[iy]], dtype=float)
            cov_2d = np.array([[cov[ix, ix], cov[ix, iy]], [cov[iy, ix], cov[iy, iy]]], dtype=float)
            self._plot_mahalanobis_ellipse(ax, center_2d, cov_2d, mah_th, linestyle='--',
                                          color=color_for_unit(unit_label), linewidth=1.5)
        
        bound_lines = []  # Store line objects for click detection
        
        # Plot current bounds being defined
        for bound_idx, bound in enumerate(self.bounds):
            if bound.property_indices == (x_idx, y_idx):
                vertices = np.array(bound.vertices)
                # Highlight selected bound
                if self.selected_bounds[plot_idx] == bound:
                    line, = ax.plot(np.append(vertices[:, 0], vertices[0, 0]), 
                                  np.append(vertices[:, 1], vertices[0, 1]), 
                                  'r-', linewidth=3, zorder=5, picker=5)
                else:
                    line, = ax.plot(np.append(vertices[:, 0], vertices[0, 0]), 
                                  np.append(vertices[:, 1], vertices[0, 1]), 
                                  'b-', linewidth=2, zorder=5, picker=5)
                line._bound_index = bound_idx
                line._bound_object = bound
                bound_lines.append(line)
        
        # Connect click event for bound selection (only connect once per figure)
        if not hasattr(self.figures[plot_idx], '_bound_pick_connected'):
            def on_pick(event):
                if hasattr(event.artist, '_bound_object'):
                    bound = event.artist._bound_object
                    self.select_bound(plot_idx, bound)
            
            self.figures[plot_idx].canvas.mpl_connect('pick_event', on_pick)
            self.figures[plot_idx]._bound_pick_connected = True
        
        # Plot focus polygon if selected
        if self.focus_polygons[plot_idx] is not None:
            focus_vertices = np.array(self.focus_polygons[plot_idx])
            ax.plot(np.append(focus_vertices[:, 0], focus_vertices[0, 0]), 
                   np.append(focus_vertices[:, 1], focus_vertices[0, 1]), 
                   'r--', linewidth=2, zorder=6)
        
        # Set axis limits from global display ranges
        if x_feat in self.global_display_ranges:
            x_center = self.global_display_ranges[x_feat]['center'].get()
            x_range = self.global_display_ranges[x_feat]['range'].get()
            ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
        
        if y_feat in self.global_display_ranges:
            y_center = self.global_display_ranges[y_feat]['center'].get()
            y_range = self.global_display_ranges[y_feat]['range'].get()
            ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
        
        # Draw grid visualization if grid window is open
        if self.grid_window is not None and self.grid_range_vars:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # Draw min/max lines for X feature
            if x_idx in self.grid_range_vars:
                x_min = self.grid_range_vars[x_idx]['min'].get()
                x_max = self.grid_range_vars[x_idx]['max'].get()
                if x_min is not None and x_max is not None:
                    # Min line
                    if xlim[0] <= x_min <= xlim[1]:
                        ax.axvline(x_min, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=10)
                    # Max line
                    if xlim[0] <= x_max <= xlim[1]:
                        ax.axvline(x_max, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=10)
                    
                    # Draw grid lines for X (only within ROI bounds, centered)
                    if x_idx in self.grid_step_vars:
                        step = self.grid_step_vars[x_idx].get()
                        if step > 0:
                            # Center the grid within the range
                            range_size = x_max - x_min
                            n_steps = int(np.floor(range_size / step))
                            grid_span = n_steps * step
                            offset = (range_size - grid_span) / 2.0
                            grid_start = x_min + offset
                            grid_x = np.arange(grid_start, grid_start + grid_span + step/2, step)
                            # Clip grid_x to ROI bounds
                            grid_x = grid_x[(grid_x >= x_min) & (grid_x <= x_max)]
                            for gx in grid_x:
                                if xlim[0] <= gx <= xlim[1]:
                                    # Get Y bounds for this plot to limit line extent
                                    y_roi_min = ylim[0]
                                    y_roi_max = ylim[1]
                                    if y_idx in self.grid_range_vars:
                                        y_roi_min = max(ylim[0], self.grid_range_vars[y_idx]['min'].get())
                                        y_roi_max = min(ylim[1], self.grid_range_vars[y_idx]['max'].get())
                                    ax.plot([gx, gx], [y_roi_min, y_roi_max], color='blue', linestyle=':', linewidth=1.2, alpha=0.6, zorder=9)
            
            # Draw min/max lines for Y feature
            if y_idx in self.grid_range_vars:
                y_min = self.grid_range_vars[y_idx]['min'].get()
                y_max = self.grid_range_vars[y_idx]['max'].get()
                if y_min is not None and y_max is not None:
                    # Min line
                    if ylim[0] <= y_min <= ylim[1]:
                        ax.axhline(y_min, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=10)
                    # Max line
                    if ylim[0] <= y_max <= ylim[1]:
                        ax.axhline(y_max, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=10)
                    
                    # Draw grid lines for Y (only within ROI bounds, centered)
                    if y_idx in self.grid_step_vars:
                        step = self.grid_step_vars[y_idx].get()
                        if step > 0:
                            # Center the grid within the range
                            range_size = y_max - y_min
                            n_steps = int(np.floor(range_size / step))
                            grid_span = n_steps * step
                            offset = (range_size - grid_span) / 2.0
                            grid_start = y_min + offset
                            grid_y = np.arange(grid_start, grid_start + grid_span + step/2, step)
                            # Clip grid_y to ROI bounds
                            grid_y = grid_y[(grid_y >= y_min) & (grid_y <= y_max)]
                            for gy in grid_y:
                                if ylim[0] <= gy <= ylim[1]:
                                    # Get X bounds for this plot to limit line extent
                                    x_roi_min = xlim[0]
                                    x_roi_max = xlim[1]
                                    if x_idx in self.grid_range_vars:
                                        x_roi_min = max(xlim[0], self.grid_range_vars[x_idx]['min'].get())
                                        x_roi_max = min(xlim[1], self.grid_range_vars[x_idx]['max'].get())
                                    ax.plot([x_roi_min, x_roi_max], [gy, gy], color='blue', linestyle=':', linewidth=1.2, alpha=0.6, zorder=9)
        
        # Plot seeds if they exist (only seeds in range of all 4 plots, same as data)
        if self.seeds is not None and len(self.seeds) > 0:
            seeds_in_display_range = np.ones(len(self.seeds), dtype=bool)
            for feat_name, range_dict in self.global_display_ranges.items():
                if feat_name not in self.prop_titles:
                    continue
                feat_idx = self.prop_titles.index(feat_name)
                center = range_dict['center'].get()
                range_val = range_dict['range'].get()
                feat_min = center - range_val / 2.0
                feat_max = center + range_val / 2.0
                seed_vals = self.all_properties[self.seeds, feat_idx]
                seeds_in_display_range &= (seed_vals >= feat_min) & (seed_vals <= feat_max)
            seed_x = self.all_properties[self.seeds, x_idx]
            seed_y = self.all_properties[self.seeds, y_idx]
            seed_x = seed_x[seeds_in_display_range]
            seed_y = seed_y[seeds_in_display_range]
            if len(seed_x) > 0:
                ax.scatter(seed_x, seed_y, s=100, c='red', edgecolors='black', 
                         linewidths=2, zorder=10, marker='o')
        
        ax.set_xlabel(x_feat)
        ax.set_ylabel(y_feat)
        ax.grid(True, alpha=0.3)
        self.figures[plot_idx].tight_layout(pad=0.5)
        self.canvases[plot_idx].draw()
    
    def update_all_plots(self):
        """Update all 4 plots"""
        # Check if we need to change label color for max dots (check once before updating plots)
        in_range_indices = self.get_dots_in_display_range()
        if in_range_indices is not None and len(in_range_indices) > 0:
            n_total = len(in_range_indices)
            n_requested = max(1, int(n_total * self.dot_density.get()))
            max_dots_val = self.max_dots.get()
            if n_requested > max_dots_val:
                # Change label color to red when max dots exceeded
                self.max_dots_label.config(foreground="red")
            else:
                # Change label color back to black when all points can be plotted
                self.max_dots_label.config(foreground="black")
        else:
            # No points in range, set to black
            self.max_dots_label.config(foreground="black")
        
        for plot_idx in range(4):
            self.update_plot(plot_idx)
        self.update_density_display()
        self.update_dot_size_display()
    
    def update_density_display(self):
        """Update density label"""
        density_val = self.dot_density.get()
        self.density_label.config(text=f"{density_val:.2f}")
    
    def update_dot_size_display(self):
        """Update dot size label"""
        if hasattr(self, 'dot_size_label'):
            try:
                self.dot_size_label.config(text=f"{self.dot_size.get():.2f}")
            except (ValueError, tk.TclError):
                pass
    
    def enable_zoom(self, plot_idx):
        """Enable box zoom mode for a plot"""
        if plot_idx < 0 or plot_idx >= len(self.axes):
            return
        
        # Disable any existing selectors
        if self.polygon_selectors[plot_idx] is not None:
            self.polygon_selectors[plot_idx].set_active(False)
            self.polygon_selectors[plot_idx] = None
        
        # Remove existing rectangle selector
        if self.rectangle_selectors[plot_idx] is not None:
            self.rectangle_selectors[plot_idx].set_active(False)
            self.rectangle_selectors[plot_idx] = None
        
        ax = self.axes[plot_idx]
        x_feat = self.feature_selections[plot_idx]['x'].get()
        y_feat = self.feature_selections[plot_idx]['y'].get()
        if not x_feat or not y_feat or x_feat not in self.prop_titles or y_feat not in self.prop_titles:
            return
        ax = self.axes[plot_idx]
        
        # Only update plot if it doesn't have data yet (to avoid unnecessary replotting)
        if len(ax.collections) == 0 and len(ax.lines) == 0:
            self.update_plot(plot_idx)
            ax = self.axes[plot_idx]
        
        # Verify axes have valid limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if xlim[0] == xlim[1] or ylim[0] == ylim[1]:
            return
        
        # Ensure canvas is drawn so axes transformation is ready
        self.canvases[plot_idx].draw()
        
        def onselect(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            center_x = (x_min + x_max) / 2.0
            range_x = x_max - x_min
            self._update_display_range(x_feat, center_x, range_x)
            
            center_y = (y_min + y_max) / 2.0
            range_y = y_max - y_min
            self._update_display_range(y_feat, center_y, range_y)
            
            # Disable and disconnect this selector
            sel = self.rectangle_selectors[plot_idx]
            if sel is not None:
                sel.set_active(False)
                sel.disconnect_events()
                self.rectangle_selectors[plot_idx] = None
            
            self.update_all_plots()
        
        # Calculate minimum spans in data coordinates (1% of range)
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        min_span_x_data = max(0.01 * x_range, 1e-10)  # At least 1% of range, but never zero
        min_span_y_data = max(0.01 * y_range, 1e-10)
        
        selector = RectangleSelector(ax, onselect, useblit=False, 
                                    button=[1], minspanx=min_span_x_data, minspany=min_span_y_data, 
                                    spancoords='data', interactive=False)
        self.rectangle_selectors[plot_idx] = selector
        selector.set_active(True)
    
    def reset_zoom(self, plot_idx):
        """Reset zoom to full range for a plot"""
        x_feat = self.feature_selections[plot_idx]['x'].get()
        y_feat = self.feature_selections[plot_idx]['y'].get()
        if x_feat and x_feat in self.prop_titles:
            x_min, x_max = self.get_feature_range(x_feat)
            if x_min is not None:
                center = (x_min + x_max) / 2.0
                range_val = (x_max - x_min) * 1.1
                self._update_display_range(x_feat, center, range_val)
        if y_feat and y_feat in self.prop_titles:
            y_min, y_max = self.get_feature_range(y_feat)
            if y_min is not None:
                center = (y_min + y_max) / 2.0
                range_val = (y_max - y_min) * 1.1
                self._update_display_range(y_feat, center, range_val)
        
        self.update_all_plots()
    
    def reset_all_zoom(self):
        """Reset all graphs to full data range."""
        if not self.global_display_ranges or self.prop_titles is None:
            return
        self.updating_programmatically = True
        try:
            for feat_name in list(self.global_display_ranges.keys()):
                if feat_name not in self.prop_titles:
                    continue
                feat_min, feat_max = self.get_feature_range(feat_name)
                if feat_min is not None and feat_max is not None:
                    center = (feat_min + feat_max) / 2.0
                    range_val = (feat_max - feat_min) * 1.1
                    self._update_display_range(feat_name, center, range_val)
        finally:
            self.updating_programmatically = False
        self.update_all_plots()
    
    def zoom_out(self, plot_idx):
        """Increase range for both axes of this plot by 50%."""
        x_feat = self.feature_selections[plot_idx]['x'].get()
        y_feat = self.feature_selections[plot_idx]['y'].get()
        if not x_feat or not y_feat or x_feat not in self.global_display_ranges or y_feat not in self.global_display_ranges:
            return
        for feat_name in (x_feat, y_feat):
            center = self.global_display_ranges[feat_name]['center'].get()
            range_val = self.global_display_ranges[feat_name]['range'].get()
            new_range = range_val * 1.5
            self._update_display_range(feat_name, center, new_range)
        self.update_all_plots()
    
    def enable_add_seed_mode(self, plot_idx):
        """Enable add/remove seed mode: single-click adds closest unassigned high-density dot; double-click removes closest seed."""
        if self.clust_ind is None or self.all_properties is None:
            messagebox.showwarning("Add/remove seed", "Calculate densities first.")
            return
        if self.unit_id is None:
            self.unit_id = np.full(self.all_properties.shape[0], -1, dtype=int)
        n_spikes = self.all_properties.shape[0]
        self._require_curr_max_prob(n_spikes)
        # Disconnect any previous add-seed handler
        if self._add_seed_cid is not None:
            try:
                self.figures[self.add_seed_plot_idx].canvas.mpl_disconnect(self._add_seed_cid)
            except Exception:
                pass
            self._add_seed_cid = None
        self.add_seed_plot_idx = plot_idx
        self._add_seed_last_click_time = 0
        self._add_seed_last_click_xy = None
        self._add_seed_cid = self.figures[plot_idx].canvas.mpl_connect(
            'button_press_event', self._on_add_seed_click)
    
    def _on_add_seed_click(self, event):
        """On click: single-click adds closest unassigned dot with clust_ind > vmax as seed;
        double-click removes the closest seed to the click. Uses timer-based double-click detection."""
        if self.add_seed_plot_idx is None or event.inaxes is not self.axes[self.add_seed_plot_idx]:
            return
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return
        plot_idx = self.add_seed_plot_idx
        x_feat = self.feature_selections[plot_idx]['x'].get()
        y_feat = self.feature_selections[plot_idx]['y'].get()
        if not x_feat or not y_feat or x_feat not in self.prop_titles or y_feat not in self.prop_titles:
            return
        x_idx = self.prop_titles.index(x_feat)
        y_idx = self.prop_titles.index(y_feat)
        now = time.time()
        xy = (event.xdata, event.ydata)

        # Cancel any pending single-click
        if self._add_seed_pending_after_id is not None:
            try:
                self.root.after_cancel(self._add_seed_pending_after_id)
            except Exception:
                pass
            self._add_seed_pending_after_id = None

        # Double-click: second click within 350ms and within 5% of axis range of first
        is_double = False
        if self._add_seed_last_click_xy is not None and (now - self._add_seed_last_click_time) < 0.35:
            dx = abs(xy[0] - self._add_seed_last_click_xy[0])
            dy = abs(xy[1] - self._add_seed_last_click_xy[1])
            x_lo, x_hi = self.get_feature_range(x_feat) or (1.0, 1.0)
            y_lo, y_hi = self.get_feature_range(y_feat) or (1.0, 1.0)
            x_range = max(abs(x_hi - x_lo), 1e-9)
            y_range = max(abs(y_hi - y_lo), 1e-9)
            if dx < 0.05 * x_range and dy < 0.05 * y_range:
                is_double = True

        self._add_seed_last_click_time = now
        self._add_seed_last_click_xy = xy

        if is_double:
            # Double-click: remove closest seed (from self.seeds, not datapoints)
            seeds_list = list(self.seeds) if self.seeds else []
            if len(seeds_list) == 0:
                messagebox.showwarning("Add/remove seed", "No seeds to remove.")
                self._add_seed_last_click_xy = None
                return
            x_seeds = self.all_properties[seeds_list, x_idx]
            y_seeds = self.all_properties[seeds_list, y_idx]
            dist_sq = (x_seeds - event.xdata) ** 2 + (y_seeds - event.ydata) ** 2
            best_idx = int(np.argmin(dist_sq))
            seed_to_remove = seeds_list[best_idx]
            self.seeds.remove(seed_to_remove)
            self._exit_add_seed_mode()
            self.update_all_plots()
            return

        def _do_single_click():
            self._add_seed_pending_after_id = None
            if self.add_seed_plot_idx is None:
                return
            try:
                vmax = float(self.density_hue_max.get())
            except (ValueError, tk.TclError):
                vmax = 6.0
            in_range_indices = self.get_dots_in_display_range()
            if in_range_indices is None or len(in_range_indices) == 0:
                messagebox.showwarning("Add/remove seed", "No points in display range.")
                self._exit_add_seed_mode()
                return
            unassigned = (self.unit_id[in_range_indices] == -1)
            high_den = (self.clust_ind[in_range_indices] > vmax)
            candidate_mask = unassigned & high_den
            if not np.any(candidate_mask):
                messagebox.showwarning("Add/remove seed", f"No unassigned point with density > {vmax}. Try lowering the density hue max or click near a brighter dot.")
                self._exit_add_seed_mode()
                return
            candidate_flat = np.where(candidate_mask)[0]
            global_indices = in_range_indices[candidate_flat]
            x_cand = self.all_properties[global_indices, x_idx]
            y_cand = self.all_properties[global_indices, y_idx]
            dist_sq = (x_cand - xy[0]) ** 2 + (y_cand - xy[1]) ** 2
            best = np.argmin(dist_sq)
            spike_id = int(global_indices[best])
            if self.seeds is None:
                self.seeds = []
            if spike_id not in self.seeds:
                self.seeds.append(spike_id)
            self._exit_add_seed_mode()
            if self.compute_gaussian_models_button is not None:
                self.compute_gaussian_models_button.config(state=tk.NORMAL)
            self.update_all_plots()

        self._add_seed_pending_after_id = self.root.after(250, _do_single_click)
    
    def _exit_add_seed_mode(self):
        """Disconnect add-seed click handler, cancel pending single-click, and clear mode."""
        if self._add_seed_pending_after_id is not None:
            try:
                self.root.after_cancel(self._add_seed_pending_after_id)
            except Exception:
                pass
            self._add_seed_pending_after_id = None
        if self._add_seed_cid is not None and self.add_seed_plot_idx is not None:
            try:
                self.figures[self.add_seed_plot_idx].canvas.mpl_disconnect(self._add_seed_cid)
            except Exception:
                pass
            self._add_seed_cid = None
        self.add_seed_plot_idx = None
    
    def move_x_range(self, plot_idx, fraction):
        """Move X axis display range center by fraction of current range"""
        x_feat = self.feature_selections[plot_idx]['x'].get()
        if not x_feat or x_feat not in self.prop_titles:
            return
        # Get current display range
        if x_feat not in self.global_display_ranges:
            return
        
        current_center = self.global_display_ranges[x_feat]['center'].get()
        current_range = self.global_display_ranges[x_feat]['range'].get()
        
        # Calculate move distance
        move_distance = fraction * current_range
        
        # Get data range limits
        x_min, x_max = self.get_feature_range(x_feat)
        if x_min is None:
            return
        
        # Calculate new center
        new_center = current_center + move_distance
        
        # Calculate display range limits
        display_min = new_center - current_range / 2.0
        display_max = new_center + current_range / 2.0
        
        # Cap at data range
        if display_min < x_min:
            new_center = x_min + current_range / 2.0
        elif display_max > x_max:
            new_center = x_max - current_range / 2.0
        
        # Update display range
        self._update_display_range(x_feat, new_center, current_range)
    
    def move_y_range(self, plot_idx, fraction):
        """Move Y axis display range center by fraction of current range"""
        y_feat = self.feature_selections[plot_idx]['y'].get()
        if not y_feat or y_feat not in self.prop_titles:
            return
        # Get current display range
        if y_feat not in self.global_display_ranges:
            return
        
        current_center = self.global_display_ranges[y_feat]['center'].get()
        current_range = self.global_display_ranges[y_feat]['range'].get()
        
        # Calculate move distance
        move_distance = fraction * current_range
        
        # Get data range limits
        y_min, y_max = self.get_feature_range(y_feat)
        if y_min is None:
            return
        
        # Calculate new center
        new_center = current_center + move_distance
        
        # Calculate display range limits
        display_min = new_center - current_range / 2.0
        display_max = new_center + current_range / 2.0
        
        # Cap at data range
        if display_min < y_min:
            new_center = y_min + current_range / 2.0
        elif display_max > y_max:
            new_center = y_max - current_range / 2.0
        
        # Update display range
        self._update_display_range(y_feat, new_center, current_range)
    
    def on_unit_selection_changed(self):
        """Handle unit selection change - update button states, center graphs on unit data, and update plots"""
        selected = self.selected_unit.get()
        
        # Enable/disable make bound buttons based on selection
        is_new_unit = (selected == "new_unit")
        for btn in self.make_bound_buttons:
            if btn is not None:
                btn.config(state=tk.NORMAL if is_new_unit else tk.DISABLED)
        
        # Enable/disable delete unit, edit unit, and ISI analysis buttons
        if self.delete_unit_btn is not None:
            self.delete_unit_btn.config(state=tk.NORMAL if not is_new_unit else tk.DISABLED)
        if getattr(self, 'edit_unit_btn', None) is not None:
            self.edit_unit_btn.config(state=tk.NORMAL if not is_new_unit else tk.DISABLED)
        if getattr(self, 'isi_analysis_btn', None) is not None:
            self.isi_analysis_btn.config(state=tk.NORMAL if not is_new_unit else tk.DISABLED)
        self._update_combined_isi_button_visibility()
        
        # When a unit is selected: center all graphs on unit data with range = 2 * unit data range
        if not is_new_unit and self.all_properties is not None and self.prop_titles is not None:
            try:
                label_str = selected.split()[1]
                unit_label = int(label_str)
            except Exception:
                unit_label = None
            if unit_label is not None and unit_label in self.unit_info:
                unit = self.unit_info[unit_label]
                spike_ids = self.get_spike_ids_for_unit(unit_label)
                if len(spike_ids) > 0:
                    self.updating_programmatically = True
                    try:
                        for feat_name in list(self.global_display_ranges.keys()):
                            if feat_name not in self.prop_titles:
                                continue
                            feat_idx = self.prop_titles.index(feat_name)
                            unit_vals = self.all_properties[spike_ids, feat_idx].astype(float)
                            data_min = np.min(unit_vals)
                            data_max = np.max(unit_vals)
                            data_range = data_max - data_min
                            if data_range <= 0:
                                data_range = 1e-9
                            center = (data_min + data_max) / 2.0
                            range_val = 2.0 * data_range
                            # Clamp display range to full data range (do not exceed global min/max)
                            global_min, global_max = self.get_feature_range(feat_name)
                            if global_min is not None and global_max is not None:
                                display_min = center - range_val / 2.0
                                display_max = center + range_val / 2.0
                                display_min = max(display_min, global_min)
                                display_max = min(display_max, global_max)
                                center = (display_min + display_max) / 2.0
                                range_val = display_max - display_min
                            self.global_display_ranges[feat_name]['center'].set(center)
                            self.global_display_ranges[feat_name]['range'].set(range_val)
                    finally:
                        self.updating_programmatically = False
        
        # Update plots to show selected unit
        self.update_all_plots()
    
    def go_prev_unit(self):
        """Select the previous unit in the list (wrap to last if on new_unit)."""
        values = list(self.unit_combo['values'])
        if len(values) <= 1:
            return
        current = self.selected_unit.get()
        try:
            idx = values.index(current)
        except ValueError:
            idx = 0
        new_idx = (idx - 1) % len(values)
        self.selected_unit.set(values[new_idx])
        self.on_unit_selection_changed()
    
    def go_next_unit(self):
        """Select the next unit in the list (wrap to new_unit if on last)."""
        values = list(self.unit_combo['values'])
        if len(values) <= 1:
            return
        current = self.selected_unit.get()
        try:
            idx = values.index(current)
        except ValueError:
            idx = 0
        new_idx = (idx + 1) % len(values)
        self.selected_unit.set(values[new_idx])
        self.on_unit_selection_changed()
    
    def _sync_compare_combo_values(self):
        """Keep compare unit menu in sync with selected unit menu (same unit list)."""
        if getattr(self, 'compare_combo', None) is None or self.unit_combo is None:
            return
        vals = self.unit_combo['values']
        self.compare_combo['values'] = list(vals) if vals else ["new_unit"]
    
    def _update_combined_isi_button_visibility(self):
        """Show Cross ISI button only when both Selected Unit and Compare Unit have a unit selected."""
        if getattr(self, 'combined_isi_btn', None) is None:
            return
        sel = (self.selected_unit.get() or "").strip()
        cmp_val = (self.compare_unit.get() or "").strip()
        both_units = (sel != "new_unit" and sel != "" and cmp_val != "new_unit" and cmp_val != "")
        if both_units:
            self.combined_isi_btn.pack(pady=2, fill=tk.X)
        else:
            self.combined_isi_btn.pack_forget()
    
    def go_prev_compare_unit(self):
        """Select the previous unit in the compare list."""
        if getattr(self, 'compare_combo', None) is None:
            return
        values = list(self.compare_combo['values'])
        if len(values) <= 1:
            return
        current = self.compare_unit.get()
        try:
            idx = values.index(current)
        except ValueError:
            idx = 0
        new_idx = (idx - 1) % len(values)
        self.compare_unit.set(values[new_idx])
        self._update_combined_isi_button_visibility()
    
    def go_next_compare_unit(self):
        """Select the next unit in the compare list."""
        if getattr(self, 'compare_combo', None) is None:
            return
        values = list(self.compare_combo['values'])
        if len(values) <= 1:
            return
        current = self.compare_unit.get()
        try:
            idx = values.index(current)
        except ValueError:
            idx = 0
        new_idx = (idx + 1) % len(values)
        self.compare_unit.set(values[new_idx])
        self._update_combined_isi_button_visibility()
    
    def delete_selected_unit(self):
        """Delete the currently selected unit"""
        selected = self.selected_unit.get()
        if selected == "new_unit" or not selected:
            return
        
        try:
            label_str = selected.split()[1]
            unit_label = int(label_str)
        except:
            return
        
        if unit_label not in self.unit_info:
            return
        
        orphan_spike_ids = self.get_spike_ids_for_unit(unit_label)
        su = np.asarray(orphan_spike_ids)
        self.unit_id[su] = -1
        n_spikes = self.all_properties.shape[0]
        self._require_curr_max_prob(n_spikes)
        self._require_background_den(n_spikes)
        self.curr_max_prob[su] = np.where(self.background_den[su] > 0, self.background_den[su].astype(float), np.inf)
        
        del self.unit_info[unit_label]
        self.unit_labels.remove(unit_label)
        
        # Update dropdown and resort labels by position
        if len(self.unit_labels) > 0:
            self.resort_unit_labels_by_position()
        else:
            self.unit_combo['values'] = ["new_unit"]
            self._sync_compare_combo_values()
            self.selected_unit.set("new_unit")
            self.compare_unit.set("new_unit")
            self.on_unit_selection_changed()
        
        # Update plot sorted checkbox state
        if len(self.unit_labels) == 0:
            self.plot_sorted_check.config(state=tk.DISABLED)
        elif self.sorting_type.get() == "Cluster":
            self.plot_sorted_check.config(state=tk.NORMAL)
    
    def refresh_gm_unit_assignment(self):
        """Reset all GM-assigned spikes, then re-assign using sorted-dim range, mah distance, and BIC refinement (data whitened by model cov for BIC)."""
        if self.all_properties is None or self.unit_id is None:
            messagebox.showwarning("Refresh", "No data or unit assignments loaded.")
            return
        n_spikes = self.all_properties.shape[0]
        self._require_curr_max_prob(n_spikes)
        self._require_background_den(n_spikes)
        gm_labels = [ul for ul in self.unit_labels if self.unit_info[ul].is_gaussian() and self.unit_info[ul].unit_variables]
        if not gm_labels:
            messagebox.showinfo("Refresh", "No Gaussian units to refresh.")
            return
        reset_mask = np.isin(self.unit_id, gm_labels)
        self.unit_id[reset_mask] = -1
        self.curr_max_prob[reset_mask] = np.where(self.background_den[reset_mask] > 0, self.background_den[reset_mask].astype(float), np.inf)
        settings = self._get_gaussian_model_settings()
        min_points_for_cluster = int(settings.get('min_points_for_cluster', 100)) if settings else 100
        sorted_global = None
        if self.included_feature_indexes is not None and len(self.included_feature_indexes) > 0:
            incl = list(self.included_feature_indexes)
            if settings is not None and 0 <= settings.get('sorted_feature_idx', -1) < len(incl):
                sorted_global = int(incl[int(settings['sorted_feature_idx'])])
            elif self.prop_titles and 'y_pos' in self.prop_titles:
                sorted_global = self.prop_titles.index('y_pos')
            else:
                sorted_global = int(incl[0])
        if sorted_global is None and self.prop_titles and 'y_pos' in self.prop_titles:
            sorted_global = self.prop_titles.index('y_pos')
        if sorted_global is None:
            messagebox.showwarning("Refresh", "Could not determine sorted dimension.")
            return
        sorted_dim_col = self.all_properties[:, sorted_global].astype(float)
        order = np.argsort(sorted_dim_col)
        sorted_vals = sorted_dim_col[order]
        n_features = self.all_properties.shape[1]
        for ul in sorted(gm_labels, key=lambda l: self.unit_info[l].mean_y_pos):
            unit = self.unit_info[ul]
            gv = unit.unit_variables[0]
            center = np.asarray(gv.get('center'), dtype=float)
            cov = np.asarray(gv.get('covariance'), dtype=float)
            mah_th = float(gv.get('mah_th', 2.0))
            bic_th = float(gv.get('bic_th', 0.2))
            feat_idx = list(gv.get('feature_indices', []))
            if not feat_idx or center is None or cov is None or sorted_global not in feat_idx:
                continue
            i_sorted = feat_idx.index(sorted_global)
            center_sorted = float(center[i_sorted])
            std_sorted = np.sqrt(max(0, float(cov[i_sorted, i_sorted])))
            if std_sorted <= 0:
                std_sorted = 1.0
            range_lo = center_sorted - mah_th * std_sorted
            range_hi = center_sorted + mah_th * std_sorted
            left = np.searchsorted(sorted_vals, range_lo, side='left')
            right = np.searchsorted(sorted_vals, range_hi, side='right')
            candidate_indices = order[left:right]
            if len(candidate_indices) == 0:
                continue
            points_cand = self.all_properties[candidate_indices][:, np.asarray(feat_idx)].astype(float)
            diff = points_cand - center
            inv_cov = np.linalg.pinv(cov)
            mah_sq = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
            mah_sq = np.maximum(mah_sq, 0.0)
            mah_dist = np.sqrt(mah_sq)
            density_curve = gv.get('density_curve')
            prob_density = prob_density_from_curve_or_formula(mah_dist, mah_th, density_curve=density_curve)
            in_bounds = mah_dist <= mah_th
            improves = in_bounds & (prob_density > self.curr_max_prob[candidate_indices])
            in_bounds_inds = np.where(improves)[0]
            if len(in_bounds_inds) < min_points_for_cluster:
                refined = in_bounds_inds
            else:
                refined = apply_bic_refinement(
                    points_cand, center, cov, in_bounds_inds, bic_th, len(feat_idx),
                    min_points_for_cluster=min_points_for_cluster, seed_row=None
                )
            if len(refined) == 0:
                continue
            final_spike_ids = candidate_indices[refined]
            self.unit_id[final_spike_ids] = ul
            self.curr_max_prob[final_spike_ids] = prob_density[refined]
        self.update_all_plots()
    
    def edit_selected_unit(self):
        """Edit the selected unit: GMM -> accept/reject window; Bounds -> restore bounds for re-edit."""
        selected = self.selected_unit.get()
        if selected == "new_unit" or not selected:
            messagebox.showwarning("Edit Unit", "Please select a unit.")
            return
        try:
            label_str = selected.split()[1]
            unit_label = int(label_str)
        except Exception:
            messagebox.showwarning("Edit Unit", "Invalid unit selection.")
            return
        if unit_label not in self.unit_info:
            messagebox.showwarning("Edit Unit", "Unit not found.")
            return
        unit = self.unit_info[unit_label]
        if unit.is_gaussian():
            self._edit_gaussian_unit(unit_label, unit)
        elif unit.is_bounded():
            self._edit_bounded_unit(unit_label, unit)
        else:
            messagebox.showwarning("Edit Unit", "Edit is only supported for GMM and Bounds units.")
    
    def _edit_gaussian_unit(self, unit_label, unit):
        """Save GM params to temp vars, delete unit, open accept/reject window from spatial window (same as defining GM from seeds)."""
        if not unit.unit_variables or not isinstance(unit.unit_variables[0], dict):
            messagebox.showwarning("Edit Unit", "Unit has no stored Gaussian model parameters.")
            return
        gv = unit.unit_variables[0]
        center = np.asarray(gv.get('center'), dtype=float).copy()
        covariance = np.asarray(gv.get('covariance'), dtype=float).copy()
        mah_th = float(gv.get('mah_th', 2.0))
        bic_th = float(gv.get('bic_th', 0.2))
        feature_indices = list(gv.get('feature_indices', self.included_feature_indexes or []))
        if center is None or covariance is None or self.all_properties is None or not feature_indices:
            messagebox.showwarning("Edit Unit", "Missing data or parameters for this unit.")
            return
        # Get assigned spikes before we unassign (for seed choice)
        spike_ids = self.get_spike_ids_for_unit(unit_label)
        # Switch to Gaussian model and selected to new_unit
        self.sorting_type.set("Single seed")
        self.selected_unit.set("new_unit")
        self.compare_unit.set("new_unit")
        self.on_sorting_type_changed()
        # Load sorted-feature and GM settings from self.settings['gm'] (same as Compute / Parameters window)
        incl = list(self.included_feature_indexes) if self.included_feature_indexes else list(feature_indices)
        included_feat_idxs = list(self.included_feature_indexes)
        gm = self.settings.get('gm') or {}
        if 0 <= gm.get('sorted_feature_idx', -1) < len(incl):
            sorted_feature_idx = int(incl[gm['sorted_feature_idx']])
        else:
            sorted_feature_idx = int(self.prop_titles.index('y_pos') if self.prop_titles and 'y_pos' in self.prop_titles else feature_indices[0])
        sorted_feature_title = self.prop_titles[self.sort_feature_idx] if (getattr(self, 'sort_feature_idx', None) is not None and self.sort_feature_idx < len(self.prop_titles)) else "sorted feature"
        max_range_sorted = float(gm.get('max_range_sorted', gm.get(f'max_range_{sorted_feature_title}', 4.0)))
        half_window = max_range_sorted / 2.0
        min_points_for_cluster = int(gm.get('min_points_for_cluster', 100))
        # Center in sorted dimension (saved center is in feature_indices space)
        if sorted_feature_idx in feature_indices:
            center_sorted = float(center[feature_indices.index(sorted_feature_idx)])
        else:
            center_sorted = float(np.median(self.all_properties[:, sorted_feature_idx]))
        # Spatial window: all spikes in sorted-feature range (half-window = max_range_sorted/2)
        sorted_feat_col = self.all_properties[:, sorted_feature_idx]
        in_range = (sorted_feat_col >= center_sorted - half_window) & (sorted_feat_col <= center_sorted + half_window)
        valid_indices_all = np.where(in_range)[0]
        if len(valid_indices_all) < 2:
            messagebox.showwarning("Edit Unit", "No spikes in sorted-feature window around unit center.")
            return
        # Build points and map unit center/covariance to included-feature space
        properties_filtered = self.all_properties[:, included_feat_idxs]
        points = properties_filtered[valid_indices_all].astype(np.float32)
        n_features = points.shape[1]
        if included_feat_idxs == feature_indices:
            center_use = center
            cov_use = covariance
        else:
            if not all(j in feature_indices for j in included_feat_idxs):
                messagebox.showwarning("Edit Unit", "Unit features differ from current included features; ellipse may not match.")
            center_use = np.array([center[feature_indices.index(j)] if j in feature_indices else np.nanmean(self.all_properties[:, j]) for j in included_feat_idxs], dtype=float)
            cov_use = np.zeros((n_features, n_features))
            for i, ii in enumerate(included_feat_idxs):
                for j, jj in enumerate(included_feat_idxs):
                    if ii in feature_indices and jj in feature_indices:
                        cov_use[i, j] = covariance[feature_indices.index(ii), feature_indices.index(jj)]
                    elif i == j:
                        cov_use[i, j] = np.nanvar(self.all_properties[:, ii]) or 1.0
        # Seed for Re-stabilize/Re-fit: assigned spike in window closest to unit center (Mahalanobis)
        in_window = np.intersect1d(spike_ids, valid_indices_all)
        seed_idx_global = None
        initial_model = None
        if len(in_window) > 0:
            positions = self.all_properties[in_window][:, included_feat_idxs].astype(float)
            inv_cov = np.linalg.pinv(cov_use)
            diff = positions - center_use
            mah_sq = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
            seed_idx_global = int(in_window[np.argmin(mah_sq)])
            # Build full GM settings for _create_init_cluster_from_seed (sorted_feature_idx = column index in properties_filtered)
            sorted_feature_idx_local = included_feat_idxs.index(sorted_feature_idx) if sorted_feature_idx in included_feat_idxs else 0
            edit_gm = dict(self.settings.get('gm') or {})
            init_stds_edit = np.asarray(edit_gm.get('initial_stds', np.sqrt(np.maximum(np.diag(cov_use), 1e-12))))
            edit_settings = {
                'sorted_feature_idx': sorted_feature_idx_local,
                'min_points_for_cluster': min_points_for_cluster,
                'initial_stds': init_stds_edit,
                'max_range_sorted': float(edit_gm.get('max_range_sorted', edit_gm.get(f'max_range_{sorted_feature_title}', float(init_stds_edit[sorted_feature_idx_local] * 4.0)))),
                'n_samples_density_curve': int(edit_gm.get('n_samples_density_curve', 101)),
            }
            for k in ['init_mah_th', 'com_iteration_threshold', 'com_iteration_max_iterations', 'density_threshold_for_init_distance',
                      'gaussian_filter_sigma', 'dist_step', 'min_change', 'multi_cluster_threshold']:
                if k in edit_gm:
                    edit_settings[k] = edit_gm[k]
            n_spikes = self.all_properties.shape[0]
            self._require_curr_max_prob(n_spikes)
            cluster_dens = self.cluster_den if self.cluster_den is not None else np.ones(properties_filtered.shape[0], dtype=float)
            init_result = _create_init_cluster_from_seed(
                properties_filtered, seed_idx_global, cluster_dens, self.curr_max_prob, edit_settings
            )
            if init_result.get('success'):
                initial_model = init_result['model'].copy()
        # Unassign spikes and remove unit (no longer know which data belonged to this unit)
        self.unit_id[spike_ids] = -1
        n_spikes = self.all_properties.shape[0]
        self._require_curr_max_prob(n_spikes)
        self._require_background_den(n_spikes)
        self.curr_max_prob[spike_ids] = np.where(self.background_den[spike_ids] > 0, self.background_den[spike_ids].astype(float), np.inf)
        del self.unit_info[unit_label]
        self.unit_labels.remove(unit_label)
        if len(self.unit_labels) > 0:
            self.resort_unit_labels_by_position()
        else:
            self.unit_combo['values'] = ["new_unit"]
            self._sync_compare_combo_values()
            self.selected_unit.set("new_unit")
            self.compare_unit.set("new_unit")
            self.on_unit_selection_changed()
        inv_cov = np.linalg.pinv(cov_use)
        diff = points - center_use
        mahal_d = np.sqrt(np.einsum('ij,jk,ik->i', diff, inv_cov, diff))
        density_curve = gv.get('density_curve')
        prob_density = prob_density_from_curve_or_formula(mahal_d, mah_th, density_curve=density_curve)
        cmp_slice = self.curr_max_prob[valid_indices_all]
        in_bounds = np.where(prob_density > cmp_slice)[0]
        in_bounds = apply_bic_refinement(points, center_use, cov_use, in_bounds, bic_th, n_features, min_points_for_cluster=min_points_for_cluster)
        HC = self.clust_ind[valid_indices_all].astype(np.float64) if self.clust_ind is not None and len(self.clust_ind) > np.max(valid_indices_all) else np.ones(len(valid_indices_all), dtype=float)
        visualization_data = {
            'points': points,
            'valid_indices': valid_indices_all.copy(),
            'all_points': points.copy(),
            'all_valid_indices': valid_indices_all.copy(),
            'all_points_full': self.all_properties[valid_indices_all].copy(),
            'center': center_use.copy(),
            'covariance': cov_use.copy(),
            'mahal_d': mahal_d,
            'mah_th': mah_th,
            'in_bounds': in_bounds,
            'HC': HC,
            'seed_point': center_use.copy(),
            'multi_cluster_threshold': bic_th,
        }
        gaussian_model = GaussianModel(mean=center_use, covariance=cov_use, bic_threshold=bic_th, mah_threshold=mah_th)
        if gm and 'viz_feature_pairs' in gm and hasattr(gm['viz_feature_pairs'], '__len__') and len(gm['viz_feature_pairs']) >= 1:
            viz_pairs = [(int(p[0]), int(p[1])) for p in gm['viz_feature_pairs']]
        else:
            viz_pairs = [(included_feat_idxs[0], included_feat_idxs[min(1, len(included_feat_idxs)-1)]),
                         (included_feat_idxs[0], included_feat_idxs[min(2, len(included_feat_idxs)-1)]),
                         (included_feat_idxs[min(3, len(included_feat_idxs)-1)], included_feat_idxs[min(4, len(included_feat_idxs)-1)]),
                         (included_feat_idxs[0], included_feat_idxs[min(5, len(included_feat_idxs)-1)])]
        cluster_dens_for_refit = self.cluster_den if self.cluster_den is not None else np.ones(properties_filtered.shape[0], dtype=float)
        settings = {
            'unit_id': self.unit_id,
            'curr_max_prob': self.curr_max_prob,
            'viz_feature_pairs': viz_pairs,
            'min_points_for_cluster': min_points_for_cluster,
        }
        accepted, accept_data = self._show_accept_reject_mahal_window(
            visualization_data, gaussian_model, seed_idx=0, n_seeds=1, feature_pairs=settings['viz_feature_pairs'], settings=settings,
            initial_model=initial_model, properties_for_refit=properties_filtered, cluster_densities_for_refit=cluster_dens_for_refit,
            seed_idx_global=seed_idx_global)
        if accepted and accept_data is not None:
            new_spike_ids = np.asarray(accept_data['spike_ids'])
            if len(new_spike_ids) > 0:
                self.gaussian_models.append(accept_data['model'])
                new_unit_label = (max(self.unit_labels) + 1) if self.unit_labels else 0
                if 'y_pos' in self.prop_titles:
                    y_pos_idx = self.prop_titles.index('y_pos')
                    mean_y_pos = float(np.mean(self.all_properties[new_spike_ids, y_pos_idx]))
                else:
                    mean_y_pos = 0.0
                m = accept_data['model']
                unit_vars = [{
                    'gaussian': True,
                    'feature_indices': list(accept_data['feature_indices']),
                    'center': np.array(m.mean, dtype=float).copy(),
                    'covariance': np.array(m.covariance, dtype=float).copy(),
                    'mah_th': float(m.mah_threshold),
                    'bic_th': float(m.bic_threshold),
                }]
                dc = getattr(m, 'density_curve', None)
                if dc is not None:
                    unit_vars[0]['density_curve'] = np.asarray(dc, dtype=float)
                if m.data_range is not None:
                    unit_vars[0]['data_range'] = m.data_range
                if m.sort_range is not None:
                    unit_vars[0]['sort_range'] = m.sort_range
                if accept_data.get('sort_feature_idx_global') is not None:
                    unit_vars[0]['sort_feature_idx'] = accept_data['sort_feature_idx_global']
                new_unit = Unit(unit_variables=unit_vars, mean_y_pos=mean_y_pos)
                self.unit_info[new_unit_label] = new_unit
                self.unit_labels.append(new_unit_label)
                self.unit_labels = sorted(self.unit_labels, key=lambda l: self.unit_info[l].mean_y_pos)
                self.unit_id[new_spike_ids] = new_unit_label
                if self.curr_max_prob is not None and 'curr_max_prob_update' in accept_data:
                    self.curr_max_prob[new_spike_ids] = np.minimum(self.curr_max_prob[new_spike_ids], accept_data['curr_max_prob_update'])
                self._remove_empty_units()
                self.resort_unit_labels_by_position()
                self.unit_combo['values'] = ["new_unit"] + [f"Unit {l}" for l in self.unit_labels]
                self._sync_compare_combo_values()
                if len(self.unit_labels) > 0:
                    self.plot_sorted_check.config(state=tk.NORMAL)
                self.update_all_plots()
    
    def _get_time_column_index(self):
        """Return the column index that corresponds to time (for ISI), or None if none found.
        Checks for common time-dimension names: 't', 'time', 'timestamp', 'spike_time' (case-sensitive)."""
        if self.prop_titles is None:
            return None
        for name in ('t', 'time', 'timestamp', 'spike_time'):
            if name in self.prop_titles:
                return self.prop_titles.index(name)
        return None
    
    def show_isi_analysis(self):
        """Show ISI histogram (0–10 ms, 0.1 ms bins) for the selected unit in a popup."""
        selected = self.selected_unit.get()
        if selected == "new_unit" or not selected:
            messagebox.showwarning("ISI analysis", "Please select a unit.")
            return
        try:
            label_str = selected.split()[1]
            unit_label = int(label_str)
        except Exception:
            messagebox.showwarning("ISI analysis", "Invalid unit selection.")
            return
        if unit_label not in self.unit_info:
            messagebox.showwarning("ISI analysis", "Unit not found.")
            return
        if self.all_properties is None or self.prop_titles is None:
            messagebox.showerror("ISI analysis", "No data loaded.")
            return
        t_idx = self._get_time_column_index()
        if t_idx is None:
            messagebox.showerror("ISI analysis", "No time column found in properties (look for 't', 'time', 'timestamp', or 'spike_time').")
            return
        unit = self.unit_info[unit_label]
        spike_ids = self.get_spike_ids_for_unit(unit_label)
        if len(spike_ids) < 2:
            messagebox.showwarning("ISI analysis", "Unit has fewer than 2 spikes; cannot compute ISIs.")
            return
        timestamps = self.all_properties[spike_ids, t_idx].astype(float)
        timestamps = np.sort(timestamps)
        isi_sec = np.diff(timestamps)
        isi_ms = isi_sec * 1000.0
        bin_edges = np.arange(0, 10.0 + 0.1, 0.1)
        counts, _ = np.histogram(isi_ms, bins=bin_edges, range=(0, 10))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        popup = tk.Toplevel(self.root)
        popup.title(f"ISI histogram - Unit {unit_label}")
        popup.transient(self.root)
        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.bar(bin_centers, counts, width=0.1, align='center', edgecolor='none', color='steelblue')
        ax.set_xlabel("ISI (ms)")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 10)
        ax.grid(True, alpha=0.3)
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()
    
    def show_cross_isi_analysis(self):
        """Cross ISI: for each spike in selected unit, time to the next spike in compare unit. Histogram 0–10 ms, 0.1 ms bins."""
        sel = (self.selected_unit.get() or "").strip()
        cmp_val = (self.compare_unit.get() or "").strip()
        if sel == "new_unit" or not sel or cmp_val == "new_unit" or not cmp_val:
            messagebox.showwarning("Cross ISI", "Please select a unit in both Selected Unit and Compare Unit.")
            return
        try:
            label_a = int(sel.split()[1])
            label_b = int(cmp_val.split()[1])
        except Exception:
            messagebox.showwarning("Cross ISI", "Invalid unit selection.")
            return
        if label_a not in self.unit_info or label_b not in self.unit_info:
            messagebox.showwarning("Cross ISI", "Unit not found.")
            return
        if self.all_properties is None or self.prop_titles is None:
            messagebox.showerror("Cross ISI", "No data loaded.")
            return
        t_idx = self._get_time_column_index()
        if t_idx is None:
            messagebox.showerror("Cross ISI", "No time column found in properties (look for 't', 'time', 'timestamp', or 'spike_time').")
            return
        times_a = np.sort(self.all_properties[self.get_spike_ids_for_unit(label_a), t_idx].astype(float))
        times_b = np.sort(self.all_properties[self.get_spike_ids_for_unit(label_b), t_idx].astype(float))
        if len(times_a) == 0 or len(times_b) == 0:
            messagebox.showwarning("Cross ISI", "One or both units have no spikes.")
            return
        # For each spike in selected unit (A), time to next spike in compare unit (B)
        cross_isi_ms = []
        for t_a in times_a:
            # first t_b >= t_a in times_b
            idx = np.searchsorted(times_b, t_a, side='left')
            if idx < len(times_b):
                cross_isi_ms.append((times_b[idx] - t_a) * 1000.0)
        cross_isi_ms = np.array(cross_isi_ms) if cross_isi_ms else np.array([])
        if len(cross_isi_ms) == 0:
            messagebox.showwarning("Cross ISI", "No cross ISIs (no compare-unit spike after any selected-unit spike).")
            return
        bin_edges = np.arange(0, 10.0 + 0.1, 0.1)
        counts, _ = np.histogram(cross_isi_ms, bins=bin_edges, range=(0, 10))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        popup = tk.Toplevel(self.root)
        popup.title(f"Cross ISI - Unit {label_a} → Unit {label_b}")
        popup.transient(self.root)
        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.bar(bin_centers, counts, width=0.1, align='center', edgecolor='none', color='steelblue')
        ax.set_xlabel("Cross ISI (ms)")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 10)
        ax.grid(True, alpha=0.3)
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()
    
    def _update_display_range(self, feat_name, center, range_val):
        """Update display range for a feature. Caller should call update_all_plots() once after batch updates."""
        if feat_name not in self.global_display_ranges:
            self.global_display_ranges[feat_name] = {
                'center': tk.DoubleVar(),
                'range': tk.DoubleVar()
            }
        
        self.global_display_ranges[feat_name]['center'].set(center)
        self.global_display_ranges[feat_name]['range'].set(range_val)
        
        # Refresh all plots once (skip if caller is batching range changes, e.g. reset_all_zoom)
        if not getattr(self, 'updating_programmatically', False):
            self.updating_programmatically = True
            try:
                self.update_all_plots()
            finally:
                self.updating_programmatically = False
    
    def define_bounds(self):
        """Enable polygon selection for defining bounds"""
        # Find first plot with valid features
        active_plot = None
        for plot_idx in range(4):
            x_feat = self.feature_selections[plot_idx]['x'].get()
            y_feat = self.feature_selections[plot_idx]['y'].get()
            if x_feat and y_feat and x_feat in self.prop_titles and y_feat in self.prop_titles:
                active_plot = plot_idx
                break
        
        if active_plot is None:
            messagebox.showerror("Error", "Please select features for at least one plot")
            return
        
        # Disable other polygon selectors
        for idx in range(4):
            if self.polygon_selectors[idx] is not None:
                self.polygon_selectors[idx].set_active(False)
                self.polygon_selectors[idx] = None
        
        # Create polygon selector for active plot
        ax = self.axes[active_plot]
        x_feat = self.feature_selections[active_plot]['x'].get()
        y_feat = self.feature_selections[active_plot]['y'].get()
        x_idx = self.prop_titles.index(x_feat)
        y_idx = self.prop_titles.index(y_feat)
        
        def onselect(vertices):
            if len(vertices) >= 3:
                bound = Bound((x_idx, y_idx), vertices, self.next_unit_label)
                self.bounds.append(bound)
                self.next_unit_label += 1
                self.update_all_plots()
                self.clear_bounds_btn.config(state=tk.NORMAL)
                self.assign_units_btn.config(state=tk.NORMAL)
        
        selector = PolygonSelector(ax, onselect, useblit=True)
        # Set line properties after creation
        if hasattr(selector, 'line'):
            selector.line.set_color('red')
            selector.line.set_linewidth(2)
        self.polygon_selectors[active_plot] = selector
    
    def select_bound(self, plot_idx, bound):
        """Select a bound for a specific plot"""
        self.selected_bounds[plot_idx] = bound
        # Show buttons above ALL graphs when any bound is selected
        # Bound buttons visibility now controlled by sorting type in on_sorting_type_changed
        self.update_plot(plot_idx)
    
    def _update_bound_buttons_visibility(self):
        """Bound buttons visibility is now controlled by sorting type, not bound selection"""
        # Buttons are always in feature frame, visibility controlled by on_sorting_type_changed
        pass
    
    def make_bound_for_plot(self, plot_idx):
        """Create a bound for a specific plot"""
        x_feat = self.feature_selections[plot_idx]['x'].get()
        y_feat = self.feature_selections[plot_idx]['y'].get()
        if not x_feat or not y_feat or x_feat not in self.prop_titles or y_feat not in self.prop_titles:
            messagebox.showerror("Error", "Please select valid features for this plot")
            return
        
        # Disable other polygon selectors
        for idx in range(4):
            if self.polygon_selectors[idx] is not None:
                self.polygon_selectors[idx].set_active(False)
                self.polygon_selectors[idx] = None
        
        # Create polygon selector for this plot
        ax = self.axes[plot_idx]
        x_idx = self.prop_titles.index(x_feat)
        y_idx = self.prop_titles.index(y_feat)
        
        def onselect(vertices):
            if len(vertices) >= 3:
                bound = Bound((x_idx, y_idx), vertices, self.next_unit_label)
                self.bounds.append(bound)
                self.next_unit_label += 1
                # Select the newly created bound
                self.select_bound(plot_idx, bound)
                self.update_all_plots()
                self.clear_bounds_btn.config(state=tk.NORMAL)
                self.assign_units_btn.config(state=tk.NORMAL)
        
        selector = PolygonSelector(ax, onselect, useblit=True)
        # Set line properties after creation
        if hasattr(selector, 'line'):
            selector.line.set_color('red')
            selector.line.set_linewidth(2)
        self.polygon_selectors[plot_idx] = selector
    
    def delete_bound_for_plot(self, plot_idx):
        """Delete bound(s) for a specific plot"""
        x_feat = self.feature_selections[plot_idx]['x'].get()
        y_feat = self.feature_selections[plot_idx]['y'].get()
        if not x_feat or not y_feat or x_feat not in self.prop_titles or y_feat not in self.prop_titles:
            messagebox.showwarning("Warning", "Please select valid features for this plot")
            return
        x_idx = self.prop_titles.index(x_feat)
        y_idx = self.prop_titles.index(y_feat)
        
        # Find bounds for this plot's feature pair
        bounds_for_plot = [b for b in self.bounds if b.property_indices == (x_idx, y_idx)]
        
        if len(bounds_for_plot) == 0:
            # No bounds found for this plot's feature pair
            return
        
        if len(bounds_for_plot) == 1:
            # Single bound - delete it
            if messagebox.askyesno("Confirm", f"Delete bound with unit label {bounds_for_plot[0].unit_label}?"):
                self.bounds.remove(bounds_for_plot[0])
                self.selected_bounds[plot_idx] = None
        else:
            # Multiple bounds - ask which one or delete all
            if messagebox.askyesno("Confirm", f"Delete all {len(bounds_for_plot)} bounds for this plot?"):
                for bound in bounds_for_plot:
                    if bound in self.bounds:
                        self.bounds.remove(bound)
                self.selected_bounds[plot_idx] = None
        
        # Update plots
        self.update_all_plots()
        
        # Update button states
        if len(self.bounds) == 0:
            if self.clear_bounds_btn is not None:
                self.clear_bounds_btn.config(state=tk.DISABLED)
            if self.assign_units_btn is not None:
                self.assign_units_btn.config(state=tk.DISABLED)
    
    def clear_bounds(self):
        """Clear all bounds"""
        self.bounds = []
        for idx in range(4):
            if self.polygon_selectors[idx] is not None:
                self.polygon_selectors[idx].set_active(False)
                self.polygon_selectors[idx] = None
            self.focus_polygons[idx] = None
            self.selected_bounds[idx] = None
            # Hide buttons
            # Bound buttons are now in feature frame, no need to hide separately
        self.update_all_plots()
        self.clear_bounds_btn.config(state=tk.DISABLED)
        self.assign_units_btn.config(state=tk.DISABLED)
    
    def define_seed_cluster(self):
        """Store current bounds and spike IDs inside them as a cluster_seed (init_cluster); do not create a unit."""
        if len(self.bounds) == 0:
            messagebox.showwarning("Warning", "No bounds defined. Please create bounds first.")
            return
        
        focus_mask = self.get_focus_mask()
        if focus_mask is None or not np.any(focus_mask):
            messagebox.showwarning("Warning", "No spikes found within all bounds.")
            return
        
        spike_ids = np.where(focus_mask)[0]
        init_cluster = {
            'bounds': list(self.bounds),
            'spike_ids': spike_ids.copy(),
        }
        self.cluster_seeds.append(init_cluster)
        
        # Clear bounds after storing (same as before)
        self.bounds = []
        for idx in range(4):
            if self.polygon_selectors[idx] is not None:
                self.polygon_selectors[idx].set_active(False)
                self.polygon_selectors[idx] = None
            self.selected_bounds[idx] = None
        self.clear_bounds_btn.config(state=tk.DISABLED)
        self.assign_units_btn.config(state=tk.DISABLED)
        if self.compute_gaussian_models_cluster_btn is not None:
            self.compute_gaussian_models_cluster_btn.config(state=tk.NORMAL)
        self._refresh_cluster_seeds_ui()
        self.update_all_plots()
    
    def _refresh_cluster_seeds_ui(self):
        """Update cluster seeds combo values and related button states (safe to call anytime)."""
        if getattr(self, 'cluster_seeds_combo', None) is not None and self.cluster_seeds_combo.winfo_exists():
            vals = ["none"] + [f"Cluster seed {i+1}" for i in range(len(getattr(self, 'cluster_seeds', [])))]
            self.cluster_seeds_combo['values'] = vals
            cur = self.selected_cluster_seed.get()
            if cur not in vals:
                self.selected_cluster_seed.set("none")
            if self.compute_gaussian_models_cluster_btn is not None and self.compute_gaussian_models_cluster_btn.winfo_exists():
                self.compute_gaussian_models_cluster_btn.config(
                    state=tk.NORMAL if getattr(self, 'cluster_seeds', None) and len(self.cluster_seeds) > 0 else tk.DISABLED)
            if getattr(self, 'delete_cluster_seed_btn', None) is not None and self.delete_cluster_seed_btn.winfo_exists():
                self.delete_cluster_seed_btn.config(
                    state=tk.NORMAL if self.selected_cluster_seed.get() and self.selected_cluster_seed.get() != "none" else tk.DISABLED)
    
    def _get_selected_cluster_seed_index(self):
        """Return 0-based index of selected cluster seed, or None if none selected."""
        sel = self.selected_cluster_seed.get()
        if not sel or sel == "none":
            return None
        try:
            return int(sel.split()[-1]) - 1
        except (ValueError, IndexError):
            return None
    
    def on_cluster_seed_selection_changed(self):
        """Focus graphs on the selected cluster seed (same idea as unit selection)."""
        idx = self._get_selected_cluster_seed_index()
        if getattr(self, 'delete_cluster_seed_btn', None) is not None and self.delete_cluster_seed_btn.winfo_exists():
            self.delete_cluster_seed_btn.config(state=tk.NORMAL if idx is not None else tk.DISABLED)
        if idx is None or not getattr(self, 'cluster_seeds', None) or idx >= len(self.cluster_seeds) or self.all_properties is None or self.prop_titles is None:
            self.update_all_plots()
            return
        cluster_seed = self.cluster_seeds[idx]
        spike_ids = np.asarray(cluster_seed['spike_ids'], dtype=int)
        if len(spike_ids) == 0:
            self.update_all_plots()
            return
        self.updating_programmatically = True
        try:
            for feat_name in list(self.global_display_ranges.keys()):
                if feat_name not in self.prop_titles:
                    continue
                feat_idx = self.prop_titles.index(feat_name)
                unit_vals = self.all_properties[spike_ids, feat_idx].astype(float)
                data_min = np.min(unit_vals)
                data_max = np.max(unit_vals)
                data_range = data_max - data_min
                if data_range <= 0:
                    data_range = 1e-9
                center = (data_min + data_max) / 2.0
                range_val = 2.0 * data_range
                global_min, global_max = self.get_feature_range(feat_name)
                if global_min is not None and global_max is not None:
                    display_min = center - range_val / 2.0
                    display_max = center + range_val / 2.0
                    display_min = max(display_min, global_min)
                    display_max = min(display_max, global_max)
                    center = (display_min + display_max) / 2.0
                    range_val = display_max - display_min
                self.global_display_ranges[feat_name]['center'].set(center)
                self.global_display_ranges[feat_name]['range'].set(range_val)
        finally:
            self.updating_programmatically = False
        self.update_all_plots()
    
    def delete_selected_cluster_seed(self):
        """Remove the cluster seed selected in the dropdown."""
        idx = self._get_selected_cluster_seed_index()
        if idx is None or not getattr(self, 'cluster_seeds', None) or idx >= len(self.cluster_seeds):
            return
        self.cluster_seeds.pop(idx)
        self.selected_cluster_seed.set("none")
        self._refresh_cluster_seeds_ui()
        self.update_all_plots()
    
    def compute_gaussian_models_from_cluster_seeds(self):
        """For each cluster_seed, build GM from cluster data, then show accept/reject window; on accept create unit."""
        if not getattr(self, 'cluster_seeds', None) or len(self.cluster_seeds) == 0:
            messagebox.showwarning("Compute Gaussian Models", "No cluster seeds defined. Use Define seed cluster first.")
            return
        if self.all_properties is None or self.included_feature_indexes is None:
            messagebox.showwarning("Compute Gaussian Models", "Load data and set included features first.")
            return
        n_spikes = self.all_properties.shape[0]
        self._require_curr_max_prob(n_spikes)
        self._require_background_den(n_spikes)
        included_feat_idxs = list(self.included_feature_indexes)
        properties_filtered = self.all_properties[:, included_feat_idxs]
        cluster_dens = self.cluster_den if self.cluster_den is not None else np.ones(properties_filtered.shape[0], dtype=float)
        settings = self._get_gaussian_model_settings()
        if settings is None:
            messagebox.showwarning("Compute Gaussian Models", "Could not get Gaussian model settings.")
            return
        gm = self.settings.get('gm') or {}
        if 'viz_feature_pairs' in gm and hasattr(gm['viz_feature_pairs'], '__len__') and len(gm['viz_feature_pairs']) >= 1:
            viz_pairs = [(int(p[0]), int(p[1])) for p in gm['viz_feature_pairs']]
        else:
            n_included = len(included_feat_idxs)
            viz_pairs = [(included_feat_idxs[0], included_feat_idxs[min(1, n_included - 1)]),
                         (included_feat_idxs[0], included_feat_idxs[min(2, n_included - 1)]),
                         (included_feat_idxs[min(3, n_included - 1)], included_feat_idxs[min(4, n_included - 1)]),
                         (included_feat_idxs[0], included_feat_idxs[min(5, n_included - 1)])]
        settings_for_dialog = {
            'unit_id': self.unit_id,
            'curr_max_prob': self.curr_max_prob,
            'viz_feature_pairs': viz_pairs,
            'min_points_for_cluster': settings.get('min_points_for_cluster', 100),
        }
        # Order seeds by cluster density (highest first): mean density at seed spike_ids
        def _seed_mean_density(cs):
            sid = np.asarray(cs['spike_ids'], dtype=int).ravel()
            return float(np.mean(cluster_dens[sid])) if len(sid) > 0 else 0.0
        seeds_by_density = sorted(list(self.cluster_seeds), key=_seed_mean_density, reverse=True)
        for cs_idx, cluster_seed in enumerate(seeds_by_density):
            spike_ids = np.asarray(cluster_seed['spike_ids'], dtype=int)
            result = make_gaussian_model_from_cluster(
                properties_filtered, cluster_dens, self.curr_max_prob, spike_ids, settings
            )
            if not result.get('success'):
                messagebox.showwarning("Cluster seed {}".format(cs_idx + 1), result.get('message', 'Fit failed'))
                continue
            viz = result['visualization_data']
            ci = result['cluster_indices']
            gaussian_model = result['model']
            # Model and viz are now window-based on full properties: use window_indices when present
            if 'window_indices' in result:
                window_indices = result['window_indices']
                valid_indices_global = np.asarray(window_indices, dtype=int)
                gaussian_model.data_range = (int(window_indices[0]), int(window_indices[-1]))
                seed_idx_global = int(result.get('seed_idx_global', window_indices[result['seed_idx_local']]))
            else:
                valid_indices_global = ci[viz['valid_indices']]
                gaussian_model.data_range = (int(valid_indices_global[0]), int(valid_indices_global[-1]))
                seed_idx_global = int(ci[result['seed_idx_local']])
            # In-bounds: same convention as Re-stabilize/Re-fit — valid_indices = global data-window indices, in_bounds = 0-based into that window
            first_global = int(valid_indices_global[0])
            last_global = int(valid_indices_global[-1])
            in_bounds_global = gaussian_model.in_bounds_indices(properties_filtered, self.curr_max_prob)
            in_bounds = (in_bounds_global[(in_bounds_global >= first_global) & (in_bounds_global <= last_global)] - first_global).astype(int)
            # Build viz window in full properties_filtered space (sort_range + margin) so we show all dots in range
            sort_lo, sort_hi = gaussian_model.sort_range
            width = sort_hi - sort_lo
            half_extra = 0.25 * width
            sorted_col = np.asarray(properties_filtered[:, gaussian_model.sort_feature_idx], dtype=float)
            viz_first = np.searchsorted(sorted_col, sort_lo - half_extra, side='left')
            viz_last = np.searchsorted(sorted_col, sort_hi + half_extra, side='right') - 1
            viz_last = min(max(viz_last, viz_first), len(properties_filtered) - 1)
            all_valid_indices_global = np.arange(viz_first, viz_last + 1)
            seed_row_in_points_arr = np.where(all_valid_indices_global == seed_idx_global)[0]
            seed_row_in_points = int(seed_row_in_points_arr[0]) if len(seed_row_in_points_arr) > 0 else None
            points_data = properties_filtered[valid_indices_global]
            inv_cov = np.linalg.pinv(gaussian_model.covariance)
            mahal_d_data = np.einsum('ij,jk,ik->i', points_data - gaussian_model.mean, inv_cov, points_data - gaussian_model.mean) ** 0.5
            visualization_data = {
                'points': points_data,
                'valid_indices': valid_indices_global,
                'mahal_d': mahal_d_data,
                'in_bounds': in_bounds,
                'all_points': properties_filtered[all_valid_indices_global],
                'all_valid_indices': all_valid_indices_global,
                'all_points_full': self.all_properties[all_valid_indices_global].copy(),
                'center': gaussian_model.mean.copy(),
                'covariance': gaussian_model.covariance.copy(),
                'HC': cluster_dens[all_valid_indices_global] if cluster_dens is not None else np.ones(len(all_valid_indices_global), dtype=float),
                'seed_point': properties_filtered[seed_idx_global].copy() if seed_idx_global < len(properties_filtered) else viz['seed_point'].copy(),
                'seed_row_in_points': seed_row_in_points,
                'multi_cluster_threshold': viz['multi_cluster_threshold'],
            }
            accepted, accept_data = self._show_accept_reject_mahal_window(
                visualization_data, gaussian_model, seed_idx=0, n_seeds=1,
                feature_pairs=viz_pairs, settings=settings_for_dialog,
                initial_model=None, properties_for_refit=properties_filtered,
                cluster_densities_for_refit=cluster_dens, seed_idx_global=seed_idx_global
            )
            try:
                self._disable_user_buttons()
                # Remove the cluster seed that was just used (whether accepted or rejected)
                if cluster_seed in self.cluster_seeds:
                    self.cluster_seeds.remove(cluster_seed)
                self._refresh_cluster_seeds_ui()
                if accepted and accept_data is not None:
                    new_spike_ids = np.asarray(accept_data['spike_ids'])
                    if len(new_spike_ids) > 0:
                        self.gaussian_models.append(accept_data['model'])
                        new_unit_label = (max(self.unit_labels) + 1) if self.unit_labels else 0
                        if 'y_pos' in self.prop_titles:
                            y_pos_idx = self.prop_titles.index('y_pos')
                            mean_y_pos = float(np.mean(self.all_properties[new_spike_ids, y_pos_idx]))
                        else:
                            mean_y_pos = 0.0
                        m = accept_data['model']
                        unit_vars = [{
                            'gaussian': True,
                            'feature_indices': list(included_feat_idxs),
                            'center': np.array(m.mean, dtype=float).copy(),
                            'covariance': np.array(m.covariance, dtype=float).copy(),
                            'mah_th': float(m.mah_threshold),
                            'bic_th': float(m.bic_threshold),
                        }]
                        if m.data_range is not None:
                            unit_vars[0]['data_range'] = m.data_range
                        if m.sort_range is not None:
                            unit_vars[0]['sort_range'] = m.sort_range
                        if accept_data.get('sort_feature_idx_global') is not None:
                            unit_vars[0]['sort_feature_idx'] = accept_data['sort_feature_idx_global']
                        new_unit = Unit(unit_variables=unit_vars, mean_y_pos=mean_y_pos)
                        self.unit_info[new_unit_label] = new_unit
                        self.unit_labels.append(new_unit_label)
                        self.unit_labels = sorted(self.unit_labels, key=lambda l: self.unit_info[l].mean_y_pos)
                        self.unit_id[new_spike_ids] = new_unit_label
                        if self.curr_max_prob is not None and 'curr_max_prob_update' in accept_data:
                            self.curr_max_prob[new_spike_ids] = np.minimum(self.curr_max_prob[new_spike_ids], accept_data['curr_max_prob_update'])
                        self._remove_empty_units()
                        self.resort_unit_labels_by_position()
                        self.unit_combo['values'] = ["new_unit"] + [f"Unit {l}" for l in self.unit_labels]
                        self._sync_compare_combo_values()
                        if len(self.unit_labels) > 0:
                            self.plot_sorted_check.config(state=tk.NORMAL)
                self.update_all_plots()
            finally:
                self._enable_user_buttons()
            # Let the main window refresh before the next cluster seed's window (if any)
            self.root.update_idletasks()
            self.root.update()
            time.sleep(0.5)
    
    def assign_units(self):
        """Create Unit objects from currently defined bounds and assign to unassigned spikes (legacy / unused in Cluster mode)."""
        if len(self.bounds) == 0:
            messagebox.showwarning("Warning", "No bounds defined. Please create bounds first.")
            return
        
        if self.unit_id is None:
            self.unit_id = np.full(self.all_properties.shape[0], -1, dtype=int)
        n_spikes = self.all_properties.shape[0]
        self._require_curr_max_prob(n_spikes)
        
        focus_mask = self.get_focus_mask()
        if focus_mask is None or not np.any(focus_mask):
            messagebox.showwarning("Warning", "No spikes found within all bounds.")
            return
        
        unassigned_in_bounds = focus_mask & (self.unit_id == -1)
        if not np.any(unassigned_in_bounds):
            messagebox.showwarning("Warning", "No unassigned spikes found within all bounds.")
            return
        
        spike_ids = np.where(unassigned_in_bounds)[0]
        if 'y_pos' in self.prop_titles:
            y_pos_idx = self.prop_titles.index('y_pos')
            mean_y_pos = np.mean(self.all_properties[spike_ids, y_pos_idx])
        else:
            mean_y_pos = 0.0
        
        if len(self.unit_labels) == 0:
            new_unit_label = 0
        else:
            new_unit_label = max(self.unit_labels) + 1
        
        unit = Unit(
            unit_variables=list(self.bounds),
            mean_y_pos=mean_y_pos
        )
        self.unit_info[new_unit_label] = unit
        self.unit_id[spike_ids] = new_unit_label
        self.curr_max_prob[spike_ids] = 0.0
        self.unit_labels.append(new_unit_label)
        self.unit_labels = sorted(self.unit_labels, key=lambda l: self.unit_info[l].mean_y_pos)
        
        self.bounds = []
        for idx in range(4):
            if self.polygon_selectors[idx] is not None:
                self.polygon_selectors[idx].set_active(False)
                self.polygon_selectors[idx] = None
            self.selected_bounds[idx] = None
        self.unit_combo['values'] = ["new_unit"] + [f"Unit {l}" for l in self.unit_labels]
        self._sync_compare_combo_values()
        self.selected_unit.set("new_unit")
        self.on_unit_selection_changed()
        if len(self.unit_labels) > 0:
            self.plot_sorted_check.config(state=tk.NORMAL)
        self.clear_bounds_btn.config(state=tk.DISABLED)
        self.assign_units_btn.config(state=tk.DISABLED)
        self.update_all_plots()
    
    def calculate_densities(self):
        """Open density calculation window"""
        if self.all_properties is None:
            messagebox.showerror("Error", "Please load data first")
            return
        
        self._open_select_features_window()
    
    def _open_select_features_window(self):
        """Open window for selecting features (included/excluded)"""
        if self.all_properties is None:
            messagebox.showerror("Error", "Please load data first")
            return
        
        feat_window = tk.Toplevel(self.root)
        feat_window.title("Select Features")
        feat_window.geometry("500x400")
        feat_window.transient(self.root)  # Keep on top of main window
        feat_window.grab_set()  # Make modal
        
        # Default: exclude 't' and 'ch', include all others
        excluded_default = ['t', 'ch']
        included_features = []
        excluded_features = []
        
        for feat_name in self.prop_titles:
            if feat_name in excluded_default:
                excluded_features.append(feat_name)
            else:
                included_features.append(feat_name)
        
        # Main frame
        main_frame = ttk.Frame(feat_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Lists frame
        lists_frame = ttk.Frame(main_frame)
        lists_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Excluded features frame
        excluded_frame = ttk.LabelFrame(lists_frame, text="Excluded Features", padding="5")
        excluded_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        excluded_listbox = tk.Listbox(excluded_frame, selectmode=tk.EXTENDED)
        excluded_listbox.pack(fill=tk.BOTH, expand=True)
        for feat in excluded_features:
            excluded_listbox.insert(tk.END, feat)
        
        # Included features frame
        included_frame = ttk.LabelFrame(lists_frame, text="Included Features", padding="5")
        included_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        included_listbox = tk.Listbox(included_frame, selectmode=tk.EXTENDED)
        included_listbox.pack(fill=tk.BOTH, expand=True)
        for feat in included_features:
            included_listbox.insert(tk.END, feat)
        
        # Buttons frame (between lists)
        buttons_frame = ttk.Frame(lists_frame)
        buttons_frame.pack(side=tk.LEFT, padx=5)
        
        def move_to_included():
            """Move selected features from excluded to included"""
            selected_indices = excluded_listbox.curselection()
            if not selected_indices:
                return
            # Get items in reverse order to maintain indices
            items_to_move = [excluded_listbox.get(i) for i in reversed(selected_indices)]
            for item in items_to_move:
                excluded_listbox.delete(excluded_listbox.get(0, tk.END).index(item))
                included_listbox.insert(tk.END, item)
        
        def move_to_excluded():
            """Move selected features from included to excluded"""
            selected_indices = included_listbox.curselection()
            if not selected_indices:
                return
            # Get items in reverse order to maintain indices
            items_to_move = [included_listbox.get(i) for i in reversed(selected_indices)]
            for item in items_to_move:
                included_listbox.delete(included_listbox.get(0, tk.END).index(item))
                excluded_listbox.insert(tk.END, item)
        
        ttk.Button(buttons_frame, text=">", command=move_to_included, width=5).pack(pady=5)
        ttk.Button(buttons_frame, text="<", command=move_to_excluded, width=5).pack(pady=5)
        
        # Done button
        def done():
            """Close feature selection and open grid window"""
            # Get included feature indexes
            included_feat_names = list(included_listbox.get(0, tk.END))
            included_indexes = [self.prop_titles.index(name) for name in included_feat_names]
            
            feat_window.destroy()
            self._open_define_grid_window(included_indexes)
        
        done_btn = ttk.Button(main_frame, text="Done", command=done)
        done_btn.pack(pady=10)
    
    def _open_define_grid_window(self, included_feature_indexes):
        """Open window for defining density calculation grid"""
        if self.all_properties is None:
            messagebox.showerror("Error", "Please load data first")
            return
        
        grid_window = tk.Toplevel(self.root)
        grid_window.title("Define Grid")
        grid_window.geometry("600x600")
        grid_window.transient(self.root)  # Keep on top of main window
        # Don't use grab_set() - allow interaction with plots
        
        # Store reference to grid window for plot updates
        self.grid_window = grid_window
        self.grid_range_vars = {}
        self.grid_step_vars = {}
        
        # Disable UI controls while grid window is open
        if self.sorting_combo is not None:
            self.sorting_combo.config(state=tk.DISABLED)
        if self.unit_combo is not None:
            self.unit_combo.config(state=tk.DISABLED)
        if self.calc_densities_btn is not None:
            self.calc_densities_btn.config(state=tk.DISABLED)
        if self.find_seeds_btn is not None:
            self.find_seeds_btn.config(state=tk.DISABLED)
        
        def on_window_close():
            """Handle window close - re-enable UI and clear visualization"""
            self.grid_window = None
            self.grid_range_vars = {}
            self.grid_step_vars = {}
            # Re-enable UI controls
            if self.sorting_combo is not None:
                self.sorting_combo.config(state="readonly")
            if self.unit_combo is not None:
                self.unit_combo.config(state="readonly")
            if self.calc_densities_btn is not None and self.all_properties is not None:
                self.calc_densities_btn.config(state=tk.NORMAL)
            if self.find_seeds_btn is not None and self.cluster_den is not None:
                self.find_seeds_btn.config(state=tk.NORMAL)
            self.update_all_plots()
            grid_window.destroy()
        
        grid_window.protocol("WM_DELETE_WINDOW", on_window_close)
        
        # Create scrollable frame
        canvas = tk.Canvas(grid_window)
        scrollbar = ttk.Scrollbar(grid_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Defaults from self.settings (populated by load data or Load settings)
        defaults_dict = dict(self.settings.get('grid', {}))
        
        # Grid ranges and steps for included features only
        ranges_frame = ttk.LabelFrame(scrollable_frame, text="Grid Ranges and Steps", padding="3")
        ranges_frame.pack(fill=tk.X, padx=3, pady=3)
        
        range_vars = {}
        step_vars = {}
        
        def update_plot_visualization():
            """Update plots to show min/max lines and grid"""
            sync_grid_to_settings()
            self.grid_range_vars = range_vars
            self.grid_step_vars = step_vars
            self.update_all_plots()
        
        def update_on_change():
            """Update both plot visualization and grid info when user finishes editing"""
            sync_grid_to_settings()
            update_plot_visualization()
            update_grid_info()
        
        for feat_idx in included_feature_indexes:
            feat_name = self.prop_titles[feat_idx]
            feat_min, feat_max = self.get_feature_range(feat_name)
            if feat_min is None:
                continue
            
            feat_frame_row = ttk.Frame(ranges_frame)
            feat_frame_row.pack(fill=tk.X, pady=1)
            
            ttk.Label(feat_frame_row, text=f"{feat_name}:", width=15).pack(side=tk.LEFT, padx=5)
            
            ttk.Label(feat_frame_row, text="Min:").pack(side=tk.LEFT, padx=2)
            min_val = defaults_dict.get(f'min_{feat_name}', defaults_dict.get(f'min_{feat_idx}', feat_min))
            min_var = tk.DoubleVar(value=float(min_val))
            min_entry = ttk.Entry(feat_frame_row, textvariable=min_var, width=10)
            min_entry.pack(side=tk.LEFT, padx=2)
            # Only update grid when Enter is pressed or focus leaves the box
            min_entry.bind('<Return>', lambda e, f=update_on_change: f())
            min_entry.bind('<FocusOut>', lambda e, f=update_on_change: f())
            
            ttk.Label(feat_frame_row, text="Max:").pack(side=tk.LEFT, padx=2)
            max_val = defaults_dict.get(f'max_{feat_name}', defaults_dict.get(f'max_{feat_idx}', feat_max))
            max_var = tk.DoubleVar(value=float(max_val))
            max_entry = ttk.Entry(feat_frame_row, textvariable=max_var, width=10)
            max_entry.pack(side=tk.LEFT, padx=2)
            # Only update grid when Enter is pressed or focus leaves the box
            max_entry.bind('<Return>', lambda e, f=update_plot_visualization: f())
            max_entry.bind('<FocusOut>', lambda e, f=update_plot_visualization: f())
            
            ttk.Label(feat_frame_row, text="Step:").pack(side=tk.LEFT, padx=2)
            step_default = (feat_max - feat_min) / 100.0
            step_val = defaults_dict.get(f'step_{feat_name}', defaults_dict.get(f'step_{feat_idx}', step_default))
            step_var = tk.DoubleVar(value=float(step_val))
            step_entry = ttk.Entry(feat_frame_row, textvariable=step_var, width=10)
            step_entry.pack(side=tk.LEFT, padx=2)
            # Only update grid when Enter is pressed or focus leaves the box
            step_entry.bind('<Return>', lambda e, f=update_on_change: f())
            step_entry.bind('<FocusOut>', lambda e, f=update_on_change: f())
            
            range_vars[feat_idx] = {'min': min_var, 'max': max_var}
            step_vars[feat_idx] = step_var
        
        # Spatial filter sigmas: only for included features whose title ends with '_pos'
        spatial_feature_indexes = [idx for idx in included_feature_indexes if idx < len(self.prop_titles) and self.prop_titles[idx].endswith('_pos')]
        spatial_frame = ttk.LabelFrame(scrollable_frame, text="Spatial Filter Sigmas (um)", padding="3")
        spatial_frame.pack(fill=tk.X, padx=3, pady=3)
        spatial_filter_vars = {}
        spatial_filter_entries = []
        for feat_idx in spatial_feature_indexes:
            feat_name = self.prop_titles[feat_idx]
            low_var = tk.DoubleVar(value=defaults_dict.get(f'spatial_{feat_name}_low', 0.0))
            high_var = tk.DoubleVar(value=defaults_dict.get(f'spatial_{feat_name}_high', 0.0))
            spatial_filter_vars[feat_idx] = {'low': low_var, 'high': high_var}
            ttk.Label(spatial_frame, text=f"{feat_name} low:").pack(side=tk.LEFT, padx=5)
            low_entry = ttk.Entry(spatial_frame, textvariable=low_var, width=10)
            low_entry.pack(side=tk.LEFT, padx=2)
            ttk.Label(spatial_frame, text=f"{feat_name} high:").pack(side=tk.LEFT, padx=5)
            high_entry = ttk.Entry(spatial_frame, textvariable=high_var, width=10)
            high_entry.pack(side=tk.LEFT, padx=2)
            spatial_filter_entries.append((low_entry, high_entry))
        
        # Memory limit
        memory_frame = ttk.Frame(scrollable_frame)
        memory_frame.pack(fill=tk.X, padx=3, pady=2)
        ttk.Label(memory_frame, text="Max memory (MB):").pack(side=tk.LEFT, padx=5)
        max_memory_var = tk.DoubleVar(value=defaults_dict.get('max_memory_mb', 512.0))
        max_memory_entry = ttk.Entry(memory_frame, textvariable=max_memory_var, width=12)
        max_memory_entry.pack(side=tk.LEFT, padx=2)
        
        # Reg density (regularization added to each voxel count)
        reg_density_frame = ttk.Frame(scrollable_frame)
        reg_density_frame.pack(fill=tk.X, padx=3, pady=2)
        ttk.Label(reg_density_frame, text="Reg density:").pack(side=tk.LEFT, padx=5)
        reg_density_var = tk.DoubleVar(value=defaults_dict.get('reg_density', 1.0))
        ttk.Entry(reg_density_frame, textvariable=reg_density_var, width=12).pack(side=tk.LEFT, padx=2)
        
        # Iteration number (for background/cluster separation)
        it_number_frame = ttk.Frame(scrollable_frame)
        it_number_frame.pack(fill=tk.X, padx=3, pady=2)
        ttk.Label(it_number_frame, text="Iteration number:").pack(side=tk.LEFT, padx=5)
        it_number_var = tk.IntVar(value=int(defaults_dict.get('it_number', 3)))
        ttk.Entry(it_number_frame, textvariable=it_number_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(it_number_frame, text="(background/cluster separation)").pack(side=tk.LEFT, padx=2)
        
        def sync_grid_to_settings():
            """Write current form values into self.settings['grid'] so they apply immediately."""
            d = {}
            d['included_feature_names'] = np.array([self.prop_titles[i] for i in included_feature_indexes], dtype=object)
            for fid in included_feature_indexes:
                name = self.prop_titles[fid]
                d[f'min_{name}'] = range_vars[fid]['min'].get()
                d[f'max_{name}'] = range_vars[fid]['max'].get()
                d[f'step_{name}'] = step_vars[fid].get()
            d['spatial_feature_names'] = np.array([self.prop_titles[i] for i in spatial_feature_indexes], dtype=object)
            for fid in spatial_feature_indexes:
                name = self.prop_titles[fid]
                d[f'spatial_{name}_low'] = spatial_filter_vars[fid]['low'].get()
                d[f'spatial_{name}_high'] = spatial_filter_vars[fid]['high'].get()
            d['max_memory_mb'] = max_memory_var.get()
            d['reg_density'] = reg_density_var.get()
            try:
                d['it_number'] = int(it_number_var.get())
            except (ValueError, tk.TclError):
                d['it_number'] = 3
            self.settings['grid'].update(d)
        
        # Grid info display
        grid_info_frame = ttk.Frame(scrollable_frame)
        grid_info_frame.pack(fill=tk.X, padx=3, pady=2)
        grid_info_label = ttk.Label(grid_info_frame, text="Grid info: Calculating...")
        grid_info_label.pack(side=tk.LEFT, padx=5)
        
        def update_grid_info():
            """Calculate and display total voxels and number of batches (same logic as density calculation, incl. padding)"""
            try:
                grid_ranges = {}
                grid_steps = {}
                for feat_idx in included_feature_indexes:
                    if feat_idx in range_vars and feat_idx in step_vars:
                        grid_ranges[feat_idx] = (range_vars[feat_idx]['min'].get(), range_vars[feat_idx]['max'].get())
                        grid_steps[feat_idx] = step_vars[feat_idx].get()
                if len(grid_ranges) == 0:
                    grid_info_label.config(text="Grid info: No features selected")
                    return
                # Build spatial_filter in grid units (required for batch estimate)
                spatial_filter_grid = {}
                for feat_idx in spatial_filter_vars:
                    if feat_idx in grid_steps and grid_steps[feat_idx] > 0:
                        dim_name = self.prop_titles[feat_idx]
                        step = grid_steps[feat_idx]
                        low = spatial_filter_vars[feat_idx]['low'].get()
                        high = spatial_filter_vars[feat_idx]['high'].get()
                        spatial_filter_grid[dim_name] = (low / step, high / step)
                if 'y_pos' not in spatial_filter_grid:
                    grid_info_label.config(text="Grid info: Set y_pos spatial filter and step for batch estimate")
                    return
                max_memory_mb = max(0.01, max_memory_var.get())
                n_batches, total_voxels = estimate_grid_batches(
                    self.prop_titles, included_feature_indexes, grid_ranges, grid_steps,
                    max_memory_mb, spatial_filter_grid)
                grid_info_label.config(text=f"Total voxels: {total_voxels:,} | Grid batches: {n_batches}")
            except Exception as e:
                grid_info_label.config(text=f"Grid info: Error calculating")
        
        # Update grid info when ranges/steps/memory change
        def update_on_change():
            update_plot_visualization()
            update_grid_info()
        
        # Bind grid info update and settings sync to Enter/FocusOut for memory and spatial sigma entries
        def on_memory_or_spatial_change():
            sync_grid_to_settings()
            update_grid_info()
        max_memory_entry.bind('<Return>', lambda e: on_memory_or_spatial_change())
        max_memory_entry.bind('<FocusOut>', lambda e: on_memory_or_spatial_change())
        for low_entry, high_entry in spatial_filter_entries:
            low_entry.bind('<Return>', lambda e: on_memory_or_spatial_change())
            low_entry.bind('<FocusOut>', lambda e: on_memory_or_spatial_change())
            high_entry.bind('<Return>', lambda e: on_memory_or_spatial_change())
            high_entry.bind('<FocusOut>', lambda e: on_memory_or_spatial_change())
        
        # Grid info also updates when min/max/step change (via update_on_change)
        # Initial calculation
        update_grid_info()
        
        # Buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Initial plot visualization
        update_plot_visualization()
        
        def start_calculation():
            """Start density calculation"""
            # Store current form into consolidated grid defaults (for main-window Save settings)
            save_dict = {}
            save_dict['included_features'] = np.array(included_feature_indexes)
            save_dict['included_feature_names'] = np.array([self.prop_titles[i] for i in included_feature_indexes], dtype=object)
            for feat_idx in included_feature_indexes:
                name = self.prop_titles[feat_idx]
                save_dict[f'min_{name}'] = range_vars[feat_idx]['min'].get()
                save_dict[f'max_{name}'] = range_vars[feat_idx]['max'].get()
                save_dict[f'step_{name}'] = step_vars[feat_idx].get()
            save_dict['spatial_feature_names'] = np.array([self.prop_titles[i] for i in spatial_feature_indexes], dtype=object)
            for feat_idx in spatial_feature_indexes:
                name = self.prop_titles[feat_idx]
                save_dict[f'spatial_{name}_low'] = spatial_filter_vars[feat_idx]['low'].get()
                save_dict[f'spatial_{name}_high'] = spatial_filter_vars[feat_idx]['high'].get()
            save_dict['max_memory_mb'] = max_memory_var.get()
            save_dict['reg_density'] = reg_density_var.get()
            self.settings['grid'].update(save_dict)
            # Build grid ranges and steps
            grid_ranges = {}
            grid_steps = {}
            for feat_idx in included_feature_indexes:
                grid_ranges[feat_idx] = (range_vars[feat_idx]['min'].get(), range_vars[feat_idx]['max'].get())
                grid_steps[feat_idx] = step_vars[feat_idx].get()
            
            # Clear plot visualization and re-enable UI
            self.grid_window = None
            self.grid_range_vars = {}
            self.grid_step_vars = {}
            # Re-enable UI controls
            if self.sorting_combo is not None:
                self.sorting_combo.config(state="readonly")
            if self.unit_combo is not None:
                self.unit_combo.config(state="readonly")
            if self.calc_densities_btn is not None and self.all_properties is not None:
                self.calc_densities_btn.config(state=tk.NORMAL)
            if self.find_seeds_btn is not None and self.cluster_den is not None:
                self.find_seeds_btn.config(state=tk.NORMAL)
            self.update_all_plots()
            
            # Build spatial filter dict (name -> (low_um, high_um))
            spatial_filter = {}
            for feat_idx in spatial_filter_vars:
                name = self.prop_titles[feat_idx]
                spatial_filter[name] = (spatial_filter_vars[feat_idx]['low'].get(), spatial_filter_vars[feat_idx]['high'].get())
            
            # Clear visualization and re-enable UI before starting calculation
            self.grid_window = None
            self.grid_range_vars = {}
            self.grid_step_vars = {}
            # Re-enable UI controls
            if self.sorting_combo is not None:
                self.sorting_combo.config(state="readonly")
            if self.unit_combo is not None:
                self.unit_combo.config(state="readonly")
            if self.calc_densities_btn is not None and self.all_properties is not None:
                self.calc_densities_btn.config(state=tk.NORMAL)
            if self.find_seeds_btn is not None and self.cluster_den is not None:
                self.find_seeds_btn.config(state=tk.NORMAL)
            grid_window.destroy()
            try:
                it_num = int(it_number_var.get())
            except (ValueError, tk.TclError):
                it_num = 3
            self._start_density_calculation(included_feature_indexes, grid_ranges, grid_steps, 
                                          max_memory_var.get(), spatial_filter, reg_density_var.get(), it_number=it_num)
        
        begin_btn = ttk.Button(button_frame, text="Begin", command=start_calculation)
        begin_btn.pack(side=tk.RIGHT, padx=5)
    
    def _start_density_calculation(self, included_feature_indexes, grid_ranges, grid_steps, max_memory_mb, spatial_filter, reg_density=1.0, it_number=3):
        """Start the density calculation process"""
        # Create progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Calculating Densities")
        progress_window.geometry("400x120")
        progress_window.transient(self.root)
        progress_window.grab_set()  # Make modal
        
        progress_label = ttk.Label(progress_window, text="Processing batches...")
        progress_label.pack(pady=10)
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100, length=350)
        progress_bar.pack(pady=10, padx=20, fill=tk.X)
        
        status_label = ttk.Label(progress_window, text="Starting...")
        status_label.pack(pady=5)
        
        # Update window to show it
        progress_window.update()
        
        def progress_callback(current_batch, total_batches):
            """Update progress bar"""
            progress = (current_batch / total_batches) * 100.0
            progress_var.set(progress)
            status_label.config(text=f"Batch {current_batch} of {total_batches}")
            progress_window.update()
        
        try:
            # Filter properties to included features
            properties_filtered = self.all_properties[:, included_feature_indexes]
            n_all = properties_filtered.shape[0]

            # Only unassigned spikes are used for density; assigned get cluster_den/background_den/clust_ind = 0
            if self.unit_id is None:
                self.unit_id = np.full(n_all, -1, dtype=int)
            # Allocate curr_max_prob if missing (only place that may create it; no fallback values elsewhere)
            if self.curr_max_prob is None or len(self.curr_max_prob) != n_all:
                self.curr_max_prob = np.full(n_all, np.inf, dtype=float)
            unassigned_mask = (self.unit_id == -1)
            properties_unassigned = properties_filtered[unassigned_mask]

            # Convert spatial filter from um to grid units
            spatial_filter_grid = {}
            for dim_name, (low_um, high_um) in spatial_filter.items():
                if dim_name in self.prop_titles:
                    dim_idx = self.prop_titles.index(dim_name)
                    if dim_idx in included_feature_indexes:
                        # Convert um to grid units using step size
                        step = grid_steps[dim_idx]
                        low_grid = low_um / step if step > 0 else 0.0
                        high_grid = high_um / step if step > 0 else 0.0
                        spatial_filter_grid[dim_name] = (low_grid, high_grid)

            # Calculate densities on unassigned spikes only
            cluster_den_batch, background_den_batch = calculate_densities_batch(
                properties_unassigned,
                self.prop_titles,
                included_feature_indexes,
                grid_ranges,
                grid_steps,
                max_memory_mb,
                spatial_filter_grid if len(spatial_filter_grid) > 0 else None,
                progress_callback=progress_callback,
                reg_density=reg_density,
                it_number=it_number
            )

            # Full-length arrays: assigned spikes stay 0
            self.cluster_den = np.zeros(n_all, dtype=cluster_den_batch.dtype)
            self.background_den = np.zeros(n_all, dtype=background_den_batch.dtype)
            self.cluster_den[unassigned_mask] = cluster_den_batch
            self.background_den[unassigned_mask] = background_den_batch
            # current_max_prob for unassigned: background density where estimated, else inf
            self.curr_max_prob[unassigned_mask] = np.where(background_den_batch > 0, background_den_batch.astype(float), np.inf)
            self.included_feature_indexes = included_feature_indexes
            self.density_ranges = grid_ranges
            self.grid_steps = grid_steps
            
            # Calculate clust_ind
            if self.background_den is not None:
                self.clust_ind = np.where(self.background_den > 0,
                                         self.cluster_den / self.background_den,
                                         0.0)
            else:
                self.clust_ind = None
            
            # Save densities
            if self.data_folder:
                densities_file = os.path.join(self.data_folder, 'densities.npz')
                np.savez(densities_file,
                        cluster_den=self.cluster_den,
                        background_den=self.background_den,
                        clust_ind=self.clust_ind,
                        included_feature_indexes=np.array(included_feature_indexes))
            
            # Close progress window
            progress_window.destroy()
            
            # Enable find seeds button
            self.find_seeds_btn.config(state=tk.NORMAL)
            
            messagebox.showinfo("Success", "Density calculation complete!")
            self.update_all_plots()
            
        except Exception as e:
            # Close progress window on error
            if 'progress_window' in locals():
                progress_window.destroy()
            messagebox.showerror("Error", f"Density calculation failed: {str(e)}")
    
    def find_local_maxima_seeds(self):
        """Find local maxima seeds"""
        if self.clust_ind is None:
            messagebox.showerror("Error", "Please calculate densities first")
            return
        
        self._open_find_seeds_window()
    
    def _open_find_seeds_window(self):
        """Open window for finding seeds"""
        if self.included_feature_indexes is None:
            messagebox.showerror("Error", "Please calculate densities first")
            return
        
        seeds_window = tk.Toplevel(self.root)
        seeds_window.title("Find Local Maxima Seeds")
        seeds_window.geometry("500x600")
        seeds_window.transient(self.root)  # Keep on top of main window
        seeds_window.lift()  # Bring to front
        seeds_window.focus_force()  # Force focus
        
        # Create scrollable frame
        canvas = tk.Canvas(seeds_window)
        scrollbar = ttk.Scrollbar(seeds_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Defaults from self.settings (populated by load data or Load settings)
        defaults_dict = dict(self.settings.get('seeds', {}))
        
        # Distance per feature (for included features only)
        included_feat_idxs = self.included_feature_indexes
        distance_vars = {}
        
        included_feature_names = [self.prop_titles[i] if i < len(self.prop_titles) else f"Feature {i}" for i in included_feat_idxs]
        for feat_idx in included_feat_idxs:
            feat_name = self.prop_titles[feat_idx] if feat_idx < len(self.prop_titles) else f"Feature {feat_idx}"
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(frame, text=f"Distance in {feat_name}:").pack(side=tk.LEFT, padx=5)
            var = tk.DoubleVar(value=defaults_dict.get(f'distance_{feat_name}', 0.0))
            ttk.Entry(frame, textvariable=var, width=12).pack(side=tk.LEFT, padx=2)
            distance_vars[feat_idx] = var
        
        # Spikes per batch
        batch_frame = ttk.Frame(scrollable_frame)
        batch_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(batch_frame, text="Spikes per batch (in thousands):").pack(side=tk.LEFT, padx=5)
        spikes_per_batch_var = tk.DoubleVar(value=defaults_dict.get('spikes_per_batch_k', 200.0))
        ttk.Entry(batch_frame, textvariable=spikes_per_batch_var, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Label(batch_frame, text="Min cluster index:").pack(side=tk.LEFT, padx=5)
        min_cluster_idx_var = tk.DoubleVar(value=defaults_dict.get('min_cluster_idx', 2))
        ttk.Entry(batch_frame, textvariable=min_cluster_idx_var, width=12).pack(side=tk.LEFT, padx=2)
        
        # Bounds (optional): per-feature min/max; seeds outside are removed
        ttk.Label(scrollable_frame, text="Bounds (optional) - seeds outside [min, max] are removed:").pack(anchor=tk.W, padx=5, pady=(10, 2))
        bound_min_vars = {}
        bound_max_vars = {}
        for feat_idx in included_feat_idxs:
            feat_name = self.prop_titles[feat_idx] if feat_idx < len(self.prop_titles) else f"Feature {feat_idx}"
            bf = ttk.Frame(scrollable_frame)
            bf.pack(fill=tk.X, padx=5, pady=1)
            ttk.Label(bf, text=f"{feat_name} min:").pack(side=tk.LEFT, padx=2)
            _s = defaults_dict.get(f'bound_min_{feat_name}', '')
            vmin = tk.StringVar(value=str(_s).strip() if _s is not None and str(_s).strip() else "")
            ttk.Entry(bf, textvariable=vmin, width=10).pack(side=tk.LEFT, padx=2)
            ttk.Label(bf, text="max:").pack(side=tk.LEFT, padx=2)
            _s = defaults_dict.get(f'bound_max_{feat_name}', '')
            vmax = tk.StringVar(value=str(_s).strip() if _s is not None and str(_s).strip() else "")
            ttk.Entry(bf, textvariable=vmax, width=10).pack(side=tk.LEFT, padx=2)
            bound_min_vars[feat_idx] = vmin
            bound_max_vars[feat_idx] = vmax
        
        def sync_seeds_to_settings():
            """Write current form values into self.settings['seeds'] so they apply immediately."""
            d = {}
            d['included_feature_names'] = np.array(included_feature_names, dtype=object)
            for feat_idx in included_feat_idxs:
                name = self.prop_titles[feat_idx] if feat_idx < len(self.prop_titles) else f"Feature {feat_idx}"
                d[f'distance_{name}'] = distance_vars[feat_idx].get()
                d[f'bound_min_{name}'] = bound_min_vars[feat_idx].get().strip()
                d[f'bound_max_{name}'] = bound_max_vars[feat_idx].get().strip()
            d['spikes_per_batch_k'] = spikes_per_batch_var.get()
            d['min_cluster_idx'] = min_cluster_idx_var.get()
            self.settings['seeds'].update(d)
        
        # Buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def begin_finding():
            """Begin finding seeds"""
            progress_window = None
            try:
                # Store current form into consolidated seeds defaults (for main-window Save settings)
                save_dict = {}
                save_dict['included_feature_names'] = np.array(included_feature_names, dtype=object)
                for feat_idx in included_feat_idxs:
                    name = self.prop_titles[feat_idx] if feat_idx < len(self.prop_titles) else f"Feature {feat_idx}"
                    save_dict[f'distance_{name}'] = distance_vars[feat_idx].get()
                    save_dict[f'bound_min_{name}'] = bound_min_vars[feat_idx].get().strip()
                    save_dict[f'bound_max_{name}'] = bound_max_vars[feat_idx].get().strip()
                save_dict['spikes_per_batch_k'] = spikes_per_batch_var.get()
                save_dict['min_cluster_idx'] = min_cluster_idx_var.get()
                self.settings['seeds'].update(save_dict)
                # Build feature distances and bounds keyed by column index (context with self.all_properties)
                feature_distances = {feat_idx: distance_vars[feat_idx].get() for feat_idx in included_feat_idxs}
                n_cols = self.all_properties.shape[1]
                sorted_feature_idx = int(self.sort_feature_idx) if (0 <= self.sort_feature_idx < n_cols) else 0
                
                # Convert spikes per batch from thousands
                spikes_per_batch = int(spikes_per_batch_var.get() * 1000)
                min_cluster_idx = int(min_cluster_idx_var.get())
                
                seeds_window.destroy()
                
                # Create progress window
                progress_window = tk.Toplevel(self.root)
                progress_window.title("Finding Local Maxima Seeds")
                progress_window.geometry("400x120")
                progress_window.transient(self.root)
                progress_window.grab_set()  # Make modal
                
                progress_label = ttk.Label(progress_window, text="Processing batches...")
                progress_label.pack(pady=10)
                
                progress_var = tk.DoubleVar()
                progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100, length=350)
                progress_bar.pack(pady=10, padx=20, fill=tk.X)
                
                status_label = ttk.Label(progress_window, text="Starting...")
                status_label.pack(pady=5)
                
                # Update window to show it
                progress_window.update()
                
                def progress_callback(current_batch, total_batches):
                    """Update progress bar"""
                    progress = (current_batch / total_batches) * 100.0
                    progress_var.set(progress)
                    status_label.config(text=f"Batch {current_batch} of {total_batches}")
                    progress_window.update()
                
                # Build optional bounds: column index -> (min or None, max or None)
                bounds = None
                try:
                    b = {}
                    for fid in included_feat_idxs:
                        mn = bound_min_vars[fid].get().strip()
                        mx = bound_max_vars[fid].get().strip()
                        if mn != "" or mx != "":
                            min_val = float(mn) if mn != "" else None
                            max_val = float(mx) if mx != "" else None
                            b[fid] = (min_val, max_val)
                    if b:
                        bounds = b
                except (ValueError, tk.TclError):
                    pass
                # Call seed finding function (all indices already in context with properties)
                self.seeds = gm_seed_local_max(
                    self.all_properties,
                    self.clust_ind,
                    spikes_per_batch / 1000.0,  # Convert to thousands
                    min_cluster_idx,
                    feature_distances,
                    sorted_feature_idx,
                    progress_callback=progress_callback,
                    bounds=bounds
                )
                
                # Destroy progress window
                if progress_window is not None:
                    progress_window.destroy()
                    progress_window = None
                
                if len(self.seeds) > 0:
                    messagebox.showinfo("Success", f"Found {len(self.seeds)} seeds")
                    # Plot seeds on graphs (seeds are now plotted in update_plot)
                    self.update_all_plots()
                    
                    # Enable compute Gaussian models button if in Gaussian model mode
                    if self.sorting_type.get() == "Single seed":
                        self.compute_gaussian_models_button.config(state=tk.NORMAL)
                else:
                    messagebox.showwarning("Warning", "No seeds found")
                    
            except Exception as e:
                # Destroy progress window if it exists
                if progress_window is not None:
                    progress_window.destroy()
                messagebox.showerror("Error", f"Seed finding failed: {str(e)}")
        
        begin_btn = ttk.Button(button_frame, text="Begin", command=begin_finding)
        begin_btn.pack(side=tk.RIGHT, padx=5)
    
    def compute_gaussian_models(self):
        """Run Gaussian model computation using current/saved parameters (no settings window)."""
        if self.seeds is None or len(self.seeds) == 0:
            messagebox.showerror("Error", "No seeds defined. Please find seeds first.")
            return
        if self.included_feature_indexes is None:
            messagebox.showerror("Error", "Please calculate densities first to define included features.")
            return
        settings = self._get_gaussian_model_settings()
        if settings is None:
            return
        included_feat_idxs = self.included_feature_indexes
        properties_filtered = self.all_properties[:, included_feat_idxs]
        self._compute_gaussian_models_loop(properties_filtered, settings)
    
    def _get_gaussian_model_settings(self):
        """Return settings dict for GM computation from self.settings['gm'].
        Returned dict has initial_stds, max_range_sorted, n_samples_density_curve."""
        if self.included_feature_indexes is None:
            return None
        included_feat_idxs = list(self.included_feature_indexes)
        n_included = len(included_feat_idxs)
        defaults_dict = dict(self.settings.get('gm', {}))
        default_vals_initial = {0: 1.5, 1: 0.5, 2: 0.1, 3: 0.025, 4: 0.05, 5: 0.2}
        init_range = defaults_dict.get('init_range', None)
        if init_range is not None and len(np.atleast_1d(init_range)) == n_included:
            init_range = np.asarray(init_range).ravel()[:n_included]
        else:
            init_range = np.array([float(defaults_dict.get(f'init_range_{self.prop_titles[f] if f < len(self.prop_titles) else f"Feature {f}"}', default_vals_initial.get(f, 1.0) * 2.0)) for f in included_feat_idxs])
        initial_stds = init_range / 2.0
        sorted_feature_title = self.prop_titles[self.sort_feature_idx] if (getattr(self, 'sort_feature_idx', None) is not None and self.sort_feature_idx < len(self.prop_titles)) else "sorted feature"
        max_range_sorted = float(defaults_dict.get('max_range_sorted', defaults_dict.get(f'max_range_{sorted_feature_title}', 4.0)))
        sorted_feature_idx = included_feat_idxs.index(self.sort_feature_idx) if self.sort_feature_idx in included_feat_idxs else 0
        n_props = len(self.prop_titles) if self.prop_titles else self.all_properties.shape[1]
        prop_titles = list(self.prop_titles) if self.prop_titles else [f"Feature {i}" for i in range(n_props)]
        viz_pairs = [
            (included_feat_idxs[0], included_feat_idxs[min(1, n_included - 1)]),
            (included_feat_idxs[0], included_feat_idxs[min(2, n_included - 1)]),
            (included_feat_idxs[min(3, n_included - 1)], included_feat_idxs[min(4, n_included - 1)]),
            (included_feat_idxs[0], included_feat_idxs[min(5, n_included - 1)])
        ]
        if 'viz_feature_pairs_names' in defaults_dict:
            raw_names = defaults_dict['viz_feature_pairs_names']
            if hasattr(raw_names, 'shape') and raw_names.shape[0] >= 1:
                n_saved = raw_names.shape[0]
                viz_pairs = []
                for i in range(n_saved):
                    n0, n1 = str(raw_names[i, 0]), str(raw_names[i, 1])
                    idx0 = prop_titles.index(n0) if n0 in prop_titles else 0
                    idx1 = prop_titles.index(n1) if n1 in prop_titles else min(1, n_props - 1)
                    viz_pairs.append((idx0, idx1))
            elif hasattr(raw_names, '__len__') and len(raw_names) >= 1:
                viz_pairs = []
                for i in range(len(raw_names)):
                    p = raw_names[i]
                    if hasattr(p, '__len__') and len(p) >= 2:
                        n0, n1 = str(p[0]), str(p[1])
                        idx0 = prop_titles.index(n0) if n0 in prop_titles else 0
                        idx1 = prop_titles.index(n1) if n1 in prop_titles else min(1, n_props - 1)
                        viz_pairs.append((idx0, idx1))
        elif 'viz_feature_pairs' in defaults_dict and hasattr(defaults_dict['viz_feature_pairs'], 'shape') and defaults_dict['viz_feature_pairs'].shape[0] >= 1:
            arr = defaults_dict['viz_feature_pairs']
            viz_pairs = [(int(arr[i, 0]), int(arr[i, 1])) for i in range(arr.shape[0])]
        elif 'viz_feature_pairs' in defaults_dict and hasattr(defaults_dict['viz_feature_pairs'], '__len__') and len(defaults_dict['viz_feature_pairs']) >= 1:
            raw = defaults_dict['viz_feature_pairs']
            viz_pairs = [(int(p[0]), int(p[1])) for p in raw]
        settings = {
            'initial_stds': initial_stds,
            'max_range_sorted': max_range_sorted,
            'n_samples_density_curve': int(defaults_dict.get('n_samples_density_curve', 101)),
            'sorted_feature_idx': sorted_feature_idx,
            'init_mah_th': float(defaults_dict.get('init_mah_th', 1.0)),
            'com_iteration_threshold': float(defaults_dict.get('com_iteration_threshold', 0.25)),
            'com_iteration_max_iterations': int(defaults_dict.get('com_iteration_max_iterations', 50)),
            'density_threshold_for_init_distance': float(defaults_dict.get('density_threshold_for_init_distance', 0.1)),
            'dist_step': float(defaults_dict.get('dist_step', 0.1)),
            'multi_cluster_threshold': float(defaults_dict.get('multi_cluster_threshold', 0.2)),
            'min_change': float(defaults_dict.get('min_change', 0.01)),
            'max_iter_for_model': int(defaults_dict.get('max_iter_for_model', 500)),
            'gaussian_filter_sigma': float(defaults_dict.get('gaussian_filter_sigma', 50.0)),
            'min_points_for_cluster': int(defaults_dict.get('min_points_for_cluster', 100)),
            'viz_feature_pairs': viz_pairs,
            'report_failures': bool(defaults_dict.get('report_failures', False)),
            'skip_user_refinement': bool(defaults_dict.get('skip_user_refinement', False)),
        }
        return settings
    
    def _open_gaussian_model_settings_window(self):
        """Open settings window for Gaussian model computation. Always create a fresh window so form shows current dict (no stale Entry values)."""
        if self.all_properties is None:
            messagebox.showwarning("Warning", "Load data first.")
            return
        existing = getattr(self, '_gaussian_settings_window', None)
        if existing is not None and existing.winfo_exists():
            existing.destroy()
            self._gaussian_settings_window = None
        settings_window = tk.Toplevel(self.root)
        self._gaussian_settings_window = settings_window
        settings_window.title("Gaussian Model Settings")
        settings_window.geometry("620x700")

        # Create scrollable frame
        canvas = tk.Canvas(settings_window)
        scrollbar = ttk.Scrollbar(settings_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Settings variables
        settings_vars = {}
        # Use included features if set, else fallback so we can still show/edit params (e.g. before densities)
        if self.included_feature_indexes is not None:
            included_feat_idxs = list(self.included_feature_indexes)
        else:
            included_feat_idxs = list(range(min(6, self.all_properties.shape[1])))
        n_included = len(included_feat_idxs)
        
        # Defaults from self.settings (populated by load data or Load settings)
        defaults_dict = dict(self.settings.get('gm', {}))
        sorted_feature_title = self.prop_titles[self.sort_feature_idx] if (getattr(self, 'sort_feature_idx', None) is not None and self.sort_feature_idx < len(self.prop_titles)) else "sorted feature"
        _max_sorted = defaults_dict.get('max_range_sorted', defaults_dict.get(f'max_range_{sorted_feature_title}', 4.0))

        # --- General settings ---
        general_frame = ttk.LabelFrame(scrollable_frame, text="General settings", padding="5")
        general_frame.pack(fill=tk.X, padx=5, pady=5)
        row_max = ttk.Frame(general_frame)
        row_max.pack(fill=tk.X, pady=2)
        ttk.Label(row_max, text=f"Max range over {sorted_feature_title} (sorted feature):").pack(side=tk.LEFT, padx=5)
        settings_vars['max_range_sorted'] = tk.DoubleVar(value=float(_max_sorted))
        ttk.Entry(row_max, textvariable=settings_vars['max_range_sorted'], width=12).pack(side=tk.LEFT, padx=2)
        row_min_pts = ttk.Frame(general_frame)
        row_min_pts.pack(fill=tk.X, pady=2)
        ttk.Label(row_min_pts, text="Minimum cluster size:").pack(side=tk.LEFT, padx=5)
        settings_vars['min_points_for_cluster'] = tk.IntVar(value=int(defaults_dict.get('min_points_for_cluster', 100)))
        ttk.Entry(row_min_pts, textvariable=settings_vars['min_points_for_cluster'], width=12).pack(side=tk.LEFT, padx=2)

        # --- Initial model for COM iteration ---
        def _safe_range_array(arr, n, default_val):
            """Get 1D array of length n from saved arr (handles None, 0-d, wrong length, very large floats)."""
            if arr is None:
                return np.array([float(default_val)] * n)
            arr = np.atleast_1d(np.asarray(arr, dtype=float))
            arr = arr.ravel()
            if len(arr) >= n:
                return np.array(arr[:n], dtype=float)
            out = np.array([float(default_val)] * n, dtype=float)
            out[:len(arr)] = arr
            return out
        _arr_init = _safe_range_array(defaults_dict.get('init_range', None), n_included, 2.0)
        default_init = np.array([float(defaults_dict.get(f'init_range_{self.prop_titles[feat_idx] if feat_idx < len(self.prop_titles) else f"Feature {feat_idx}"}', _arr_init[i])) for i, feat_idx in enumerate(included_feat_idxs)])

        initial_model_frame = ttk.LabelFrame(scrollable_frame, text="Initial model for COM iteration", padding="5")
        initial_model_frame.pack(fill=tk.X, padx=5, pady=5)
        # One row: label "Range" on left, then one column per feature with title on top and Entry below
        range_row = ttk.Frame(initial_model_frame)
        range_row.pack(fill=tk.X, pady=2)
        ttk.Label(range_row, text="Range", width=12, anchor='w').pack(side=tk.LEFT, padx=5)
        init_range_vars = {}
        for i, feat_idx in enumerate(included_feat_idxs):
            col = ttk.Frame(range_row)
            col.pack(side=tk.LEFT, padx=4)
            feat_name = self.prop_titles[feat_idx] if feat_idx < len(self.prop_titles) else f"Feat {feat_idx}"
            ttk.Label(col, text=feat_name, anchor='w').pack(fill=tk.X)
            v_init = tk.DoubleVar(master=settings_window, value=float(default_init[i]))
            ttk.Entry(col, textvariable=v_init, width=10).pack(fill=tk.X, padx=2)
            init_range_vars[feat_idx] = v_init
        settings_vars['init_range_vars'] = init_range_vars
        settings_window.after_idle(lambda: [init_range_vars[feat_idx].set(float(default_init[i])) for i, feat_idx in enumerate(included_feat_idxs)])

        row_bic = ttk.Frame(initial_model_frame)
        row_bic.pack(fill=tk.X, pady=2)
        ttk.Label(row_bic, text="Multi-cluster BIC threshold:").pack(side=tk.LEFT, padx=5)
        settings_vars['multi_cluster_threshold'] = tk.DoubleVar(value=float(defaults_dict.get('multi_cluster_threshold', 0.2)))
        ttk.Entry(row_bic, textvariable=settings_vars['multi_cluster_threshold'], width=12).pack(side=tk.LEFT, padx=2)

        row_size = ttk.Frame(initial_model_frame)
        row_size.pack(fill=tk.X, pady=2)
        ttk.Label(row_size, text="Size:").pack(side=tk.LEFT, padx=5)
        settings_vars['init_mah_th'] = tk.DoubleVar(value=float(defaults_dict.get('init_mah_th', 1.0)))
        ttk.Entry(row_size, textvariable=settings_vars['init_mah_th'], width=12).pack(side=tk.LEFT, padx=2)

        # --- Settings for COM iteration ---
        com_frame = ttk.LabelFrame(scrollable_frame, text="Settings for COM iteration", padding="5")
        com_frame.pack(fill=tk.X, padx=5, pady=5)
        for key, label_text, default_val, use_int in [
            ('com_iteration_threshold', 'COM accuracy (relative to model range):', 0.25, False),
            ('com_iteration_max_iterations', 'Max iterations:', 50, True),
        ]:
            row = ttk.Frame(com_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label_text).pack(side=tk.LEFT, padx=5)
            var = tk.IntVar(value=int(defaults_dict.get(key, default_val))) if use_int else tk.DoubleVar(value=defaults_dict.get(key, default_val))
            ttk.Entry(row, textvariable=var, width=12).pack(side=tk.LEFT, padx=2)
            settings_vars[key] = var

        # --- Probability density curve estimation ---
        prob_curve_frame = ttk.LabelFrame(scrollable_frame, text="Probability density curve estimation", padding="5")
        prob_curve_frame.pack(fill=tk.X, padx=5, pady=5)
        row_dens = ttk.Frame(prob_curve_frame)
        row_dens.pack(fill=tk.X, pady=2)
        ttk.Label(row_dens, text="Density at initial model bounds (relative to center):").pack(side=tk.LEFT, padx=5)
        settings_vars['density_threshold_for_init_distance'] = tk.DoubleVar(value=float(defaults_dict.get('density_threshold_for_init_distance', 0.1)))
        ttk.Entry(row_dens, textvariable=settings_vars['density_threshold_for_init_distance'], width=12).pack(side=tk.LEFT, padx=2)
        row_sigma = ttk.Frame(prob_curve_frame)
        row_sigma.pack(fill=tk.X, pady=2)
        ttk.Label(row_sigma, text="Gaussian filter sigma (over samples):").pack(side=tk.LEFT, padx=5)
        settings_vars['gaussian_filter_sigma'] = tk.DoubleVar(value=float(defaults_dict.get('gaussian_filter_sigma', 50.0)))
        ttk.Entry(row_sigma, textvariable=settings_vars['gaussian_filter_sigma'], width=12).pack(side=tk.LEFT, padx=2)
        row_n_samp = ttk.Frame(prob_curve_frame)
        row_n_samp.pack(fill=tk.X, pady=2)
        ttk.Label(row_n_samp, text="N of probability density curve:").pack(side=tk.LEFT, padx=5)
        settings_vars['n_samples_density_curve'] = tk.IntVar(value=int(defaults_dict.get('n_samples_density_curve', 101)))
        ttk.Entry(row_n_samp, textvariable=settings_vars['n_samples_density_curve'], width=12).pack(side=tk.LEFT, padx=2)

        # --- Settings for GM iteration ---
        gm_frame = ttk.LabelFrame(scrollable_frame, text="Settings for GM iteration", padding="5")
        gm_frame.pack(fill=tk.X, padx=5, pady=5)
        for key, label_text, default_val, use_int in [
            ('dist_step', 'Size fitting resolution:', 0.1, False),
            ('min_change', 'Minimum change in cluster size for stability (proportional):', 0.01, False),
            ('max_iter_for_model', 'Max iterations:', 500, True),
        ]:
            row = ttk.Frame(gm_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label_text).pack(side=tk.LEFT, padx=5)
            var = tk.IntVar(value=int(defaults_dict.get(key, default_val))) if use_int else tk.DoubleVar(value=defaults_dict.get(key, default_val))
            ttk.Entry(row, textvariable=var, width=12).pack(side=tk.LEFT, padx=2)
            settings_vars[key] = var

        # Views for user inspections: variable number of plots (x, y by property)
        all_prop_titles = list(self.prop_titles) if self.prop_titles else [f"Feature {i}" for i in range(self.all_properties.shape[1])]
        n_props = len(all_prop_titles)
        default_pairs_global = [
            (included_feat_idxs[0], included_feat_idxs[min(1, n_included - 1)]),
            (included_feat_idxs[0], included_feat_idxs[min(2, n_included - 1)]),
            (included_feat_idxs[min(3, n_included - 1)], included_feat_idxs[min(4, n_included - 1)]),
            (included_feat_idxs[0], included_feat_idxs[min(5, n_included - 1)])
        ]
        saved_pairs_names = defaults_dict.get('viz_feature_pairs_names', None)
        saved_pairs = defaults_dict.get('viz_feature_pairs', None)
        if saved_pairs_names is not None and hasattr(saved_pairs_names, 'shape') and saved_pairs_names.shape[0] >= 1:
            n_saved = saved_pairs_names.shape[0]
            default_pairs_global = []
            for i in range(n_saved):
                n0, n1 = str(saved_pairs_names[i, 0]), str(saved_pairs_names[i, 1])
                idx0 = all_prop_titles.index(n0) if n0 in all_prop_titles else 0
                idx1 = all_prop_titles.index(n1) if n1 in all_prop_titles else min(1, n_props - 1)
                default_pairs_global.append((idx0, idx1))
        elif saved_pairs_names is not None and hasattr(saved_pairs_names, '__len__') and len(saved_pairs_names) >= 1:
            default_pairs_global = []
            for i in range(len(saved_pairs_names)):
                p = saved_pairs_names[i]
                if hasattr(p, '__len__') and len(p) >= 2:
                    n0, n1 = str(p[0]), str(p[1])
                    idx0 = all_prop_titles.index(n0) if n0 in all_prop_titles else 0
                    idx1 = all_prop_titles.index(n1) if n1 in all_prop_titles else min(1, n_props - 1)
                    default_pairs_global.append((idx0, idx1))
        elif saved_pairs is not None and hasattr(saved_pairs, 'shape') and saved_pairs.shape[0] >= 1:
            n_saved = saved_pairs.shape[0]
            default_pairs_global = [(int(saved_pairs[i, 0]), int(saved_pairs[i, 1])) for i in range(n_saved)]
            for i in range(len(default_pairs_global)):
                p0, p1 = default_pairs_global[i]
                if p0 >= n_props or p1 >= n_props:
                    default_pairs_global[i] = (min(p0, n_props - 1), min(p1, n_props - 1))
        elif saved_pairs is not None and hasattr(saved_pairs, '__len__') and len(saved_pairs) >= 1:
            default_pairs_global = []
            for i in range(len(saved_pairs)):
                p = saved_pairs[i]
                if hasattr(p, '__len__') and len(p) >= 2:
                    p0, p1 = int(p[0]), int(p[1])
                    p0 = min(max(0, p0), n_props - 1) if n_props > 0 else 0
                    p1 = min(max(0, p1), n_props - 1) if n_props > 0 else 0
                    default_pairs_global.append((p0, p1))
        viz_pairs_frame = ttk.LabelFrame(scrollable_frame, text="Views for user inspections:", padding="5")
        viz_pairs_frame.pack(fill=tk.X, padx=5, pady=5)
        settings_vars['viz_feature_pairs'] = []
        settings_vars['_viz_feature_names'] = all_prop_titles
        settings_vars['_viz_rows_container'] = []  # list of (row_frame, cb0, cb1) for add/remove

        def add_viz_plot(prop_idx0=None, prop_idx1=None):
            i = len(settings_vars['viz_feature_pairs'])
            row_frame = ttk.Frame(viz_pairs_frame)
            row_frame.pack(fill=tk.X, pady=2)
            ttk.Label(row_frame, text=f"Plot {i + 1} (x, y):", width=14, anchor='w').pack(side=tk.LEFT, padx=5)
            idx0 = int(prop_idx0) if prop_idx0 is not None else min(0, n_props - 1)
            idx1 = int(prop_idx1) if prop_idx1 is not None else min(1, n_props - 1)
            idx0 = max(0, min(idx0, n_props - 1))
            idx1 = max(0, min(idx1, n_props - 1))
            cb0 = ttk.Combobox(row_frame, values=all_prop_titles, state='readonly', width=18)
            cb0.set(all_prop_titles[idx0])
            cb0.pack(side=tk.LEFT, padx=2)
            cb1 = ttk.Combobox(row_frame, values=all_prop_titles, state='readonly', width=18)
            cb1.set(all_prop_titles[idx1])
            cb1.pack(side=tk.LEFT, padx=2)
            settings_vars['viz_feature_pairs'].append((cb0, cb1))
            settings_vars['_viz_rows_container'].append((row_frame, cb0, cb1))
            remove_btn.config(state=tk.NORMAL if len(settings_vars['viz_feature_pairs']) > 1 else tk.DISABLED)

        def remove_viz_plot():
            if len(settings_vars['viz_feature_pairs']) <= 1:
                return
            settings_vars['viz_feature_pairs'].pop()
            row_tuple = settings_vars['_viz_rows_container'].pop()
            row_tuple[0].destroy()
            remove_btn.config(state=tk.NORMAL if len(settings_vars['viz_feature_pairs']) > 1 else tk.DISABLED)

        add_btn = ttk.Button(viz_pairs_frame, text="Add plot", command=lambda: add_viz_plot())
        add_btn.pack(side=tk.LEFT, padx=5, pady=2)
        remove_btn = ttk.Button(viz_pairs_frame, text="Remove plot", command=remove_viz_plot)
        remove_btn.pack(side=tk.LEFT, padx=5, pady=2)
        for i in range(len(default_pairs_global)):
            p0, p1 = default_pairs_global[i]
            add_viz_plot(p0, p1)
        if len(settings_vars['viz_feature_pairs']) == 0:
            add_viz_plot(0, min(1, n_props - 1))
        remove_btn.config(state=tk.DISABLED if len(settings_vars['viz_feature_pairs']) <= 1 else tk.NORMAL)

        # Report failures: when checked, show failure reason and accept/reject-style visualization on fit failure
        report_failures_var = tk.BooleanVar(value=bool(defaults_dict.get('report_failures', False)))
        report_failures_frame = ttk.Frame(scrollable_frame)
        report_failures_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(report_failures_frame, text="Report failures (show reason and visualization when a seed fails)",
                        variable=report_failures_var).pack(anchor='w')
        settings_vars['report_failures'] = report_failures_var

        # Skip user refinement: when checked, auto-accept all successful fits and show progress window
        skip_refinement_var = tk.BooleanVar(value=bool(defaults_dict.get('skip_user_refinement', False)))
        skip_refinement_frame = ttk.Frame(scrollable_frame)
        skip_refinement_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(skip_refinement_frame, text="Skip user refinement (auto-accept successful fits; show progress)",
                        variable=skip_refinement_var).pack(anchor='w')
        settings_vars['skip_user_refinement'] = skip_refinement_var

        # Buttons at bottom
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def test_settings():
            """Run fit with debug callbacks to preview settings. Single seed: 3 windows. Cluster: 2 windows (init+HC, final). Errors always shown in test."""
            try:
                properties_filtered = self.all_properties[:, included_feat_idxs]
                settings = {}
                irv = settings_vars['init_range_vars']
                init_range = np.array([float(irv[feat_idx].get()) for feat_idx in included_feat_idxs])
                settings['initial_stds'] = init_range / 2.0
                settings['max_range_sorted'] = float(settings_vars['max_range_sorted'].get())
                settings['n_samples_density_curve'] = int(settings_vars['n_samples_density_curve'].get())
                settings['sorted_feature_idx'] = included_feat_idxs.index(self.sort_feature_idx) if self.sort_feature_idx in included_feat_idxs else 0
                for key in ['init_mah_th', 'com_iteration_threshold', 'com_iteration_max_iterations',
                           'density_threshold_for_init_distance', 'dist_step',
                           'multi_cluster_threshold', 'min_change', 'max_iter_for_model',
                           'gaussian_filter_sigma']:
                    val = settings_vars[key].get()
                    settings[key] = int(val) if key in ('com_iteration_max_iterations', 'max_iter_for_model') else float(val)
                settings['min_points_for_cluster'] = int(settings_vars['min_points_for_cluster'].get())
                names = settings_vars['_viz_feature_names']
                viz_pairs = []
                for i in range(len(settings_vars['viz_feature_pairs'])):
                    v0 = settings_vars['viz_feature_pairs'][i][0].get()
                    v1 = settings_vars['viz_feature_pairs'][i][1].get()
                    idx0 = names.index(v0) if v0 in names else 0
                    idx1 = names.index(v1) if v1 in names else 0
                    viz_pairs.append((idx0, idx1))
                settings['viz_feature_pairs'] = viz_pairs
                settings['report_failures'] = bool(settings_vars['report_failures'].get())
                settings['unit_id'] = self.unit_id if self.unit_id is not None else np.full(self.all_properties.shape[0], -1, dtype=int)
                n_spikes = self.all_properties.shape[0]
                self._require_curr_max_prob(n_spikes)
                settings['curr_max_prob'] = self.curr_max_prob
                incl = list(included_feat_idxs)

                if self.sorting_type.get() == "Cluster":
                    cluster_seeds = getattr(self, 'cluster_seeds', [])
                    if not cluster_seeds or len(cluster_seeds) == 0:
                        messagebox.showerror("Test settings", "No cluster seeds defined. Define a seed cluster first.")
                        return
                    spike_ids = np.asarray(cluster_seeds[0]['spike_ids'], dtype=int)
                    settings['_test_properties'] = properties_filtered
                    def _init_cluster_cb(d):
                        d['included_feat_idxs'] = incl
                        d['show_seed'] = False
                        if hasattr(self, 'all_properties') and self.all_properties is not None and 'valid_indices' in d:
                            vi = np.asarray(d['valid_indices'], dtype=int)
                            if len(vi) > 0 and vi.max() < self.all_properties.shape[0]:
                                d['all_points_full'] = self.all_properties[vi].copy()
                        self._show_post_com_mahal_debug(d, settings)
                    def _after_gm_cb_cluster(viz, model):
                        win_idx = viz.get('window_indices')
                        viz_for_final = dict(viz)
                        if win_idx is not None:
                            viz_for_final['valid_indices'] = win_idx
                        if hasattr(self, 'all_properties') and self.all_properties is not None and win_idx is not None:
                            vi = np.asarray(win_idx, dtype=int)
                            if len(vi) > 0 and vi.max() < self.all_properties.shape[0]:
                                viz_for_final['all_points_full'] = self.all_properties[vi].copy()
                        self._show_final_gm_debug(viz_for_final, model, settings)
                    settings['debug_init_cluster_callback'] = _init_cluster_cb
                    settings['debug_after_gm_callback'] = _after_gm_cb_cluster
                    cluster_dens = self.cluster_den if self.cluster_den is not None else np.ones(properties_filtered.shape[0], dtype=float)
                    result = make_gaussian_model_from_cluster(
                        properties_filtered, cluster_dens, self.curr_max_prob, spike_ids, settings
                    )
                    if not result.get('success'):
                        messagebox.showerror("Test settings", result.get('message', "Cluster fit failed"))
                    return

                if not self.seeds or len(self.seeds) == 0:
                    messagebox.showerror("Test settings", "No seeds available. Compute densities and find seeds first.")
                    return
                def _init_cb(d):
                    d['included_feat_idxs'] = incl
                    if hasattr(self, 'all_properties') and self.all_properties is not None and 'valid_indices' in d:
                        vi = np.asarray(d['valid_indices'], dtype=int)
                        if len(vi) > 0 and vi.max() < self.all_properties.shape[0]:
                            d['all_points_full'] = self.all_properties[vi].copy()
                    self._show_init_model_debug(d, settings)
                def _post_com_cb(d):
                    d['included_feat_idxs'] = incl
                    if hasattr(self, 'all_properties') and self.all_properties is not None and 'valid_indices' in d:
                        vi = np.asarray(d['valid_indices'], dtype=int)
                        if len(vi) > 0 and vi.max() < self.all_properties.shape[0]:
                            d['all_points_full'] = self.all_properties[vi].copy()
                    self._show_post_com_mahal_debug(d, settings)
                def _after_gm_cb(viz, model):
                    vi = viz.get('window_indices') or viz.get('valid_indices')
                    if hasattr(self, 'all_properties') and self.all_properties is not None and vi is not None:
                        vi = np.asarray(vi, dtype=int)
                        if len(vi) > 0 and vi.max() < self.all_properties.shape[0]:
                            viz = dict(viz)
                            viz['all_points_full'] = self.all_properties[vi].copy()
                    self._show_final_gm_debug(viz, model, settings)
                settings['debug_init_callback'] = _init_cb
                settings['debug_after_com_callback'] = _post_com_cb
                settings['debug_after_gm_callback'] = _after_gm_cb
                first_seed = self.seeds[0]
                cluster_dens = self.cluster_den if self.cluster_den is not None else np.ones(properties_filtered.shape[0], dtype=float)
                result = fit_gaussian_model_from_seed(properties_filtered, first_seed, cluster_dens, self.curr_max_prob, settings)
                # Test mode: always show errors regardless of report_failures; window 3 shown by callback on success
                if isinstance(result, dict) and not result.get('success', True):
                    hist = result.get('stability_iteration_history')
                    if hist and len(hist) > 0:
                        self._show_stability_iteration_debug(hist, result.get('message', 'Fit failed'))
                    else:
                        messagebox.showerror("Test settings", result.get('message', "Fit failed"))
            except Exception as e:
                messagebox.showerror("Test settings", str(e))

        test_btn = ttk.Button(button_frame, text="Test settings", command=test_settings)
        test_btn.pack(side=tk.RIGHT, padx=5)

        def build_settings_from_form():
            """Build settings dict from current form values (for saving or use by Compute Gaussian models)."""
            settings = {}
            irv = settings_vars['init_range_vars']
            init_range = np.array([float(irv[feat_idx].get()) for feat_idx in included_feat_idxs], dtype=float)
            settings['initial_stds'] = init_range / 2.0
            settings['init_range'] = np.array(init_range, dtype=float)
            for feat_idx in included_feat_idxs:
                title = self.prop_titles[feat_idx] if feat_idx < len(self.prop_titles) else f"Feature {feat_idx}"
                settings[f'init_range_{title}'] = float(irv[feat_idx].get())
            settings['max_range_sorted'] = float(settings_vars['max_range_sorted'].get())
            settings['n_samples_density_curve'] = int(settings_vars['n_samples_density_curve'].get())
            settings['sorted_feature_idx'] = included_feat_idxs.index(self.sort_feature_idx) if self.sort_feature_idx in included_feat_idxs else 0
            for key in ['init_mah_th', 'com_iteration_threshold', 'com_iteration_max_iterations',
                        'density_threshold_for_init_distance', 'dist_step',
                        'multi_cluster_threshold', 'min_change', 'max_iter_for_model',
                        'gaussian_filter_sigma']:
                val = settings_vars[key].get()
                settings[key] = int(val) if key in ('com_iteration_max_iterations', 'max_iter_for_model') else float(val)
            settings['min_points_for_cluster'] = int(settings_vars['min_points_for_cluster'].get())
            names = settings_vars['_viz_feature_names']
            pair_names = []
            viz_pairs = []
            for i in range(len(settings_vars['viz_feature_pairs'])):
                v0 = settings_vars['viz_feature_pairs'][i][0].get()
                v1 = settings_vars['viz_feature_pairs'][i][1].get()
                pair_names.append((str(v0), str(v1)))
                idx0 = names.index(v0) if v0 in names else 0
                idx1 = names.index(v1) if v1 in names else 0
                viz_pairs.append((idx0, idx1))
            settings['viz_feature_pairs_names'] = np.array(pair_names, dtype=object)
            settings['viz_feature_pairs'] = viz_pairs
            settings['report_failures'] = bool(settings_vars['report_failures'].get())
            settings['skip_user_refinement'] = bool(settings_vars['skip_user_refinement'].get())
            return settings

        def on_close():
            """Write current form values into self.settings['gm'] and save to disk so they persist when reopening."""
            try:
                new_gm = build_settings_from_form()
                if 'gm' not in self.settings:
                    self.settings['gm'] = {}
                for k, v in new_gm.items():
                    self.settings['gm'][k] = v
                self._save_last_settings()
            except Exception as e:
                messagebox.showwarning("Settings not saved", f"Could not save Gaussian model settings (e.g. invalid or very large values): {e}")
            if getattr(self, '_gaussian_settings_window', None) is settings_window:
                self._gaussian_settings_window = None
            settings_window.destroy()

        settings_window.protocol("WM_DELETE_WINDOW", on_close)
        close_btn = ttk.Button(button_frame, text="Close", command=on_close)
        close_btn.pack(side=tk.RIGHT, padx=5)

        def _fit_settings_to_content():
            try:
                settings_window.update_idletasks()
                rw = settings_window.winfo_reqwidth()
                rh = settings_window.winfo_reqheight()
                sw = settings_window.winfo_screenwidth()
                sh = settings_window.winfo_screenheight()
                min_w = 620
                min_h = 400
                w = min(max(rw, min_w), sw)
                h = min(max(rh, min_h), sh)
                settings_window.geometry(f"{w}x{h}+0+0")
            except tk.TclError:
                pass
        settings_window.after(150, _fit_settings_to_content)

    def _compute_gaussian_models_loop(self, properties_filtered, settings):
        """Loop through seeds and compute Gaussian models. Only unassigned (unit_id == -1) data are used."""
        n_seeds = len(self.seeds)
        if self.unit_id is None:
            self.unit_id = np.full(self.all_properties.shape[0], -1, dtype=int)
        n_spikes = self.all_properties.shape[0]
        self._require_curr_max_prob(n_spikes)
        self._require_background_den(n_spikes)
        settings = dict(settings)
        settings['unit_id'] = self.unit_id
        settings['curr_max_prob'] = self.curr_max_prob

        included_feat_idxs = self.included_feature_indexes
        prev_fail_viz_popup, prev_fail_viz_fig = None, None

        skip_refinement = settings.get('skip_user_refinement', False)
        progress_win, progress_label, progress_bar = None, None, None
        if skip_refinement and n_seeds > 0:
            progress_win = tk.Toplevel(self.root)
            progress_win.title("Gaussian models progress")
            progress_win.transient(self.root)
            progress_win.geometry("400x100")
            progress_label = ttk.Label(progress_win, text=f"Remaining seeds: {len(self.seeds)}")
            progress_label.pack(pady=(15, 5))
            progress_bar = ttk.Progressbar(progress_win, maximum=n_seeds, value=0, length=360)
            progress_bar.pack(pady=5)
            progress_win.update_idletasks()
            progress_win.update()

        cluster_dens = self.cluster_den if self.cluster_den is not None else np.ones(properties_filtered.shape[0], dtype=float)
        # Order seeds by cluster density (highest first)
        seeds_by_density = sorted(list(self.seeds), key=lambda s: float(cluster_dens[s]), reverse=True)

        def _run_completion():
            if progress_win is not None and progress_win.winfo_exists():
                try:
                    progress_bar['value'] = n_seeds
                    progress_label.config(text="Remaining seeds: 0")
                    progress_win.update_idletasks()
                    progress_win.destroy()
                except tk.TclError:
                    pass
            self.update_all_plots()
            if self.compute_gaussian_models_button is not None:
                self.compute_gaussian_models_button.config(state=tk.DISABLED)
            if len(self.unit_labels) > 0:
                self.resort_unit_labels_by_position()
            messagebox.showinfo("Complete", f"Gaussian model computation complete. {len(self.gaussian_models)} models accepted.")

        def _process_accept_reject(accepted, accept_data, popup, update_content_fn, current_seed_index_ref):
            try:
                self._disable_user_buttons()
                if popup.winfo_exists():
                    self._set_buttons_in_frame_state(popup, tk.DISABLED)
                if accepted:
                    spike_ids = np.asarray(accept_data['spike_ids'])
                    if len(spike_ids) > 0:
                        self.gaussian_models.append(accept_data['model'])
                        new_unit_label = (max(self.unit_labels) + 1) if self.unit_labels else 0
                        if self.all_properties is not None and 'y_pos' in self.prop_titles:
                            y_pos_idx = self.prop_titles.index('y_pos')
                            mean_y_pos = float(np.mean(self.all_properties[spike_ids, y_pos_idx]))
                        else:
                            mean_y_pos = 0.0
                        m = accept_data['model']
                        unit_vars = [{
                            'gaussian': True,
                            'feature_indices': list(accept_data['feature_indices']),
                            'center': np.array(m.mean, dtype=float).copy(),
                            'covariance': np.array(m.covariance, dtype=float).copy(),
                            'mah_th': float(m.mah_threshold),
                            'bic_th': float(m.bic_threshold),
                        }]
                        if m.data_range is not None:
                            unit_vars[0]['data_range'] = m.data_range
                        if m.sort_range is not None:
                            unit_vars[0]['sort_range'] = m.sort_range
                        if accept_data.get('sort_feature_idx_global') is not None:
                            unit_vars[0]['sort_feature_idx'] = accept_data['sort_feature_idx_global']
                        unit = Unit(unit_variables=unit_vars, mean_y_pos=mean_y_pos)
                        self.unit_info[new_unit_label] = unit
                        self.unit_labels.append(new_unit_label)
                        self.unit_labels = sorted(self.unit_labels, key=lambda l: self.unit_info[l].mean_y_pos)
                        self.unit_id[spike_ids] = new_unit_label
                        if self.curr_max_prob is not None and 'curr_max_prob_update' in accept_data:
                            self.curr_max_prob[spike_ids] = accept_data['curr_max_prob_update']
                        self._remove_empty_units()
                        self.unit_combo['values'] = ["new_unit"] + [f"Unit {l}" for l in self.unit_labels]
                        self._sync_compare_combo_values()
                        if len(self.unit_labels) > 0:
                            self.plot_sorted_check.config(state=tk.NORMAL)
                    m = accept_data['model']
                    mah_th = float(m.mah_threshold)
                    mean = m.mean
                    inv_cov = np.linalg.pinv(m.covariance)
                    to_remove = set()
                    for s in self.seeds:
                        diff = properties_filtered[s] - mean
                        mahal_sq = np.dot(diff, np.dot(inv_cov, diff))
                        if np.sqrt(mahal_sq) <= mah_th:
                            to_remove.add(s)
                    self.seeds = [s for s in self.seeds if s not in to_remove]
                else:
                    if accept_data is not None and 'rejected_spike_indices' in accept_data:
                        to_remove = set(accept_data['rejected_spike_indices'])
                        self.seeds = [s for s in self.seeds if s not in to_remove]
                    else:
                        seed_orig = seeds_by_density[current_seed_index_ref[0]] if current_seed_index_ref[0] < len(seeds_by_density) else None
                        if seed_orig is not None:
                            self.seeds = [s for s in self.seeds if s != seed_orig]
                self.update_all_plots()
            finally:
                self._enable_user_buttons()
            # Next seed: find next in seeds_by_density that is still in self.seeds
            next_idx = current_seed_index_ref[0] + 1
            while next_idx < n_seeds and (seeds_by_density[next_idx] if next_idx < len(seeds_by_density) else None) not in self.seeds:
                next_idx += 1
            while next_idx < n_seeds:
                seed_original = seeds_by_density[next_idx]
                result = fit_gaussian_model_from_seed(
                    properties_filtered, seed_original, cluster_dens, self.curr_max_prob, settings
                )
                if isinstance(result, dict) and result.get('success', False) and 'model' in result and 'visualization_data' in result:
                    gaussian_model = result['model']
                    visualization_data = dict(result['visualization_data'])
                    all_valid = visualization_data.get('all_valid_indices')
                    if all_valid is not None:
                        visualization_data['all_points_full'] = self.all_properties[all_valid].copy()
                    current_seed_index_ref[0] = next_idx
                    update_content_fn(visualization_data, gaussian_model, next_idx, n_seeds, seed_original)
                    if popup.winfo_exists():
                        self._set_buttons_in_frame_state(popup, tk.NORMAL)
                    return
                if isinstance(result, dict) and not result.get('success', False) and settings.get('report_failures', False):
                    hist = result.get('stability_iteration_history', None)
                    if hist is not None and len(hist) > 0:
                        self._show_stability_iteration_debug(hist, result.get('message', 'Model did not reach stability'))
                    else:
                        failure_message = result.get('message', 'Unknown failure')
                        messagebox.showinfo("Seed failed", f"Seed {next_idx + 1}/{n_seeds} failed.\n\n{failure_message}")
                self.seeds = [s for s in self.seeds if s != seed_original]
                next_idx += 1
                while next_idx < n_seeds and (seeds_by_density[next_idx] if next_idx < len(seeds_by_density) else None) not in self.seeds:
                    next_idx += 1
            if popup.winfo_exists():
                popup.destroy()
            _run_completion()

        single_window_opened = False
        for seed_idx, seed_original in enumerate(seeds_by_density):
            try:
                if seed_original not in self.seeds:
                    continue
                if progress_win is not None and progress_win.winfo_exists():
                    progress_bar['value'] = seed_idx
                    progress_label.config(text=f"Remaining seeds: {len(self.seeds)}")
                    progress_win.update_idletasks()
                    progress_win.update()
                    time.sleep(0.1)
                # Close previous failure's visualization when moving to next seed
                if prev_fail_viz_popup is not None or prev_fail_viz_fig is not None:
                    try:
                        if prev_fail_viz_popup is not None and prev_fail_viz_popup.winfo_exists():
                            prev_fail_viz_popup.destroy()
                    except tk.TclError:
                        pass
                    if prev_fail_viz_fig is not None:
                        try:
                            plt.close(prev_fail_viz_fig)
                        except Exception:
                            pass
                    prev_fail_viz_popup, prev_fail_viz_fig = None, None
                
                result = fit_gaussian_model_from_seed(
                    properties_filtered,
                    seed_original,  # This should work if properties_filtered maintains row order
                    cluster_dens,
                    self.curr_max_prob,
                    settings
                )
                
                # Check if successful - handle new format with visualization_data (do not show accept/reject when stability failed)
                if isinstance(result, dict) and result.get('success', False) and 'model' in result and 'visualization_data' in result:
                    gaussian_model = result['model']
                    visualization_data = result['visualization_data']
                    all_valid = visualization_data.get('all_valid_indices')
                    if all_valid is not None:
                        visualization_data = dict(visualization_data)
                        visualization_data['all_points_full'] = self.all_properties[all_valid].copy()
                    feature_pairs = settings.get('viz_feature_pairs', [(0, 1), (0, 2), (3, 4), (0, 5)])
                    if skip_refinement:
                        valid_indices = visualization_data['valid_indices']
                        in_b = visualization_data['in_bounds']
                        bic_th = settings.get('multi_cluster_threshold', 0.2)
                        sort_feat_local = settings.get('sorted_feature_idx', 0)
                        mah_th = float(gaussian_model.mah_threshold)
                        center = gaussian_model.mean
                        covariance = gaussian_model.covariance
                        std = np.sqrt(max(0.0, float(covariance[sort_feat_local, sort_feat_local])))
                        if std <= 0:
                            std = 1.0
                        sort_lo = float(center[sort_feat_local]) - std * mah_th
                        sort_hi = float(center[sort_feat_local]) + std * mah_th
                        data_range = (int(valid_indices[0]), int(valid_indices[-1]))
                        sort_range = (sort_lo, sort_hi)
                        dc = getattr(gaussian_model, 'density_curve', None)
                        accept_data = {
                            'spike_ids': valid_indices[in_b],
                            'model': GaussianModel(mean=center.copy(), covariance=covariance.copy(),
                                                  bic_threshold=bic_th, mah_threshold=mah_th,
                                                  data_range=data_range, sort_range=sort_range, sort_feature_idx=sort_feat_local,
                                                  density_curve=dc),
                            'feature_indices': list(included_feat_idxs),
                            'sort_feature_idx_global': self.sort_feature_idx,
                        }
                        mahal_in_b = visualization_data['mahal_d'][in_b]
                        prob_density = prob_density_from_curve_or_formula(mahal_in_b, mah_th, density_curve=dc)
                        accept_data['curr_max_prob_update'] = np.asarray(prob_density, dtype=float)
                        accepted = True
                    else:
                        settings['sort_feature_idx_global'] = self.sort_feature_idx
                        current_seed_index_ref = [seed_idx]
                        def on_done(accepted, accept_data):
                            _process_accept_reject(accepted, accept_data, popup, update_content_fn, current_seed_index_ref)
                        popup, update_content_fn = self._show_accept_reject_mahal_window(
                            visualization_data, gaussian_model, seed_idx, n_seeds, feature_pairs=feature_pairs, settings=settings,
                            initial_model=result.get('initial_model'), properties_for_refit=properties_filtered,
                            cluster_densities_for_refit=cluster_dens, seed_idx_global=seed_original,
                            on_done_callback=on_done
                        )
                        single_window_opened = True
                        break
                    try:
                        self._disable_user_buttons()
                        if accepted:
                            # Assign all accepted spikes (including reassigning from other units if this model improves their curr_max_prob)
                            spike_ids = np.asarray(accept_data['spike_ids'])
                            if len(spike_ids) > 0:
                                self.gaussian_models.append(accept_data['model'])
                                new_unit_label = (max(self.unit_labels) + 1) if self.unit_labels else 0
                                if self.all_properties is not None and 'y_pos' in self.prop_titles:
                                    y_pos_idx = self.prop_titles.index('y_pos')
                                    mean_y_pos = float(np.mean(self.all_properties[spike_ids, y_pos_idx]))
                                else:
                                    mean_y_pos = 0.0
                                m = accept_data['model']
                                unit_vars = [{
                                    'gaussian': True,
                                    'feature_indices': list(accept_data['feature_indices']),
                                    'center': np.array(m.mean, dtype=float).copy(),
                                    'covariance': np.array(m.covariance, dtype=float).copy(),
                                    'mah_th': float(m.mah_threshold),
                                    'bic_th': float(m.bic_threshold),
                                }]
                                if m.data_range is not None:
                                    unit_vars[0]['data_range'] = m.data_range
                                if m.sort_range is not None:
                                    unit_vars[0]['sort_range'] = m.sort_range
                                if accept_data.get('sort_feature_idx_global') is not None:
                                    unit_vars[0]['sort_feature_idx'] = accept_data['sort_feature_idx_global']
                                unit = Unit(unit_variables=unit_vars, mean_y_pos=mean_y_pos)
                                self.unit_info[new_unit_label] = unit
                                self.unit_labels.append(new_unit_label)
                                self.unit_labels = sorted(self.unit_labels, key=lambda l: self.unit_info[l].mean_y_pos)
                                self.unit_id[spike_ids] = new_unit_label
                                if self.curr_max_prob is not None and 'curr_max_prob_update' in accept_data:
                                    self.curr_max_prob[spike_ids] = accept_data['curr_max_prob_update']
                                self._remove_empty_units()
                                self.unit_combo['values'] = ["new_unit"] + [f"Unit {l}" for l in self.unit_labels]
                                self._sync_compare_combo_values()
                                if len(self.unit_labels) > 0:
                                    self.plot_sorted_check.config(state=tk.NORMAL)
                            # Remove seeds within accepted model boundary (whether or not we created a unit)
                            m = accept_data['model']
                            mah_th = float(m.mah_threshold)
                            mean = m.mean
                            inv_cov = np.linalg.pinv(m.covariance)
                            to_remove = set()
                            for s in self.seeds:
                                diff = properties_filtered[s] - mean
                                mahal_sq = np.dot(diff, np.dot(inv_cov, diff))
                                if np.sqrt(mahal_sq) <= mah_th:
                                    to_remove.add(s)
                            self.seeds = [s for s in self.seeds if s not in to_remove]
                        else:
                            # Rejected: remove all seeds that fall within the rejected model
                            if accept_data is not None and 'rejected_spike_indices' in accept_data:
                                to_remove = set(accept_data['rejected_spike_indices'])
                                self.seeds = [s for s in self.seeds if s not in to_remove]
                            else:
                                self.seeds = [s for s in self.seeds if s != seed_original]
                        self.update_all_plots()
                    finally:
                        self._enable_user_buttons()
                    if not accepted:
                        continue
                    # Let the main window refresh before showing the next seed's window (avoids feeling of immediate close/reopen)
                    self.root.update_idletasks()
                    self.root.update()
                    time.sleep(0.5)
                
                # Handle error case (fit failed): remove this seed
                elif isinstance(result, dict) and not result.get('success', False):
                    if settings.get('report_failures', False):
                        hist = result.get('stability_iteration_history', None)
                        if hist is not None and len(hist) > 0:
                            self._show_stability_iteration_debug(hist, result.get('message', 'Model did not reach stability'))
                        else:
                            failure_message = result.get('message', 'Unknown failure')
                            messagebox.showinfo("Seed failed", f"Seed {seed_idx + 1}/{n_seeds} failed.\n\n{failure_message}")
                    self.seeds = [s for s in self.seeds if s != seed_original]
                    continue
                
            except Exception as e:
                if settings.get('report_failures', False):
                    messagebox.showerror("Error", f"Error processing seed {seed_idx + 1}/{n_seeds}: {str(e)}")
                self.seeds = [s for s in self.seeds if s != seed_original]
                continue
        
        # Close any remaining failure visualization window
        if prev_fail_viz_popup is not None or prev_fail_viz_fig is not None:
            try:
                if prev_fail_viz_popup is not None and prev_fail_viz_popup.winfo_exists():
                    prev_fail_viz_popup.destroy()
            except tk.TclError:
                pass
            if prev_fail_viz_fig is not None:
                try:
                    plt.close(prev_fail_viz_fig)
                except Exception:
                    pass

        if not single_window_opened:
            _run_completion()
    
    def _show_data_views(self, viz_available_h,viz_bracket_w,n_scatter,popup):
        plot_height = min(viz_available_h, 250)
        n_cols = max(1, int(0.9 * viz_bracket_w / plot_height))
        n_rows = math.ceil(n_scatter / n_cols) if n_scatter else 1
        fig_w_px = plot_height
        fig_h_px = plot_height
        dpi = 100
        _rc = {k: plt.rcParams[k] for k in ('font.size', 'xtick.labelsize', 'ytick.labelsize')}
        plt.rcParams.update({'font.size': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8})
        ax_h = fig_h_px*n_rows*1.1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w_px*n_cols / dpi, ax_h / dpi), dpi=dpi)
        plt.rcParams.update(_rc)
        if n_scatter == 1:
            axes = np.array([[axes]])
        elif axes.ndim == 1:
            axes = axes.reshape(-1, n_cols)
        axes_flat = axes.flatten()
        fig.subplots_adjust(left=0.11, right=0.91, top=0.96, bottom=0.08, wspace=0.35, hspace=0.35)
        plot_canvas_size = ax_h + 50
        data_views_container = tk.Frame(popup, height=min(viz_available_h,plot_canvas_size))
        data_views_container.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)
        data_views_container.pack_propagate(False)
        data_views_lf = ttk.LabelFrame(data_views_container, text="Data views", padding=2)
        data_views_lf.pack(fill=tk.BOTH, expand=False)
        view_canvas = tk.Canvas(data_views_lf, width=viz_bracket_w, height=min(viz_available_h,ax_h))
        view_scrollbar = ttk.Scrollbar(data_views_lf, orient="vertical", command=view_canvas.yview)
        view_inner = ttk.Frame(view_canvas)
        view_inner.bind("<Configure>", lambda e: view_canvas.configure(scrollregion=view_canvas.bbox("all")))
        inner_id = view_canvas.create_window((0, 0), window=view_inner, anchor="nw")
        view_canvas.itemconfig(inner_id, width=viz_bracket_w, height=ax_h)
        view_canvas.configure(yscrollcommand=view_scrollbar.set)
        view_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        view_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        def _on_mousewheel(event):
            view_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        view_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas = FigureCanvasTkAgg(fig, master=view_inner)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=False)
        return canvas, axes_flat

    def _set_buttons_in_frame_state(self, parent, state):
        """Recursively set state (tk.NORMAL or tk.DISABLED) on all Button widgets under parent."""
        for w in parent.winfo_children():
            if isinstance(w, (ttk.Button, tk.Button)):
                try:
                    w.config(state=state)
                except tk.TclError:
                    pass
            if w.winfo_children():
                self._set_buttons_in_frame_state(w, state)

    def _disable_user_buttons(self):
        """Disable all user-action buttons in the toolbar so user cannot interact during processing."""
        if getattr(self, 'buttons_frame', None) is not None and self.buttons_frame.winfo_exists():
            self._set_buttons_in_frame_state(self.buttons_frame, tk.DISABLED)
        if getattr(self, 'calc_densities_btn', None) is not None and self.calc_densities_btn.winfo_exists():
            try:
                self.calc_densities_btn.config(state=tk.DISABLED)
            except tk.TclError:
                pass

    def _enable_user_buttons(self):
        """Re-enable toolbar buttons and re-apply conditional states (e.g. compute button when seeds exist)."""
        if getattr(self, 'buttons_frame', None) is not None and self.buttons_frame.winfo_exists():
            self._set_buttons_in_frame_state(self.buttons_frame, tk.NORMAL)
        if getattr(self, 'calc_densities_btn', None) is not None and self.calc_densities_btn.winfo_exists():
            try:
                self.calc_densities_btn.config(state=tk.NORMAL if self.all_properties is not None else tk.DISABLED)
            except tk.TclError:
                pass
        # Re-apply conditional states for compute buttons
        if getattr(self, 'compute_gaussian_models_button', None) is not None and self.compute_gaussian_models_button.winfo_exists():
            try:
                self.compute_gaussian_models_button.config(
                    state=tk.NORMAL if (getattr(self, 'seeds', None) and len(self.seeds) > 0) else tk.DISABLED
                )
            except tk.TclError:
                pass
        if getattr(self, 'compute_gaussian_models_cluster_btn', None) is not None and self.compute_gaussian_models_cluster_btn.winfo_exists():
            try:
                self.compute_gaussian_models_cluster_btn.config(
                    state=tk.NORMAL if (getattr(self, 'cluster_seeds', None) and len(self.cluster_seeds) > 0) else tk.DISABLED
                )
            except tk.TclError:
                pass
        if getattr(self, 'delete_cluster_seed_btn', None) is not None and self.delete_cluster_seed_btn.winfo_exists():
            try:
                idx = self._get_selected_cluster_seed_index()
                self.delete_cluster_seed_btn.config(state=tk.NORMAL if idx is not None else tk.DISABLED)
            except (tk.TclError, Exception):
                pass

    def _show_accept_reject_mahal_window(self, visualization_data, gaussian_model, seed_idx, n_seeds, feature_pairs=None, settings=None,
                                         initial_model=None, properties_for_refit=None, cluster_densities_for_refit=None, seed_idx_global=None,
                                         on_done_callback=None):
        """
        Single window: N-panel Mahal plot (in-cluster=red, out=orange->black) + Size, BIC threshold,
        Re-stabilize, Re-fit model, Accept, Reject. feature_pairs: list of (global_prop_idx_0, global_prop_idx_1).
        initial_model: optional model from _create_init_cluster_from_seed (for Re-stabilize).
        properties_for_refit / cluster_densities_for_refit / seed_idx_global: used for Re-stabilize and Re-fit model.
        on_done_callback: optional callable(accepted, accept_data). If set, Accept/Reject call it and do not destroy
            the window; returns (popup, update_content_fn) and no wait_window. update_content_fn(viz, model, seed_idx, n_seeds) updates the window.
        Returns (accepted: bool, accept_data: dict or None) in modal mode, or (popup, update_content_fn) in reusable mode.
        """
        if self.included_feature_indexes is None or len(self.included_feature_indexes) < 2:
            messagebox.showwarning("Warning", "Cannot show window: need at least 2 included features")
            return (False, None) if on_done_callback is None else (None, None)
        reusable = on_done_callback is not None
        ref_viz = {'v': visualization_data}
        points = visualization_data['points']
        all_points_full = visualization_data.get('all_points_full')
        seed_point = visualization_data.get('seed_point')
        included_feat_idxs = list(self.included_feature_indexes)
        if feature_pairs is None or len(feature_pairs) == 0:
            feature_pairs = [(included_feat_idxs[0], included_feat_idxs[1])]
        feature_pairs = list(feature_pairs)
        n_plots = len(feature_pairs)
        bic_th_default = visualization_data.get('multi_cluster_threshold', 0.2)
        min_points_for_cluster = int(settings.get('min_points_for_cluster', 100)) if settings else 100
        unit_id = np.array(settings.get('unit_id'), dtype=int) if settings and settings.get('unit_id') is not None else None
        n_spikes = self.all_properties.shape[0]
        if not settings or settings.get('curr_max_prob') is None:
            raise ValueError("curr_max_prob is required in settings (same length as spikes, valid values or inf)")
        curr_max_prob = np.asarray(settings['curr_max_prob'], dtype=float)
        if len(curr_max_prob) != n_spikes:
            raise ValueError(f"curr_max_prob length ({len(curr_max_prob)}) must match number of spikes ({n_spikes})")
        if np.any(np.isnan(curr_max_prob)) or np.any((curr_max_prob < 0) & np.isfinite(curr_max_prob)):
            raise ValueError("curr_max_prob must contain only valid (non-negative finite) values or inf")
        _SU_COLORS = ['yellow', 'blue', 'green', 'purple', 'brown', 'cyan']
        def su_color_for_unit(ul):
            if ul not in self.unit_labels:
                return 'gray'
            return _SU_COLORS[self.unit_labels.index(ul) % len(_SU_COLORS)]
        # Mutable state: use passed-in model and viz as-is (no recomputation until user clicks Recompute)
        HC = np.asarray(visualization_data['HC'], dtype=float) if 'HC' in visualization_data else None
        in_bounds_raw = np.asarray(visualization_data['in_bounds'], dtype=float)
        if in_bounds_raw.size == 0 or np.any(np.isnan(in_bounds_raw)):
            in_bounds_init = np.array([], dtype=int)
        else:
            in_bounds_init = np.asarray(in_bounds_raw, dtype=int)
            # state['in_bounds'] are indices into the data window (valid_indices); clamp to valid range
            n_data_window = len(visualization_data['valid_indices'])
            in_bounds_init = in_bounds_init[(in_bounds_init >= 0) & (in_bounds_init < n_data_window)]
        init_center = np.array(gaussian_model.mean, dtype=float).copy()
        init_cov = np.array(gaussian_model.covariance, dtype=float).copy()
        init_mahal_d = np.array(visualization_data['mahal_d'], dtype=float)
        density_curve = getattr(gaussian_model, 'density_curve', None)
        density_curve = np.asarray(density_curve, dtype=float) if density_curve is not None else None
        state = {
            'center': init_center,
            'covariance': init_cov,
            'mahal_d': init_mahal_d,
            'mah_th': float(gaussian_model.mah_threshold),
            'bic_th': float(bic_th_default),
            'density_curve': density_curve,
            'in_bounds': in_bounds_init,
            '_data_range': getattr(gaussian_model, 'data_range', None),
            '_sort_range': getattr(gaussian_model, 'sort_range', None),
            '_sort_feature_idx': getattr(gaussian_model, 'sort_feature_idx', None),
        }
        ref_state = {'s': state}
        ref_seed_global = {'g': seed_idx_global}

        def _state_from_viz_model(viz, model):
            """Build state dict from visualization_data and gaussian_model (for update_content)."""
            in_b_raw = np.asarray(viz['in_bounds'], dtype=float)
            if in_b_raw.size == 0 or np.any(np.isnan(in_b_raw)):
                in_b = np.array([], dtype=int)
            else:
                in_b = np.asarray(in_b_raw, dtype=int)
                n_dw = len(viz['valid_indices'])
                in_b = in_b[(in_b >= 0) & (in_b < n_dw)]
            dc = getattr(model, 'density_curve', None)
            dc = np.asarray(dc, dtype=float) if dc is not None else None
            return {
                'center': np.array(model.mean, dtype=float).copy(),
                'covariance': np.array(model.covariance, dtype=float).copy(),
                'mahal_d': np.array(viz['mahal_d'], dtype=float).copy(),
                'mah_th': float(model.mah_threshold),
                'bic_th': float(getattr(model, 'bic_threshold', bic_th_default)),
                'density_curve': dc,
                'in_bounds': in_b,
                '_data_range': getattr(model, 'data_range', None),
                '_sort_range': getattr(model, 'sort_range', None),
                '_sort_feature_idx': getattr(model, 'sort_feature_idx', None),
            }

        result = {'accepted': False, 'accept_data': None}
        popup = tk.Toplevel(self.root)
        popup.title(f"Seed {seed_idx + 1}/{n_seeds} - Accept/Reject Gaussian model")
        try:
            popup.update_idletasks()
            win_h = int(popup.winfo_screenheight() * 0.9)
            win_w = win_h
            popup.geometry(f"{win_w}x{win_h}+0+0")
        except tk.TclError:
            win_h, win_w = 700, 700
            popup.geometry(f"{win_w}x{win_h}+0+0")

        popup.transient(self.root)
        popup.grab_set()
        canvas_widget = None
        
        def re_stabilize():
            """Run iterate_GM_model with current displayed model (and BIC) and user-entered size; if stable, update displayed model."""
            seed_global = ref_seed_global['g']
            if properties_for_refit is None or cluster_densities_for_refit is None or seed_global is None:
                messagebox.showwarning("Re-stabilize", "Re-stabilize is not available (e.g. when editing an existing unit).")
                return
            st = ref_state['s']
            viz = ref_viz['v']
            try:
                self._disable_user_buttons()
                self._set_buttons_in_frame_state(popup, tk.DISABLED)
                try:
                    new_size = float(mah_th_var.get())
                except (ValueError, tk.TclError):
                    messagebox.showwarning("Re-stabilize", "Invalid size value.")
                    return
                sort_idx = st.get('_sort_feature_idx')
                if sort_idx is None:
                    sort_idx = 0
                model_copy = GaussianModel(
                    mean=st['center'].copy(),
                    covariance=st['covariance'].copy(),
                    mah_threshold=new_size,
                    bic_threshold=st['bic_th'],
                    density_curve=st.get('density_curve'),
                    data_range=st.get('_data_range'),
                    sort_range=st.get('_sort_range'),
                    sort_feature_idx=sort_idx,
                )
                if model_copy.data_range is None:
                    model_copy.compute_data_range(properties_for_refit)
                gm = self.settings.get('gm') or {}
                init_stds = gm.get('initial_stds')
                if init_stds is None or len(np.atleast_1d(init_stds)) != st['covariance'].shape[0]:
                    init_stds = np.sqrt(np.maximum(np.diag(st['covariance']), 1e-12))
                else:
                    init_stds = np.asarray(init_stds, dtype=float).ravel()[: st['covariance'].shape[0]]
                sort_feature_idx = getattr(model_copy, 'sort_feature_idx', 0)
                max_range_sorted = float(settings.get('max_range_sorted', gm.get('max_range_sorted', init_stds[sort_feature_idx] * 4.0)))
                min_change = settings.get('min_change', 0.01)
                try:
                    outcome, model_out, assignment_out, explode_reason = iterate_GM_model(
                        properties_for_refit, cluster_densities_for_refit, model_copy, sort_feature_idx, max_range_sorted,
                        seed_global, min_change=min_change, min_points_for_cluster=min_points_for_cluster,
                        curr_max_prob=curr_max_prob
                    )
                except Exception as e:
                    messagebox.showerror("Re-stabilize", str(e))
                    return
                if outcome == 'stable':
                    first_idx, last_idx = model_out.data_range
                    valid_indices_new = np.arange(first_idx, last_idx + 1)
                    points_new = properties_for_refit[valid_indices_new]
                    inv_cov = np.linalg.pinv(model_out.covariance)
                    diff_new = points_new - model_out.mean
                    mahal_d_new = np.einsum('ij,jk,ik->i', diff_new, inv_cov, diff_new)**0.5
                    sort_lo, sort_hi = model_out.sort_range
                    width = sort_hi - sort_lo
                    half_extra = 0.25 * width
                    sorted_col = np.asarray(properties_for_refit[:, model_out.sort_feature_idx], dtype=float)
                    viz_first = np.searchsorted(sorted_col, sort_lo - half_extra, side='left')
                    viz_last = np.searchsorted(sorted_col, sort_hi + half_extra, side='right') - 1
                    viz_last = min(max(viz_last, viz_first), len(properties_for_refit) - 1)
                    valid_indices_all_new = np.arange(viz_first, viz_last + 1)
                    points_all_new = properties_for_refit[valid_indices_all_new]
                    viz['points'] = points_new
                    viz['valid_indices'] = valid_indices_new
                    viz['mahal_d'] = mahal_d_new
                    viz['in_bounds'] = np.asarray(assignment_out, dtype=int)
                    viz['all_points'] = points_all_new
                    viz['all_valid_indices'] = valid_indices_all_new
                    if hasattr(self, 'all_properties') and self.all_properties is not None:
                        viz['all_points_full'] = self.all_properties[valid_indices_all_new].copy()
                    st['center'] = np.array(model_out.mean, dtype=float).copy()
                    st['covariance'] = np.array(model_out.covariance, dtype=float).copy()
                    st['mah_th'] = float(model_out.mah_threshold)
                    st['density_curve'] = getattr(model_out, 'density_curve', None)
                    st['density_curve'] = np.asarray(st['density_curve'], dtype=float) if st['density_curve'] is not None else None
                    st['mahal_d'] = mahal_d_new.copy()
                    st['in_bounds'] = np.asarray(assignment_out, dtype=int)
                    st['_data_range'] = model_out.data_range
                    st['_sort_range'] = getattr(model_out, 'sort_range', None)
                    st['_sort_feature_idx'] = getattr(model_out, 'sort_feature_idx', None)
                    mah_th_var.set(round(float(st['mah_th']), 2))
                    redraw()
                else:
                    msg = f"Outcome was '{outcome}'."
                    if outcome == 'exploded' and explode_reason:
                        msg += " " + explode_reason
                    msg += " Model unchanged."
                    messagebox.showwarning("Re-stabilize", msg)
            finally:
                self._enable_user_buttons()
                try:
                    self._set_buttons_in_frame_state(popup, tk.NORMAL)
                except tk.TclError:
                    pass

        def re_fit_model():
            """Run _create_init_cluster_from_seed with new BIC threshold; if success, replace displayed model."""
            seed_global = ref_seed_global['g']
            if properties_for_refit is None or cluster_densities_for_refit is None or seed_global is None:
                messagebox.showwarning("Re-fit model", "Re-fit is not available (e.g. when editing an existing unit).")
                return
            st = ref_state['s']
            viz = ref_viz['v']
            try:
                new_bic = float(bic_th_var.get())
            except (ValueError, tk.TclError):
                messagebox.showwarning("Re-fit model", "Invalid BIC threshold value.")
                return
            try:
                self._disable_user_buttons()
                self._set_buttons_in_frame_state(popup, tk.DISABLED)
                gm = dict(self.settings.get('gm') or {})
                n_f = properties_for_refit.shape[1]
                sort_local = gm.get('sorted_feature_idx', 0)
                if sort_local < 0 or sort_local >= n_f:
                    sort_local = 0
                init_stds = gm.get('initial_stds')
                if init_stds is None or len(np.atleast_1d(init_stds)) != n_f:
                    messagebox.showwarning("Re-fit model", "initial_stds must be set in settings and match number of features.")
                    return
                init_stds = np.asarray(init_stds, dtype=float).ravel()[:n_f]
                if len(init_stds) < n_f:
                    messagebox.showwarning("Re-fit model", "initial_stds must have one value per feature.")
                    return
                new_settings = {
                    'sorted_feature_idx': int(sort_local),
                    'min_points_for_cluster': int(gm.get('min_points_for_cluster', 100)),
                    'initial_stds': np.asarray(init_stds, dtype=float),
                    'max_range_sorted': float(gm.get('max_range_sorted', gm.get(f'max_range_{self.prop_titles[self.sort_feature_idx] if getattr(self, "sort_feature_idx", None) is not None and self.sort_feature_idx < len(self.prop_titles) else "sorted feature"}', init_stds[sort_local] * 4.0))),
                    'n_samples_density_curve': int(gm.get('n_samples_density_curve', 101)),
                    'multi_cluster_threshold': new_bic,
                }
                for k in ['init_mah_th', 'com_iteration_threshold', 'com_iteration_max_iterations', 'density_threshold_for_init_distance',
                          'gaussian_filter_sigma', 'dist_step', 'min_change']:
                    if k in gm:
                        new_settings[k] = gm[k]
                init_result = _create_init_cluster_from_seed(
                    properties_for_refit, seed_global, cluster_densities_for_refit, curr_max_prob, new_settings
                )
                if not init_result['success']:
                    messagebox.showerror("Re-fit model", init_result.get('message', 'Fit failed'))
                    return
                model = init_result['model']
                in_bounds_new = init_result['in_bounds']
                # Run size iteration so displayed model is stabilized (can grow with new BIC), not just the initial boundary
                model, in_bounds_new, _hist, _hist_full, _stability_failed, _iter_count = _fit_model_size(
                    properties_for_refit, cluster_densities_for_refit, model, in_bounds_new,
                    seed_global, curr_max_prob, new_settings
                )
                # One iterate_GM_model run with final model (same as Re-stabilize) so assignment_out
                # and data_range come from the same run — then use model_out for all viz state
                sort_feature_idx = getattr(model, 'sort_feature_idx', 0)
                max_range_sorted = float(new_settings.get('max_range_sorted', init_stds[sort_feature_idx] * 4.0))
                min_points_for_cluster = int(new_settings.get('min_points_for_cluster', 100))
                try:
                    _outcome, _model_out, assignment_out, _ = iterate_GM_model(
                        properties_for_refit, cluster_densities_for_refit, model.copy(), sort_feature_idx, max_range_sorted,
                        seed_global, min_change=new_settings.get('min_change', 0.01),
                        min_points_for_cluster=min_points_for_cluster, curr_max_prob=curr_max_prob
                    )
                    if _outcome == 'stable':
                        model = _model_out
                except Exception:
                    pass
                first_idx, last_idx = model.data_range
                valid_indices_new = np.arange(first_idx, last_idx + 1)
                # Red dots from model's in_bounds rule so they always match the displayed model
                in_bounds_global = model.in_bounds_indices(properties_for_refit, curr_max_prob)
                in_bounds_for_viz = (in_bounds_global - first_idx).astype(int)
                in_bounds_for_viz = in_bounds_for_viz[(in_bounds_for_viz >= 0) & (in_bounds_for_viz < len(valid_indices_new))]
                points_new = properties_for_refit[valid_indices_new]
                inv_cov = np.linalg.pinv(model.covariance)
                diff_new = points_new - model.mean
                mahal_d_new = np.einsum('ij,jk,ik->i', diff_new, inv_cov, diff_new)**0.5
                sort_lo, sort_hi = model.sort_range
                width = sort_hi - sort_lo
                half_extra = 0.25 * width
                sorted_col = np.asarray(properties_for_refit[:, model.sort_feature_idx], dtype=float)
                viz_first = np.searchsorted(sorted_col, sort_lo - half_extra, side='left')
                viz_last = np.searchsorted(sorted_col, sort_hi + half_extra, side='right') - 1
                viz_last = min(max(viz_last, viz_first), len(properties_for_refit) - 1)
                valid_indices_all_new = np.arange(viz_first, viz_last + 1)
                points_all_new = properties_for_refit[valid_indices_all_new]
                viz['points'] = points_new
                viz['valid_indices'] = valid_indices_new
                viz['mahal_d'] = mahal_d_new
                viz['in_bounds'] = in_bounds_for_viz
                viz['all_points'] = points_all_new
                viz['all_valid_indices'] = valid_indices_all_new
                if hasattr(self, 'all_properties') and self.all_properties is not None:
                    viz['all_points_full'] = self.all_properties[valid_indices_all_new].copy()
                st['center'] = np.array(model.mean, dtype=float).copy()
                st['covariance'] = np.array(model.covariance, dtype=float).copy()
                st['mah_th'] = float(model.mah_threshold)
                st['bic_th'] = float(model.bic_threshold)
                st['density_curve'] = getattr(model, 'density_curve', None)
                st['density_curve'] = np.asarray(st['density_curve'], dtype=float) if st['density_curve'] is not None else None
                st['mahal_d'] = mahal_d_new.copy()
                st['in_bounds'] = in_bounds_for_viz.copy()
                st['_data_range'] = model.data_range
                st['_sort_range'] = getattr(model, 'sort_range', None)
                st['_sort_feature_idx'] = getattr(model, 'sort_feature_idx', None)
                mah_th_var.set(round(float(st['mah_th']), 2))
                bic_th_var.set(st['bic_th'])
                redraw()
            finally:
                self._enable_user_buttons()
                try:
                    self._set_buttons_in_frame_state(popup, tk.NORMAL)
                except tk.TclError:
                    pass

        def redraw():
            viz = ref_viz['v']
            st = ref_state['s']
            points = viz['points']
            valid_indices = viz['valid_indices']
            all_points = viz.get('all_points')
            all_valid_indices = viz.get('all_valid_indices')
            mah_th = st['mah_th']
            in_bounds = st['in_bounds']
            center = st['center']
            covariance = st['covariance']
            mahal_d = st['mahal_d']
            # Use full-window points (assigned + unassigned) when available so sorted spikes show in unit colors
            if all_points is not None and all_valid_indices is not None and all_points.shape[0] == len(all_valid_indices):
                plot_points = all_points
                plot_valid_indices = all_valid_indices
                try: 
                    inv_cov = np.linalg.pinv(covariance)
                except:
                    inv_cov = np.ones_like(covariance)
                    print("Warning: SVD did not converge while computing pseudoinverse.")
                diff_all = all_points - center
                mahal_d_plot = np.einsum('ij,jk,ik->i', diff_all, inv_cov, diff_all) ** 0.5
                # st['in_bounds'] are indices into the data window (valid_indices); map to plot window
                in_bounds_global = valid_indices[st['in_bounds']]
                in_bounds_mask = np.isin(all_valid_indices, in_bounds_global)
                if unit_id is not None and len(unit_id) > 0 and np.max(all_valid_indices) < len(unit_id):
                    point_unit_ids = unit_id[all_valid_indices]
                else:
                    point_unit_ids = np.full(len(all_points), -1, dtype=int)
            else:
                plot_points = points
                plot_valid_indices = valid_indices
                mahal_d_plot = mahal_d
                in_bounds_mask = np.zeros(len(points), dtype=bool)
                in_bounds_mask[in_bounds] = True
                if unit_id is not None and len(unit_id) > 0 and (len(valid_indices) == 0 or np.max(valid_indices) < len(unit_id)):
                    point_unit_ids = unit_id[valid_indices]
                else:
                    point_unit_ids = np.full(len(points), -1, dtype=int)
            unassigned = (point_unit_ids < 0)
            unassigned_out_ind = np.where(unassigned & ~in_bounds_mask)[0]
            unassigned_in_ind = np.where(unassigned & in_bounds_mask)[0]
            in_bounds_ind = np.where(in_bounds_mask)[0]
            try:
                dot_sz = max(0.5, float(dot_size_var.get()))
            except (ValueError, tk.TclError, NameError):
                dot_sz = 2.0
            try:
                ss = max(1, min(10, int(ss_var.get())))
            except (ValueError, tk.TclError, NameError):
                ss = 2
            n_pts = len(plot_points)
            sub_idx = np.arange(0, n_pts, ss) if (ss > 1 and n_pts > 0) else np.arange(n_pts)
            if ss > 1 and n_pts > 0:
                unassigned_out_ind = unassigned_out_ind[np.isin(unassigned_out_ind, sub_idx)]
                unassigned_in_ind = unassigned_in_ind[np.isin(unassigned_in_ind, sub_idx)]
                in_bounds_ind = in_bounds_ind[np.isin(in_bounds_ind, sub_idx)]
                sort_out = np.argsort(mahal_d_plot[unassigned_out_ind])[::-1] if len(unassigned_out_ind) > 0 else np.array([], dtype=int)
            # Full-property data available for panels that use a non-included feature
            use_full = (all_points_full is not None and all_points_full.shape[0] == len(plot_valid_indices)
                        and (not feature_pairs or all_points_full.shape[1] > max(max(p0, p1) for p0, p1 in feature_pairs)))
            # Step 1: Compute ranges for all plots (center/range per plot; cap to full data extent)
            plot_ranges = []
            for plot_idx, (prop_idx_0, prop_idx_1) in enumerate(feature_pairs):
                both_included = (prop_idx_0 in included_feat_idxs and prop_idx_1 in included_feat_idxs)
                if both_included:
                    local_i0 = included_feat_idxs.index(prop_idx_0)
                    local_i1 = included_feat_idxs.index(prop_idx_1)
                    col0, col1 = local_i0, local_i1
                    data_2d = plot_points
                else:
                    # Non-included feature plot: need full-property data so this graph can refresh
                    data_2d = None
                    col0, col1 = prop_idx_0, prop_idx_1
                    if all_points_full is not None and all_points_full.shape[0] == len(plot_valid_indices):
                        nfc = all_points_full.shape[1]
                        if nfc >= 2:
                            data_2d = all_points_full
                            col0 = min(prop_idx_0, nfc - 1)
                            col1 = min(prop_idx_1, nfc - 1)
                            if col0 == col1:
                                col1 = max(0, col0 - 1)
                    if data_2d is None and hasattr(self, 'all_properties') and self.all_properties is not None and len(plot_valid_indices) > 0:
                        pvi = np.asarray(plot_valid_indices, dtype=int)
                        if pvi.max() < self.all_properties.shape[0] and self.all_properties.shape[1] >= 2:
                            data_2d = np.asarray(self.all_properties[pvi, :], dtype=float)
                            nfc = data_2d.shape[1]
                            col0 = min(prop_idx_0, nfc - 1)
                            col1 = min(prop_idx_1, nfc - 1)
                            if col0 == col1:
                                col1 = max(0, col0 - 1)
                    if data_2d is None:
                        col0, col1 = -1, -1
                if data_2d is None or col0 < 0 or col1 < 0 or col0 >= data_2d.shape[1] or col1 >= data_2d.shape[1]:
                    plot_ranges.append(None)
                    continue
                data_min_x = float(np.min(data_2d[:, col0]))
                data_max_x = float(np.max(data_2d[:, col0]))
                data_min_y = float(np.min(data_2d[:, col1]))
                data_max_y = float(np.max(data_2d[:, col1]))
                li0, li1 = None, None
                if both_included:
                    li0, li1 = local_i0, local_i1
                    center_2d = np.array([center[local_i0], center[local_i1]])
                    cov_2d = np.array([
                        [covariance[local_i0, local_i0], covariance[local_i0, local_i1]],
                        [covariance[local_i1, local_i0], covariance[local_i1, local_i1]]
                    ])
                    sx = np.sqrt(max(float(cov_2d[0, 0]), 1e-12))
                    sy = np.sqrt(max(float(cov_2d[1, 1]), 1e-12))
                    half_x = 2.0 * mah_th * sx
                    half_y = 2.0 * mah_th * sy
                    axis_half_x = 1.5 * half_x
                    axis_half_y = 1.5 * half_y
                    xmin = center_2d[0] - axis_half_x
                    xmax = center_2d[0] + axis_half_x
                    ymin = center_2d[1] - axis_half_y
                    ymax = center_2d[1] + axis_half_y
                else:
                    # data_2d (all_points_full) has same row order as plot_points; use plot indices for in-bounds
                    in_b = np.where(in_bounds_mask)[0]
                    if len(in_b) > 0:
                        inc_x = data_2d[in_b, col0]
                        inc_y = data_2d[in_b, col1]
                        cx = float(np.mean(inc_x))
                        cy = float(np.mean(inc_y))
                        span_x = float(np.ptp(inc_x))
                        span_y = float(np.ptp(inc_y))
                        if span_x <= 0:
                            span_x = max(abs(cx) * 0.1, 1e-6)
                        if span_y <= 0:
                            span_y = max(abs(cy) * 0.1, 1e-6)
                        half_x = span_x * 0.75
                        half_y = span_y * 0.75
                        xmin = cx - half_x
                        xmax = cx + half_x
                        ymin = cy - half_y
                        ymax = cy + half_y
                    else:
                        xmin, xmax = data_min_x, data_max_x
                        ymin, ymax = data_min_y, data_max_y
                xmin = max(xmin, data_min_x)
                xmax = min(xmax, data_max_x)
                ymin = max(ymin, data_min_y)
                ymax = min(ymax, data_max_y)
                if xmax <= xmin:
                    xmin, xmax = data_min_x, data_max_x
                if ymax <= ymin:
                    ymin, ymax = data_min_y, data_max_y
                plot_ranges.append((data_2d, col0, col1, both_included, li0, li1, xmin, xmax, ymin, ymax))
            # Sync ranges per feature: each feature uses the max range across all plots that show it
            feat_range_min = {}
            feat_range_max = {}
            for plot_idx, (prop_idx_0, prop_idx_1) in enumerate(feature_pairs):
                if plot_idx >= len(plot_ranges) or plot_ranges[plot_idx] is None:
                    continue
                pr = plot_ranges[plot_idx]
                xmin, xmax, ymin, ymax = pr[6], pr[7], pr[8], pr[9]
                for prop_idx, lo, hi in [(prop_idx_0, xmin, xmax), (prop_idx_1, ymin, ymax)]:
                    feat_range_min[prop_idx] = min(feat_range_min.get(prop_idx, lo), lo)
                    feat_range_max[prop_idx] = max(feat_range_max.get(prop_idx, hi), hi)
            for plot_idx, (prop_idx_0, prop_idx_1) in enumerate(feature_pairs):
                if plot_idx >= len(plot_ranges) or plot_ranges[plot_idx] is None:
                    continue
                pr = list(plot_ranges[plot_idx])
                pr[6] = feat_range_min.get(prop_idx_0, pr[6])
                pr[7] = feat_range_max.get(prop_idx_0, pr[7])
                pr[8] = feat_range_min.get(prop_idx_1, pr[8])
                pr[9] = feat_range_max.get(prop_idx_1, pr[9])
                plot_ranges[plot_idx] = tuple(pr)
            # Step 2: Select point indices that are in range for all plots
            in_range_mask = np.ones(n_pts, dtype=bool)
            for pr in plot_ranges:
                if pr is None:
                    continue
                data_2d, col0, col1 = pr[0], pr[1], pr[2]
                xmin, xmax, ymin, ymax = pr[6], pr[7], pr[8], pr[9]
                in_range_mask &= (data_2d[:, col0] >= xmin) & (data_2d[:, col0] <= xmax) & (data_2d[:, col1] >= ymin) & (data_2d[:, col1] <= ymax)
            # Step 3: Plot only in-range data and set axis limits
            for plot_idx, (prop_idx_0, prop_idx_1) in enumerate(feature_pairs):
                if plot_idx >= len(axes_flat) or plot_idx >= len(plot_ranges) or plot_ranges[plot_idx] is None:
                    continue
                pr = plot_ranges[plot_idx]
                data_2d, col0, col1, both_included = pr[0], pr[1], pr[2], pr[3]
                local_i0, local_i1 = pr[4], pr[5]
                xmin, xmax, ymin, ymax = pr[6], pr[7], pr[8], pr[9]
                ax = axes_flat[plot_idx]
                ax.clear()
                out_in_range = unassigned_out_ind[in_range_mask[unassigned_out_ind]]
                in_bounds_in_range = in_bounds_ind[in_range_mask[in_bounds_ind]]
                if ss > 1 and n_pts > 0:
                    out_in_range = out_in_range[np.isin(out_in_range, sub_idx)]
                    in_bounds_in_range = in_bounds_in_range[np.isin(in_bounds_in_range, sub_idx)]
                sort_out_r = np.argsort(mahal_d_plot[out_in_range])[::-1] if len(out_in_range) > 0 else np.array([], dtype=int)
                if len(out_in_range) > 0:
                    out_order = out_in_range[sort_out_r]
                    x_out = data_2d[out_order, col0]
                    y_out = data_2d[out_order, col1]
                    d_out = mahal_d_plot[out_order]
                    th2 = 2.0 * mah_th
                    t = np.clip((d_out - mah_th) / (th2 - mah_th) if th2 > mah_th else 0, 0, 1)
                    colors_out = np.zeros((len(d_out), 3))
                    colors_out[:, 0] = 1.0 - t
                    colors_out[:, 1] = 0.5 * (1.0 - t)
                    colors_out[:, 2] = 0.0
                    alpha_out = 0.1 + 0.7 * (1.0 - t)
                    ax.scatter(x_out, y_out, c=colors_out, s=dot_sz, alpha=alpha_out, zorder=1)
                if len(in_bounds_in_range) > 0:
                    x_in = data_2d[in_bounds_in_range, col0]
                    y_in = data_2d[in_bounds_in_range, col1]
                    ax.scatter(x_in, y_in, c='red', s=dot_sz, alpha=0.8, zorder=2)
                for ul in self.unit_labels:
                    ass = (point_unit_ids == ul) & in_range_mask & ~in_bounds_mask
                    if ss > 1 and n_pts > 0:
                        ass = ass & np.isin(np.arange(n_pts), sub_idx)
                    if not np.any(ass):
                        continue
                    ind = np.where(ass)[0]
                    x_a = data_2d[ind, col0]
                    y_a = data_2d[ind, col1]
                    ax.scatter(x_a, y_a, c=su_color_for_unit(ul), s=dot_sz, alpha=0.8, zorder=3)
                if both_included and local_i0 is not None and local_i1 is not None:
                    center_2d = np.array([center[local_i0], center[local_i1]])
                    cov_2d = np.array([
                        [covariance[local_i0, local_i0], covariance[local_i0, local_i1]],
                        [covariance[local_i1, local_i0], covariance[local_i1, local_i1]]
                    ])
                    ax.scatter(center_2d[0], center_2d[1], s=100, c='red', edgecolors='black', linewidths=2, zorder=5)
                    if seed_point is not None and local_i0 < len(seed_point) and local_i1 < len(seed_point):
                        ax.scatter(seed_point[local_i0], seed_point[local_i1], s=200, c='orange', edgecolors='black', linewidths=2, zorder=6)
                    self._plot_mahalanobis_ellipse(ax, center_2d, cov_2d, mah_th, linestyle='--', color='red', linewidth=1.5)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                feat_name_0 = self.prop_titles[prop_idx_0] if prop_idx_0 < len(self.prop_titles) else f"F{prop_idx_0}"
                feat_name_1 = self.prop_titles[prop_idx_1] if prop_idx_1 < len(self.prop_titles) else f"F{prop_idx_1}"
                ax.set_xlabel(feat_name_0)
                ax.set_ylabel(feat_name_1)
                ax.grid(True, alpha=0.3)
            for idx in range(n_plots, len(axes_flat)):
                axes_flat[idx].set_visible(False)
            if canvas_widget is not None:
                canvas_widget.draw()
        
        # Controls row
        ctrl = ttk.Frame(popup)
        ctrl.pack(fill=tk.X, padx=10, pady=5)
        dot_size_var = tk.DoubleVar(value=2.0)
        ss_var = tk.IntVar(value=1)
        ttk.Label(ctrl, text="Dot size:").pack(side=tk.LEFT, padx=(0, 2))
        dot_scale = tk.Scale(ctrl, from_=0.5, to=20, resolution=0.5, orient=tk.HORIZONTAL,
                             variable=dot_size_var, length=80)
        dot_scale.pack(side=tk.LEFT, padx=2)
        dot_scale.bind('<ButtonRelease-1>', lambda e: redraw())
        ttk.Label(ctrl, text="SS:").pack(side=tk.LEFT, padx=(8, 2))
        ss_scale = tk.Scale(ctrl, from_=1, to=10, resolution=1, orient=tk.HORIZONTAL,
                            variable=ss_var, length=60)
        ss_scale.pack(side=tk.LEFT, padx=2)
        ss_scale.bind('<ButtonRelease-1>', lambda e: redraw())
        ttk.Label(ctrl, text="Size:").pack(side=tk.LEFT, padx=(10, 2))
        mah_th_var = tk.DoubleVar(value=round(float(state['mah_th']), 2))
        mah_entry = ttk.Entry(ctrl, textvariable=mah_th_var, width=10)
        mah_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="Re-stabilize", command=re_stabilize).pack(side=tk.LEFT, padx=2)
        ttk.Label(ctrl, text="BIC threshold:").pack(side=tk.LEFT, padx=(10, 2))
        bic_th_var = tk.DoubleVar(value=state['bic_th'])
        bic_entry = ttk.Entry(ctrl, textvariable=bic_th_var, width=10)
        bic_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl, text="Re-fit model", command=re_fit_model).pack(side=tk.LEFT, padx=2)
        def on_accept():
            st = ref_state['s']
            viz = ref_viz['v']
            in_b = st['in_bounds']
            result['accepted'] = True
            vind = viz['valid_indices']
            data_range = st.get('_data_range') or getattr(gaussian_model, 'data_range', None)
            sort_range = st.get('_sort_range') or getattr(gaussian_model, 'sort_range', None)
            sort_feat_local = st.get('_sort_feature_idx') or getattr(gaussian_model, 'sort_feature_idx', None)
            mah_th = st['mah_th']
            dc = st.get('density_curve')
            mahal_in_b = st['mahal_d'][in_b]
            prob_density_in_b = prob_density_from_curve_or_formula(mahal_in_b, mah_th, density_curve=dc)
            result['accept_data'] = {
                'spike_ids': vind[in_b],
                'model': GaussianModel(mean=st['center'], covariance=st['covariance'],
                                      bic_threshold=st['bic_th'], mah_threshold=st['mah_th'],
                                      data_range=data_range, sort_range=sort_range, sort_feature_idx=sort_feat_local,
                                      density_curve=dc),
                'feature_indices': list(included_feat_idxs),
                'sort_feature_idx_global': settings.get('sort_feature_idx_global') if settings else None,
            }
            result['accept_data']['curr_max_prob_update'] = np.asarray(prob_density_in_b, dtype=float)
            if reusable:
                on_done_callback(True, result['accept_data'])
            else:
                popup.destroy()
        def on_reject():
            st = ref_state['s']
            viz = ref_viz['v']
            result['accepted'] = False
            rejected_indices = viz['valid_indices'][st['in_bounds']]
            result['accept_data'] = {'rejected_spike_indices': np.asarray(rejected_indices).tolist()}
            if reusable:
                on_done_callback(False, result['accept_data'])
            else:
                popup.destroy()
        ttk.Button(ctrl, text="Accept", command=on_accept).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl, text="Reject", command=on_reject).pack(side=tk.LEFT, padx=5)

        # Data views: scrollable container, fixed height to match additional-graph brackets
        n_scatter = len(feature_pairs)
        ctrl_height = ctrl.winfo_reqheight()
        vis_h = win_h-ctrl_height-10
        canvas_widget, axes_flat = self._show_data_views(vis_h,win_w,n_scatter,popup)
        redraw()
        canvas_widget.draw()
        popup.update_idletasks()
        w = max(popup.winfo_width(), ctrl.winfo_reqwidth())
        h = popup.winfo_reqheight()
        popup.geometry(f"{w}x{h}")

        def update_content(new_viz, new_model, new_seed_idx, new_n_seeds, new_seed_idx_global):
            ref_viz['v'] = new_viz
            ref_state['s'] = _state_from_viz_model(new_viz, new_model)
            ref_seed_global['g'] = new_seed_idx_global
            mah_th_var.set(round(float(ref_state['s']['mah_th']), 2))
            bic_th_var.set(ref_state['s']['bic_th'])
            popup.title(f"Seed {new_seed_idx + 1}/{new_n_seeds} - Accept/Reject Gaussian model")
            redraw()
            canvas_widget.draw()

        if reusable:
            return popup, update_content
        popup.wait_window()
        return result['accepted'], result['accept_data']

    def _show_init_model_debug(self, debug_data, settings):
        """Testing window 1/3: original init model (cov from initial_stds) and in_bounds dots."""
        included_feat_idxs = list(debug_data.get('included_feat_idxs', self.included_feature_indexes or []))
        if not included_feat_idxs:
            return
        init_mah_d = float(debug_data.get('init_mah_d', 1.0))
        feature_pairs = settings.get('viz_feature_pairs', [(0, 1), (0, 2)])
        points = debug_data['points']
        center = debug_data['center']
        covs = debug_data['covariance']
        n_cols = len(included_feat_idxs)
        if points.shape[1] != n_cols or covs.shape != (n_cols, n_cols):
            messagebox.showwarning("Testing", f"Feature/covariance dimension mismatch: points cols={points.shape[1]}, covs={covs.shape}, included={n_cols}")
            return
        mahal_d = debug_data['mahal_d']
        seed_idx = debug_data['seed_idx']
        properties = debug_data['properties']
        in_bounds = debug_data.get('in_bounds', np.array([], dtype=int))
        show_seed = debug_data.get('show_seed', True)
        seed_point = np.copy(properties[seed_idx, :]) if show_seed else None
        in_bounds_mask = np.zeros(len(points), dtype=bool)
        if len(in_bounds) > 0:
            in_bounds_mask[in_bounds] = True
        out_ind = np.where(~in_bounds_mask)[0]
        in_ind = np.where(in_bounds_mask)[0]
        all_points_full = debug_data.get('all_points_full', None)
        n_plots = len(feature_pairs)
        dot_size_var = tk.DoubleVar(value=2.0)
        canvas_widget = None

        def redraw():
            dot_sz = max(0.5, float(dot_size_var.get()))
            th2 = 2.0 * init_mah_d
            sort_out = np.argsort(mahal_d[out_ind])[::-1] if len(out_ind) > 0 else np.array([], dtype=int)
            for plot_idx, (g0, g1) in enumerate(feature_pairs):
                if plot_idx >= len(axes_flat):
                    break
                both_included = (g0 in included_feat_idxs and g1 in included_feat_idxs)
                if both_included:
                    col0 = included_feat_idxs.index(g0)
                    col1 = included_feat_idxs.index(g1)
                    data_2d = points
                else:
                    col0, col1 = g0, g1
                    data_2d = None
                    if all_points_full is not None and all_points_full.shape[0] == len(points):
                        nfc = all_points_full.shape[1]
                        if nfc > max(g0, g1):
                            data_2d = all_points_full
                            col0 = min(g0, nfc - 1)
                            col1 = min(g1, nfc - 1)
                            if col0 == col1:
                                col1 = max(0, col0 - 1)
                if data_2d is None or col0 < 0 or col1 < 0 or col0 >= data_2d.shape[1] or col1 >= data_2d.shape[1]:
                    axes_flat[plot_idx].set_visible(False)
                    continue
                ax = axes_flat[plot_idx]
                ax.clear()
                ax.set_visible(True)
                if len(out_ind) > 0:
                    out_order = out_ind[sort_out]
                    x_out = data_2d[out_order, col0]
                    y_out = data_2d[out_order, col1]
                    d_out = mahal_d[out_order]
                    t = np.clip((d_out - init_mah_d) / (th2 - init_mah_d) if th2 > init_mah_d else 0, 0, 1)
                    colors_out = np.zeros((len(d_out), 3))
                    colors_out[:, 0] = 1.0 - t
                    colors_out[:, 1] = 0.5 * (1.0 - t)
                    colors_out[:, 2] = 0.0
                    alpha_out = 0.1 + 0.7 * (1.0 - t)
                    ax.scatter(x_out, y_out, c=colors_out, s=dot_sz, alpha=alpha_out, zorder=1)
                if len(in_ind) > 0:
                    x_in = data_2d[in_ind, col0]
                    y_in = data_2d[in_ind, col1]
                    ax.scatter(x_in, y_in, c='red', s=dot_sz, alpha=0.8, zorder=2)
                data_min_x = float(np.min(data_2d[:, col0]))
                data_max_x = float(np.max(data_2d[:, col0]))
                data_min_y = float(np.min(data_2d[:, col1]))
                data_max_y = float(np.max(data_2d[:, col1]))
                if both_included:
                    center_2d = np.array([center[col0], center[col1]])
                    cov_2d = np.array([[covs[col0, col0], covs[col0, col1]], [covs[col1, col0], covs[col1, col1]]])
                    ax.scatter(center_2d[0], center_2d[1], s=100, c='red', edgecolors='black', linewidths=2, zorder=5)
                    if seed_point is not None and col0 < len(seed_point) and col1 < len(seed_point):
                        ax.scatter(seed_point[col0], seed_point[col1], s=200, c='orange', edgecolors='black', linewidths=2, zorder=6)
                    self._plot_mahalanobis_ellipse(ax, center_2d, cov_2d, init_mah_d, linestyle='--', color='red', linewidth=1.5)
                    sx = np.sqrt(max(float(cov_2d[0, 0]), 1e-12))
                    sy = np.sqrt(max(float(cov_2d[1, 1]), 1e-12))
                    half_x = 2.0 * init_mah_d * sx
                    half_y = 2.0 * init_mah_d * sy
                    axis_half_x = 1.5 * half_x
                    axis_half_y = 1.5 * half_y
                    xmin = max(center_2d[0] - axis_half_x, data_min_x)
                    xmax = min(center_2d[0] + axis_half_x, data_max_x)
                    ymin = max(center_2d[1] - axis_half_y, data_min_y)
                    ymax = min(center_2d[1] + axis_half_y, data_max_y)
                else:
                    xmin, xmax = data_min_x, data_max_x
                    ymin, ymax = data_min_y, data_max_y
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_xlabel(self.prop_titles[g0] if self.prop_titles else f"F{g0}")
                ax.set_ylabel(self.prop_titles[g1] if self.prop_titles else f"F{g1}")
                ax.grid(True, alpha=0.3)
            for j in range(n_plots, len(axes_flat)):
                axes_flat[j].set_visible(False)
            if canvas_widget is not None:
                canvas_widget.draw()

        popup = tk.Toplevel(self.root)
        popup.title("1/3: Center of mass estimation.")
        popup.transient(self.root)
        popup.grab_set()
        try:
            popup.update_idletasks()
            win_h = int(popup.winfo_screenheight() * 0.9)
            win_w = win_h
            popup.geometry(f"{win_w}x{win_h}+0+0")
        except tk.TclError:
            win_h, win_w = 700, 700
            popup.geometry(f"{win_w}x{win_h}+0+0")
        ctrl = ttk.Frame(popup)
        ctrl.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(ctrl, text="Dot size:", font=("", 10)).pack(side=tk.LEFT, padx=(0, 2))
        dot_scale = tk.Scale(ctrl, from_=0.5, to=20, resolution=0.5, orient=tk.HORIZONTAL,
                             variable=dot_size_var, length=80, font=("", 10))
        dot_scale.pack(side=tk.LEFT, padx=2)
        dot_scale.bind('<ButtonRelease-1>', lambda e: redraw())
        def close_debug():
            popup.destroy()
        ttk.Button(ctrl, text="Close and continue", command=close_debug).pack(side=tk.LEFT, padx=10)
        ctrl_height = 70
        vis_h = win_h - ctrl_height
        n_scatter = len(feature_pairs)
        canvas_widget, axes_flat = self._show_data_views(vis_h,win_w-50,n_scatter,popup)
        redraw()
        canvas_widget.draw()
        popup.update_idletasks()
        w = max(popup.winfo_width(), ctrl.winfo_reqwidth())
        h = min(popup.winfo_height(), popup.winfo_reqheight())
        popup.geometry(f"{w}x{h}")
        popup.wait_window()
    
    def _show_post_com_mahal_debug(self, debug_data, settings):
        """Testing window 2/3: model after COM, recomputed cov, mah_th, density plot; in_bounds by mah_th rule."""
        included_feat_idxs = list(debug_data.get('included_feat_idxs', self.included_feature_indexes or []))
        if not included_feat_idxs:
            return
        init_mah_d = float(debug_data.get('init_mah_d', 1.0))  # COM uses mah_th=1 for init range
        feature_pairs = settings.get('viz_feature_pairs', [(0, 1), (0, 2)])
        points = debug_data['points']
        center = debug_data['center']
        covs = debug_data['covariance']
        n_cols = len(included_feat_idxs)
        if points.shape[1] != n_cols or covs.shape != (n_cols, n_cols):
            messagebox.showwarning("Testing", f"Feature/covariance dimension mismatch: points cols={points.shape[1]}, covs={covs.shape}, included={n_cols}")
            return
        mahal_d = debug_data['mahal_d']
        seed_idx = debug_data['seed_idx']
        properties = debug_data['properties']
        in_bounds = debug_data.get('in_bounds', np.array([], dtype=int))
        show_seed = debug_data.get('show_seed', True)
        seed_point = np.copy(properties[seed_idx, :]) if show_seed else None
        mah_th = debug_data.get('mah_th', None)
        mahal_d_s = debug_data.get('mahal_d_s', None)
        rolling_HC = debug_data.get('rolling_HC', None)
        in_bounds_mask = np.zeros(len(points), dtype=bool)
        if len(in_bounds) > 0:
            in_bounds_mask[in_bounds] = True
        out_ind = np.where(~in_bounds_mask)[0]
        in_ind = np.where(in_bounds_mask)[0]
        valid_indices = debug_data.get('valid_indices', np.arange(len(points)))
        curr_max_prob = None
        if settings and settings.get('curr_max_prob') is not None:
            cmp_full = np.asarray(settings['curr_max_prob'], dtype=float)
            if len(valid_indices) <= len(cmp_full):
                curr_max_prob = cmp_full[valid_indices]
        density_curve = debug_data.get('density_curve')
        if mah_th is not None and mah_th > 0:
            prob_density = prob_density_from_curve_or_formula(mahal_d, mah_th, density_curve=density_curve)
        else:
            prob_density = np.full(len(points), 1.0, dtype=float)
        inside_mah = (mahal_d <= mah_th) if mah_th is not None else np.zeros(len(points), dtype=bool)
        all_points_full = debug_data.get('all_points_full', None)
        n_scatter = len(feature_pairs)
        has_density = mah_th is not None and mahal_d_s is not None and rolling_HC is not None
        ax_den_ref = [None]
        fig_den_ref = [None]
        dot_size_var = tk.DoubleVar(value=2.0)
        canvas_widget = None
        canvas_den_widget = [None]
        _linew = 1.5

        def redraw():
            dot_sz = max(0.5, float(dot_size_var.get()))
            th2 = 2.0 * init_mah_d
            sort_out = np.argsort(mahal_d[out_ind])[::-1] if len(out_ind) > 0 else np.array([], dtype=int)
            for plot_idx, (g0, g1) in enumerate(feature_pairs):
                if plot_idx >= len(axes_flat):
                    break
                both_included = (g0 in included_feat_idxs and g1 in included_feat_idxs)
                if both_included:
                    col0 = included_feat_idxs.index(g0)
                    col1 = included_feat_idxs.index(g1)
                    data_2d = points
                else:
                    col0, col1 = g0, g1
                    data_2d = None
                    if all_points_full is not None and all_points_full.shape[0] == len(points):
                        nfc = all_points_full.shape[1]
                        if nfc > max(g0, g1):
                            data_2d = all_points_full
                            col0 = min(g0, nfc - 1)
                            col1 = min(g1, nfc - 1)
                            if col0 == col1:
                                col1 = max(0, col0 - 1)
                if data_2d is None or col0 < 0 or col1 < 0 or col0 >= data_2d.shape[1] or col1 >= data_2d.shape[1]:
                    axes_flat[plot_idx].set_visible(False)
                    continue
                ax = axes_flat[plot_idx]
                ax.clear()
                ax.set_visible(True)
                if len(out_ind) > 0:
                    out_order = out_ind[sort_out]
                    x_out = data_2d[out_order, col0]
                    y_out = data_2d[out_order, col1]
                    d_out = mahal_d[out_order]
                    t = np.clip((d_out - init_mah_d) / (th2 - init_mah_d) if th2 > init_mah_d else 0, 0, 1)
                    colors_out = np.zeros((len(d_out), 3))
                    colors_out[:, 0] = 1.0 - t
                    colors_out[:, 1] = 0.5 * (1.0 - t)
                    colors_out[:, 2] = 0.0
                    alpha_out = 0.1 + 0.7 * (1.0 - t)
                    ax.scatter(x_out, y_out, c=colors_out, s=dot_sz, alpha=alpha_out, zorder=1)
                if len(in_ind) > 0:
                    x_in = data_2d[in_ind, col0]
                    y_in = data_2d[in_ind, col1]
                    ax.scatter(x_in, y_in, c='red', s=dot_sz, alpha=0.8, zorder=2)
                data_min_x = float(np.min(data_2d[:, col0]))
                data_max_x = float(np.max(data_2d[:, col0]))
                data_min_y = float(np.min(data_2d[:, col1]))
                data_max_y = float(np.max(data_2d[:, col1]))
                if both_included:
                    center_2d = np.array([center[col0], center[col1]])
                    cov_2d = np.array([[covs[col0, col0], covs[col0, col1]], [covs[col1, col0], covs[col1, col1]]])
                    ax.scatter(center_2d[0], center_2d[1], s=100, c='red', edgecolors='black', linewidths=2, zorder=5)
                    if seed_point is not None and col0 < len(seed_point) and col1 < len(seed_point):
                        ax.scatter(seed_point[col0], seed_point[col1], s=200, c='orange', edgecolors='black', linewidths=2, zorder=6)
                    self._plot_mahalanobis_ellipse(ax, center_2d, cov_2d, init_mah_d, linestyle='--', color='red', linewidth=1.5)
                    if mah_th is not None:
                        self._plot_mahalanobis_ellipse(ax, center_2d, cov_2d, mah_th, linestyle='-', color='blue', linewidth=1.5)
                    sx = np.sqrt(max(float(cov_2d[0, 0]), 1e-12))
                    sy = np.sqrt(max(float(cov_2d[1, 1]), 1e-12))
                    mah_for_range = mah_th if mah_th is not None else init_mah_d
                    half_x = 2.0 * mah_for_range * sx
                    half_y = 2.0 * mah_for_range * sy
                    axis_half_x = 1.5 * half_x
                    axis_half_y = 1.5 * half_y
                    xmin = max(center_2d[0] - axis_half_x, data_min_x)
                    xmax = min(center_2d[0] + axis_half_x, data_max_x)
                    ymin = max(center_2d[1] - axis_half_y, data_min_y)
                    ymax = min(center_2d[1] + axis_half_y, data_max_y)
                else:
                    xmin, xmax = data_min_x, data_max_x
                    ymin, ymax = data_min_y, data_max_y
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_xlabel(self.prop_titles[g0] if self.prop_titles else f"F{g0}")
                ax.set_ylabel(self.prop_titles[g1] if self.prop_titles else f"F{g1}")
                ax.grid(True, alpha=0.3)
            if has_density and ax_den_ref[0] is not None:
                ax_den = ax_den_ref[0]
                ax_den.clear()
                ax_den.set_visible(True)
                ax_den.plot(mahal_d_s, rolling_HC, color='black', linewidth=_linew, label='Cluster density')
                ax_den.axvline(x=mah_th, color='red', linestyle='-', linewidth=_linew, label=f'Size={mah_th:.3f}')
                ax_den.set_xlabel('Mahalanobis distance')
                ax_den.set_ylabel('Cluster density')
                ax_den.set_xlim(0, 1.2 * mah_th)
                ax_den.grid(True, alpha=0.3)
                from matplotlib.lines import Line2D
                legend_handles = [
                    Line2D([0], [0],  color='black', linewidth=_linew, label='Cluster density'),
                    Line2D([0], [0], color='red', linestyle='-', linewidth=_linew, label=f'Size={mah_th:.3f}')
                ]
                ax_legend_ref.legend(handles=legend_handles, loc='center', bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax_legend_ref.transAxes, frameon=True)
            for j in range(n_scatter, len(axes_flat)):
                axes_flat[j].set_visible(False)
            if canvas_widget is not None:
                canvas_widget.draw()
            if has_density and canvas_den_widget[0] is not None:
                canvas_den_widget[0].draw()

        popup = tk.Toplevel(self.root)
        popup.title("2/3: Initial model estimation.")
        popup.transient(self.root)
        popup.grab_set()
        try:
            popup.update_idletasks()
            win_h = int(popup.winfo_screenheight() * 0.9)
            win_w = win_h
            popup.geometry(f"{win_w}x{win_h}+0+0")
        except tk.TclError:
            win_h, win_w = 700, 700
            popup.geometry(f"{win_w}x{win_h}+0+0")
        ctrl = ttk.Frame(popup)
        ctrl.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(ctrl, text="Dot size:", font=("", 10)).pack(side=tk.LEFT, padx=(0, 2))
        dot_scale = tk.Scale(ctrl, from_=0.5, to=20, resolution=0.5, orient=tk.HORIZONTAL,
                             variable=dot_size_var, length=80, font=("", 10))
        dot_scale.pack(side=tk.LEFT, padx=2)
        dot_scale.bind('<ButtonRelease-1>', lambda e: redraw())
        def close_debug():
            if fig_den_ref[0] is not None:
                plt.close(fig_den_ref[0])
            popup.destroy()
        ttk.Button(ctrl, text="Close and continue", command=close_debug).pack(side=tk.LEFT, padx=10)
        available_h = win_h - 20
        viz_bracket_w = win_w - 40
        if has_density:
            den_bracket_h = min(available_h // 2, 250)
            den_fig_h_px = min(den_bracket_h-5, int(viz_bracket_w / 2))
            den_fig_w_px = 2 * den_fig_h_px
            dpi_den = 100
            _rc = {k: plt.rcParams[k] for k in ('font.size', 'xtick.labelsize', 'ytick.labelsize')}
            plt.rcParams.update({'font.size': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8})
            fig_den_ref[0] = plt.figure(figsize=(den_fig_w_px / dpi_den, den_fig_h_px / dpi_den), dpi=dpi_den)
            gs = fig_den_ref[0].add_gridspec(1, 2, width_ratios=[3, 1.5])
            ax_den_ref[0] = fig_den_ref[0].add_subplot(gs[0, 0])
            ax_legend_ref = fig_den_ref[0].add_subplot(gs[0, 1])
            ax_legend_ref.clear()
            ax_legend_ref.axis("off")
            fig_den_ref[0].subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.18)
            plt.rcParams.update(_rc)
            den_container = tk.Frame(popup, width=viz_bracket_w, height=den_bracket_h)
            den_container.pack(fill=tk.X, padx=10, pady=5)
            den_container.pack_propagate(False)
            density_lf = ttk.LabelFrame(den_container, text="Cluster density curve", padding=2)
            density_lf.pack(fill=tk.BOTH, expand=False)
            view_canvas_den = tk.Canvas(density_lf, width=viz_bracket_w, height=den_bracket_h)    
            view_inner_den = ttk.Frame(view_canvas_den)
            inner_id = view_canvas_den.create_window((viz_bracket_w//2, 0), window=view_inner_den, anchor="n")
            view_canvas_den.itemconfig(inner_id, width=den_fig_w_px, height=den_bracket_h-25)
            view_canvas_den.pack(fill=tk.BOTH, expand=False)
            canvas_den = FigureCanvasTkAgg(fig_den_ref[0], master=view_inner_den)
            canvas_den.get_tk_widget().pack(fill=tk.BOTH, expand=False)
            canvas_den_widget[0] = canvas_den
            vis_h = available_h - den_bracket_h - 30
        else:
            vis_h = available_h
        
        canvas_widget, axes_flat = self._show_data_views(vis_h,win_w-50,n_scatter,popup)
        redraw()
        canvas_widget.draw()
        popup.update_idletasks()
        w = max(popup.winfo_width(), ctrl.winfo_reqwidth())
        h = min(popup.winfo_height(), popup.winfo_reqheight())
        popup.geometry(f"{w}x{h}")
        popup.wait_window()

    def _show_final_gm_debug(self, visualization_data, gaussian_model, settings):
        """Testing window 3/3: final model and in_bounds (BIC-refined) after GM iteration."""
        if self.included_feature_indexes is None or len(self.included_feature_indexes) < 2:
            return
        points = visualization_data['points']
        valid_indices = np.asarray(visualization_data['valid_indices'], dtype=int)
        mahal_d = np.array(visualization_data['mahal_d'], dtype=float)
        mah_th = float(gaussian_model.mah_threshold)
        center = np.array(gaussian_model.mean, dtype=float)
        covariance = np.array(gaussian_model.covariance, dtype=float)
        seed_point = visualization_data.get('seed_point')
        test_props = settings.get('_test_properties')
        if test_props is not None and settings.get('curr_max_prob') is not None:
            in_bounds_global = gaussian_model.in_bounds_indices(test_props, np.asarray(settings['curr_max_prob'], dtype=float))
            in_bounds = np.where(np.isin(valid_indices, in_bounds_global))[0]
        else:
            in_bounds_raw = np.asarray(visualization_data['in_bounds'], dtype=float)
            if in_bounds_raw.size == 0 or np.any(np.isnan(in_bounds_raw)):
                in_bounds = np.array([], dtype=int)
            else:
                in_bounds = np.asarray(in_bounds_raw, dtype=int)
        iteration_history_full = visualization_data.get('iteration_history_full', [])
        all_points_full = visualization_data.get('all_points_full', None)
        included_feat_idxs = list(self.included_feature_indexes)
        feature_pairs = settings.get('viz_feature_pairs', [(0, 1), (0, 2)])
        feature_pairs = list(feature_pairs)
        in_bounds_mask = np.zeros(len(points), dtype=bool)
        if len(in_bounds) > 0:
            in_bounds_mask[in_bounds] = True
        out_ind = np.where(~in_bounds_mask)[0]
        in_ind = np.where(in_bounds_mask)[0]
        n_scatter = len(feature_pairs)
        has_iter_history = len(iteration_history_full) > 0
        ax_iter_ref = [None]
        ax_legend_ref = [None]
        fig_iter_ref = [None]
        dot_size_var = tk.DoubleVar(value=2.0)
        canvas_widget = [None]
        canvas_iter_widget = [None]

        def redraw():
            dot_sz = max(0.5, float(dot_size_var.get()))
            th2 = 2.0 * mah_th
            sort_out = np.argsort(mahal_d[out_ind])[::-1] if len(out_ind) > 0 else np.array([], dtype=int)
            for plot_idx, (g0, g1) in enumerate(feature_pairs):
                if plot_idx >= len(axes_flat):
                    break
                both_included = (g0 in included_feat_idxs and g1 in included_feat_idxs)
                if both_included:
                    col0 = included_feat_idxs.index(g0)
                    col1 = included_feat_idxs.index(g1)
                    data_2d = points
                else:
                    col0, col1 = g0, g1
                    data_2d = None
                    if all_points_full is not None and all_points_full.shape[0] == len(points):
                        nfc = all_points_full.shape[1]
                        if nfc > max(g0, g1):
                            data_2d = all_points_full
                            col0 = min(g0, nfc - 1)
                            col1 = min(g1, nfc - 1)
                            if col0 == col1:
                                col1 = max(0, col0 - 1)
                if data_2d is None or col0 < 0 or col1 < 0 or col0 >= data_2d.shape[1] or col1 >= data_2d.shape[1]:
                    axes_flat[plot_idx].set_visible(False)
                    continue
                ax = axes_flat[plot_idx]
                ax.clear()
                ax.set_visible(True)
                if len(out_ind) > 0:
                    out_order = out_ind[sort_out]
                    x_out = data_2d[out_order, col0]
                    y_out = data_2d[out_order, col1]
                    d_out = mahal_d[out_order]
                    t = np.clip((d_out - mah_th) / (th2 - mah_th) if th2 > mah_th else 0, 0, 1)
                    colors_out = np.zeros((len(d_out), 3))
                    colors_out[:, 0] = 1.0 - t
                    colors_out[:, 1] = 0.5 * (1.0 - t)
                    colors_out[:, 2] = 0.0
                    alpha_out = 0.1 + 0.7 * (1.0 - t)
                    ax.scatter(x_out, y_out, c=colors_out, s=dot_sz, alpha=alpha_out, zorder=1)
                if len(in_ind) > 0:
                    x_in = data_2d[in_ind, col0]
                    y_in = data_2d[in_ind, col1]
                    ax.scatter(x_in, y_in, c='red', s=dot_sz, alpha=0.8, zorder=2)
                data_min_x = float(np.min(data_2d[:, col0]))
                data_max_x = float(np.max(data_2d[:, col0]))
                data_min_y = float(np.min(data_2d[:, col1]))
                data_max_y = float(np.max(data_2d[:, col1]))
                if both_included:
                    center_2d = np.array([center[col0], center[col1]])
                    cov_2d = np.array([
                        [covariance[col0, col0], covariance[col0, col1]],
                        [covariance[col1, col0], covariance[col1, col1]]
                    ])
                    ax.scatter(center_2d[0], center_2d[1], s=100, c='red', edgecolors='black', linewidths=2, zorder=5)
                    if seed_point is not None and col0 < len(seed_point) and col1 < len(seed_point):
                        ax.scatter(seed_point[col0], seed_point[col1], s=200, c='orange', edgecolors='black', linewidths=2, zorder=6)
                    self._plot_mahalanobis_ellipse(ax, center_2d, cov_2d, mah_th, linestyle='--', color='red', linewidth=1.5)
                    sx = np.sqrt(max(float(cov_2d[0, 0]), 1e-12))
                    sy = np.sqrt(max(float(cov_2d[1, 1]), 1e-12))
                    half_x = 2.0 * mah_th * sx
                    half_y = 2.0 * mah_th * sy
                    axis_half_x = 1.5 * half_x
                    axis_half_y = 1.5 * half_y
                    xmin = max(center_2d[0] - axis_half_x, data_min_x)
                    xmax = min(center_2d[0] + axis_half_x, data_max_x)
                    ymin = max(center_2d[1] - axis_half_y, data_min_y)
                    ymax = min(center_2d[1] + axis_half_y, data_max_y)
                else:
                    xmin, xmax = data_min_x, data_max_x
                    ymin, ymax = data_min_y, data_max_y
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.set_xlabel(self.prop_titles[g0] if self.prop_titles else f"F{g0}")
                ax.set_ylabel(self.prop_titles[g1] if self.prop_titles else f"F{g1}")
                ax.grid(True, alpha=0.3)
            if has_iter_history and ax_iter_ref[0] is not None:
                ax_iter = ax_iter_ref[0]
                ax_iter.clear()
                ax_iter.set_visible(True)
                iters = [h[0] for h in iteration_history_full]
                mah_ths = [h[1] for h in iteration_history_full]
                outcomes = [h[2] for h in iteration_history_full]
                colors = {'collapsed': 'blue', 'exploded': 'red', 'stable': 'green'}
                c = [colors.get(o, 'gray') for o in outcomes]
                ax_iter.scatter(iters, mah_ths, c=c, s=40, alpha=0.8, edgecolors='black', linewidths=0.5)
                ax_iter.set_xlabel('Iteration')
                ax_iter.set_ylabel('Size')
                ax_iter.grid(True, alpha=0.3)
                from matplotlib.lines import Line2D
                legend_handles = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, markeredgecolor='black', markeredgewidth=0.5, label='collapsed'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, markeredgecolor='black', markeredgewidth=0.5, label='exploded'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, markeredgecolor='black', markeredgewidth=0.5, label='stable'),
                ]
                ax_legend_ref.legend(handles=legend_handles, loc='center', bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax_legend_ref.transAxes, frameon=True)
            for j in range(n_scatter, len(axes_flat)):
                axes_flat[j].set_visible(False)
            if canvas_widget is not None:
                canvas_widget.draw()
            if has_iter_history and canvas_iter_widget[0] is not None:
                canvas_iter_widget[0].draw()

        # Final model range per dimension: range = 2 * mah_th * std
        stds = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        final_ranges = 2.0 * mah_th * stds
        def _fmt2(x):
            v = float(x)
            return f"{v:.2f}" if abs(v) >= 1 else (f"{v:.2f}" if abs(v) >= 0.01 else f"{v:.2g}")  # 2 decimals if >=1, else 2 non-zero
        range_lines = [f"Size = {_fmt2(mah_th)}", "Final range per dim:"]
        for i, feat_idx in enumerate(included_feat_idxs):
            name = self.prop_titles[feat_idx] if self.prop_titles and feat_idx < len(self.prop_titles) else f"F{feat_idx}"
            range_lines.append(f"  {name}: {_fmt2(final_ranges[i])}")

        popup = tk.Toplevel(self.root)
        title = "3/3: Final model."
        if not visualization_data.get('success', True):
            title += " [Fit failed]"
        popup.title(title)
        popup.transient(self.root)
        popup.grab_set()
        try:
            popup.update_idletasks()
            win_h = int(popup.winfo_screenheight() * 0.9)
            win_w = win_h
            popup.geometry(f"{win_w}x{win_h}+0+0")
        except tk.TclError:
            win_h, win_w = 700, 700
            popup.geometry(f"{win_w}x{win_h}+0+0")
        range_frame = ttk.LabelFrame(popup, text="Final model ranges", padding=5)
        range_frame.pack(fill=tk.X, padx=10, pady=5)
        range_text = tk.Text(range_frame, height=2, width=80, font=("Consolas", 10))
        range_text.pack(fill=tk.X)
        range_text.insert(tk.END, " |".join(range_lines))
        range_text.config(state=tk.DISABLED)
        ctrl = ttk.Frame(popup)
        ctrl.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(ctrl, text="Dot size:", font=("", 10)).pack(side=tk.LEFT, padx=(0, 2))
        dot_scale = tk.Scale(ctrl, from_=0.5, to=20, resolution=0.5, orient=tk.HORIZONTAL,
                             variable=dot_size_var, length=80, font=("", 10))
        dot_scale.pack(side=tk.LEFT, padx=2)
        dot_scale.bind('<ButtonRelease-1>', lambda e: redraw())
        def close_final():
            if fig_iter_ref[0] is not None:
                plt.close(fig_iter_ref[0])
            popup.destroy()
        ttk.Button(ctrl, text="Close", command=close_final).pack(side=tk.LEFT, padx=10)
        popup.update_idletasks()
        available_h = win_h - 20
        viz_bracket_w = win_w - 40
        if has_iter_history:
            iter_bracket_h = min(available_h // 2, 250)
            iter_fig_h_px = iter_bracket_h-5
            iter_fig_w_px = 2 * iter_fig_h_px
            dpi_iter = 100
            _rc = {k: plt.rcParams[k] for k in ('font.size', 'xtick.labelsize', 'ytick.labelsize')}
            plt.rcParams.update({'font.size': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8})
            fig_iter_ref[0] = plt.figure(figsize=(iter_fig_w_px / dpi_iter, iter_fig_h_px / dpi_iter), dpi=dpi_iter)
            gs = fig_iter_ref[0].add_gridspec(1, 2, width_ratios=[3, 1])
            ax_iter_ref[0] = fig_iter_ref[0].add_subplot(gs[0, 0])
            ax_legend_ref = fig_iter_ref[0].add_subplot(gs[0, 1])
            ax_legend_ref.clear()
            ax_legend_ref.axis("off")
            fig_iter_ref[0].subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.18)
            plt.rcParams.update(_rc)
            iter_container = tk.Frame(popup, width=viz_bracket_w, height=iter_bracket_h)
            iter_container.pack(fill=tk.X, padx=10, pady=5)
            iter_container.pack_propagate(False)
            iter_lf = ttk.LabelFrame(iter_container, text="Size iteration history", padding=2)
            iter_lf.pack(fill=tk.BOTH, expand=False)
            view_canvas_iter = tk.Canvas(iter_lf, width=viz_bracket_w, height=iter_bracket_h)    
            view_inner_iter = ttk.Frame(view_canvas_iter)
            inner_id = view_canvas_iter.create_window((viz_bracket_w//2, 0), window=view_inner_iter, anchor="n")
            view_canvas_iter.itemconfig(inner_id, width=iter_fig_w_px, height=iter_bracket_h-25)
            view_canvas_iter.pack(fill=tk.BOTH, expand=False)
            canvas_iter = FigureCanvasTkAgg(fig_iter_ref[0], master=view_inner_iter)
            canvas_iter.get_tk_widget().pack(fill=tk.BOTH, expand=False)
            canvas_iter_widget[0] = canvas_iter
            vis_h = available_h -iter_bracket_h - 80
        else:
            vis_h = available_h
        
        canvas_widget, axes_flat = self._show_data_views(vis_h,win_w-50,n_scatter,popup)
        redraw()
        canvas_widget.draw()
        popup.update_idletasks()
        w = max(popup.winfo_width(), ctrl.winfo_reqwidth())
        h = min(popup.winfo_height(), popup.winfo_reqheight())
        popup.geometry(f"{w}x{h}")
        popup.wait_window()

    def _show_stability_iteration_debug(self, iteration_history, failure_message=''):
        """Scatter: mah_th (y) vs iteration (x), red=exploded, blue=collapsed."""
        if not iteration_history:
            return
        iters = np.array([h[0] for h in iteration_history])
        mah_ths = np.array([h[1] for h in iteration_history])
        outcomes = np.array([h[2] for h in iteration_history])
        expl = outcomes == 'exploded'
        coll = outcomes == 'collapsed'
        fig, ax = plt.subplots(figsize=(8, 5))
        if np.any(expl):
            ax.scatter(iters[expl], mah_ths[expl], c='red', s=40, label='exploded', zorder=2)
        if np.any(coll):
            ax.scatter(iters[coll], mah_ths[coll], c='blue', s=40, label='collapsed', zorder=2)
        ax.set_xlabel('Iteration number')
        ax.set_ylabel('Mah threshold tried')
        ax.legend()
        ax.grid(True, alpha=0.3)
        title = failure_message if failure_message else 'Model did not reach stability'
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        popup = tk.Toplevel(self.root)
        popup.title("Testing: Stability iteration history")
        popup.transient(self.root)
        popup.grab_set()
        ttk.Label(popup, text=failure_message, wraplength=500).pack(padx=10, pady=5)
        def close_it():
            plt.close(fig)
            popup.destroy()
        ttk.Button(popup, text="Close", command=close_it).pack(pady=5)
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        popup.wait_window()

    def _plot_mahalanobis_ellipse(self, ax, mean_2d, cov_2d, mahal_distance, linestyle='--', color='blue', linewidth=1.5):
        """Plot Mahalanobis distance ellipse for 2D Gaussian."""
        # Compute eigendecomposition of 2D covariance
        eigvals, eigvecs = np.linalg.eigh(cov_2d)
        # Ensure positive eigenvalues
        eigvals = np.maximum(eigvals, 1e-12)
        
        # Get angle of rotation
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        
        # Get width and height of ellipse (scaled by Mahalanobis distance)
        width = 2 * mahal_distance * np.sqrt(eigvals[0])
        height = 2 * mahal_distance * np.sqrt(eigvals[1])
        
        # Create ellipse
        ellipse = Ellipse(mean_2d, width, height, angle=angle, 
                         linestyle=linestyle, edgecolor=color, facecolor='none', linewidth=linewidth)
        ax.add_patch(ellipse)
    
    def _accept_reject_dialog(self, seed_idx, total_seeds, viz_popup=None, viz_fig=None):
        """Show accept/reject dialog and wait for user input. Closes viz window when user accepts or rejects."""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Seed {seed_idx + 1}/{total_seeds}")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.geometry("300x100")
        dialog.resizable(False, False)
        
        result = {'accepted': False}
        
        def _close_viz():
            if viz_popup is not None:
                try:
                    if viz_popup.winfo_exists():
                        viz_popup.destroy()
                except tk.TclError:
                    pass
            if viz_fig is not None:
                try:
                    plt.close(viz_fig)
                except Exception:
                    pass
        
        def accept():
            result['accepted'] = True
            _close_viz()
            dialog.destroy()
        
        def reject():
            result['accepted'] = False
            _close_viz()
            dialog.destroy()
        
        ttk.Label(dialog, text=f"Accept Gaussian model for seed {seed_idx + 1}?").pack(pady=10)
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        accept_btn = ttk.Button(button_frame, text="Accept", command=accept)
        accept_btn.pack(side=tk.LEFT, padx=5)
        
        reject_btn = ttk.Button(button_frame, text="Reject", command=reject)
        reject_btn.pack(side=tk.LEFT, padx=5)
        
        dialog.wait_window()
        return result['accepted']


def main():
    root = tk.Tk()
    app = GMCSorter(root)
    root.mainloop()


if __name__ == "__main__":
    main()
