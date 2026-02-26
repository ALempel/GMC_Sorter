# Gaussian model from seed: why “large range + small group of dots” can be stable

## Pipeline overview

1. **`fit_gaussian_model_from_seed`**  
   - Builds initial model from seed via `_create_init_cluster_from_seed`.  
   - Runs **`_fit_model_size`** to find a stable model size (mah_threshold).  
   - Returns model + visualization data; viz uses **BIC-refined** `in_bounds` for the “dots”.

2. **`_fit_model_size`** (bisection on size)  
   - Tries mah_threshold values between S1 and S2.  
   - For each size, calls **`iterate_GM_model`**.  
   - **“Stable”** means: `iterate_GM_model` returned `outcome='stable'`.  
   - When the bracket is narrow and S1 was stable, it re-runs `iterate_GM_model` at S1 and keeps that model + assignment.

3. **`iterate_GM_model`** (inner refinement at fixed size)  
   - **Input:** model with a **fixed** `mah_threshold` (and thus fixed ellipse “size”).  
   - Loop:  
     - `assignment` = in_bounds from current model (prob_density > curr_max_prob).  
     - **Refresh** mean/covariance from `points[assignment]`.  
     - Recompute `assignment` = in_bounds with new mean/cov (same mah_th).  
     - **BIC refinement:** `assignment = apply_bic_refinement(...)` (can shrink to one sub-cluster).  
     - If assignment shrank and we already did BIC once → **explode** (“repeated BIC refinement”).  
     - If assignment shrank first time → **continue** (next iteration uses the smaller set to refresh).  
     - Else compare `len(assignment)` to `len(previous_assignment)`:  
       - Shrink &gt; min_change → **collapse**.  
       - Grow &gt; min_change → **continue**.  
       - Else → **break** and return **stable** (model = last refresh, assignment = current BIC-refined set).

So **stability is only “assignment count converged”**: no collapse, no explosion, and the number of in-bounds points (after BIC) stopped changing beyond `min_change`.

---

## Why you can get “large range” but “small group of dots”

- **Range** (what you see as the big window/ellipse) comes from the **model**:
  - `sort_range = center ± std * mah_threshold`  
    (`compute_data_range`, using diagonal of current covariance for the sort feature.)
  - So range is driven by **mah_threshold** (the size chosen by bisection) and **std** from the **current** covariance.

- **Dots** in the accept/reject window are the **BIC-refined** in_bounds:
  - In `fit_gaussian_model_from_seed`, viz uses `in_bounds_all = apply_bic_refinement(...)` (lines 1634–1639), so the highlighted “in cluster” set is the BIC-refined subset, not the full mahal-based in_bounds.

So you can get:

1. Bisection finds a **large** mah_threshold S1 where the inner loop “stabilizes” (assignment count stops changing).
2. During that inner loop, BIC refinement **shrinks** assignment to a small subset (one sub-cluster); we refresh mean/cov from that subset and then assignment count stabilizes → we return **stable**.
3. The returned model has:
   - **mah_threshold = S1** (large),
   - **mean/covariance** from the **small** BIC-refined set.
4. **sort_range** is then:
   - `center ± std * mah_threshold`  
   with **std** from that small set’s covariance. If that subset has non-trivial spread on the sort feature (or covariance is not tiny), **std * S1** can still be large → **large range**.
5. The UI shows:
   - **Large range** = viz window from that sort_range (plus margin).
   - **Small group of dots** = BIC-refined in_bounds.

So the model is “stable” in the code’s sense (count converged after BIC), but geometrically you see a large ellipse/window and a small cluster of dots because:

- **Stability does not check** that the ellipse “fits” the cluster tightly; it only checks that the **number** of points in the refined assignment stopped changing.
- **mah_threshold** is **never reduced** after BIC shrinks the set; it stays at the size that passed the stability test.
- **Range** uses that same mah_threshold and the (possibly moderate) std of the small subset, so the displayed range can remain large while the **displayed dots** are only the BIC-refined subset.

---

## Summary

| Concept | What it is |
|--------|------------|
| **Stable** | `iterate_GM_model` exited with assignment count unchanged (within `min_change`), after BIC refinement. |
| **Range** | `sort_range` = center ± std × mah_threshold; mah_threshold is the bisection size; std from current (BIC-refined) covariance. |
| **Dots in UI** | BIC-refined in_bounds only. |
| **Why large range + small dots** | Size (mah_threshold) was chosen where count converged; BIC then shrunk the set to a small subset; model keeps that large size and uses cov of the small set, so range can stay large while only the small subset is shown as “in bounds”. |

So such a model is consistent with the current pipeline: it **is** “stable” by the implemented criterion (count convergence), even though visually it looks like a large ellipse with a small group of dots due to BIC refinement.
