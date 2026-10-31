# Music recommendation research

Unsupervised feature selection, clustering, membership analysis, and listening-test exports for 25,000 FMA medium tracks. Use [FeatureSelectionforClustering.ipynb](FeatureSelectionforClustering.ipynb) with this directory as the working directory and the local `.venv` interpreter. `.env` supplies `FMA_METADATA_DIR`, normally pointing to `../fma_metadata`. Original notebooks remain in `../archive`.

## Current experiment

The active policy is `fixed_candidate`, selecting `chained:rho95_var01_group30`. In saved run **`20261029T235248511652Z`**, this produces **180 original features: 101 segment-derived and 79 global**, out of 1,064 eligible descriptors. All 25,000 tracks remain in the clustering sweep and final fits.

The chain applies absolute Spearman filtering at 0.95, variance above 0.01 after development-fitted Min-Max scaling, and average-linkage feature grouping at distance 0.30. The resulting count happens to be 180; it is not rounded or imposed by a feature-count increment. Ranked methods separately use coarse steps of 50 and refinement steps of 10 and 1.

Development uses 20,000 tracks; the disjoint confirmation sample uses 5,000. Imputation/scaling are fitted on development data. No genre labels are used. The compact subset has:

| Diagnostic | Saved value |
|---|---:|
| Development reference-neighbour overlap | 48.08% |
| Confirmation reference-neighbour overlap | 52.83% |
| Selection stability, mean Jaccard | 99.45% |
| Earlier preservation requirement | 70% |

**The compact subset did not pass the earlier preservation requirement.** It is explicitly authorized as a fixed-candidate experiment; `confirmation_passed` remains false and `clustering_authorized` is recorded separately. This exploratory choice followed inspection of runtime and earlier results. Stability and preservation do not establish listener-perceived music similarity.

The alternative `smallest_qualifying` and `near_best_preservation` policies remain available and enforce their confirmation requirements. The latter selected a much wider representation in earlier experiments; those results must not be presented as measurements of the compact subset.

## Run or reuse the notebook

From PowerShell, without activating the environment:

```powershell
Set-Location 'C:\Users\salom\Documents\PHD\music-recommendation-research'
& .\.venv\Scripts\python.exe -m jupyterlab FeatureSelectionforClustering.ipynb
```

Run imports, rendering helpers, configuration, and project-path cells first, then choose one route:

1. **New experiment:** use Alternative A; skip Alternative B. Load the feature tables and run preprocessing, candidate comparison, selection, and confirmation diagnostics. Saving a changed feature list creates a separate run destination so earlier results are retained.
2. **Reuse a saved run:** set `SAVED_RUN_ID` to the desired run before running Alternative B; the configuration may still default to an earlier run. For the current compact artifacts, use `20261029T235248511652Z`. Skip expensive selection and clustering computations. Rerun Kneed detection, the shortlist cell, and model setup to apply the corrected elbow rule to saved curves.

When switching existing in-memory candidates to the compact experiment, rerun configuration and the selection/confirmation/saving cells. Existing candidate tables can be reused. Do not reuse curves or fitted models computed from a different feature representation as results for the new selection.

Catalog preparation needs raw tables with the matching preprocessing, or an exactly compatible saved bundle. An old wide-feature KMeans bundle cannot supply the current 180-feature representation. When raw data and preprocessing are in memory, either catalog-preparation cell uses the current selected features. Expected compact catalog shape is `(25000, 180)`.

Fit/save cells are separate for each final model. Skip fitting when the matching artifacts already exist. Load assignments before full-catalog results, agreement, and listening-test generation. There is no automatic resume mode; do not Run All when reusing saved results.

### Terminal execution

```powershell
# Execute a fresh experiment; stream output and retain a log:
& .\.venv\Scripts\python.exe -u .\run_research.py 2>&1 | Tee-Object -FilePath research_run.log

# Prepare a fresh-run notebook without executing it:
& .\.venv\Scripts\python.exe .\run_research.py --prepare-only

# Include listening-test generation in the fresh run:
& .\.venv\Scripts\python.exe -u .\run_research.py --include-listening-tests 2>&1 | Tee-Object -FilePath research_run.log
```

The runner excludes saved-results loading, the optional GMM benchmark, and the alternate catalog-loading cell. By default it stops after agreement plots, before the detailed KMeans/GMM overlap and listening-test section. `--include-listening-tests` includes those sections. The final saved-results summary is excluded in both cases.

Each invocation starts a new run. A separate notebook is written under `artifacts/executed_notebooks/`; output is streamed and progress saved after cells, including on failure. A stopped sweep is not automatically resumed, and its scores CSV is written after the sweep completes. Execution can require substantial time and memory.

## Clustering and GMM timing

The sweep evaluates k=7 through 40 on all catalog tracks:

- KMeans and GMM use seeds `[0, 42, 123]`, one fit per seed and k.
- GMM uses full covariance, **one initialization per seed**, and the default 100-iteration limit.
- Ward and Birch run once per k; Ward caches its full hierarchy across k values.
- There are 272 estimator fit calls in total. Native threads are limited to four in the sweep and final fits.
- Sweep silhouette uses at most 5,000 tracks; final-result silhouette uses at most 1,000.

The optional benchmark immediately before the sweep uses the selected features and all 25,000 tracks, the first configured k/seed, full covariance, one initialization, and at most 100 iterations. Unlike the sweep, it has no explicit native-thread cap. It prints a start marker, elapsed-time updates every 15 seconds, and preparation, KMeans initialization, per-iteration, fit, and metric timings. It does not change the sweep or final-fit settings. Interrupt an earlier running cell before launching it.

A convergence warning means the fit did not reach its likelihood-change tolerance within the iteration limit. A completed fit or saved metrics alone do not establish convergence. Increasing `max_iter` permits more work but does not guarantee convergence. The current compact subset reduces covariance dimensions without reducing the catalog; runtime must be measured for each configuration.

### Elbow selection

Kneed uses original mean assignment distortion (`WCSS / number of tracks`) with a convex decreasing curve and sensitivity 1. Local increases no longer prevent Kneed from running. GMM optimizes likelihood rather than assignment WCSS, so its distortion may increase locally. Such selections are retained with `candidate_detected_non_monotonic` for review, without smoothing or replacing the curve.

Sparse, flat, nonfinite, non-decreasing-overall curves and undetected elbows remain unresolved. There is no metric-ranking fallback. Unresolved fit/save cells report the missing choice and skip; `MANUAL_CLUSTER_COUNTS` in model setup allows an explicit choice among evaluated k values after curve review. Manual choices are recorded separately from automatic elbows in `final_model_selections.csv`.

The current saved curves select **KMeans 21, GMM 14, Ward 19, and Birch 20**. GMM k=14 has the non-monotonic-curve flag. Other internal metrics remain diagnostics; balance failure is flagged rather than silently changing k.

## Saved models and results

The compact run has these ordinary sklearn estimators and assignments:

```text
models/20261029T235248511652Z/simple_elbow/
    KMeans_k21.pkl
    GaussianMixture_k14.pkl
    Ward_k19.pkl
    Birch_k20.pkl
    <model>_k<count>_assignments.csv
    preprocessing.pkl
```

Each assignment CSV contains 25,000 tracks. Shared preprocessing stores the fitted imputer/scaler and ordered input/selected feature names. Ward membership is retained through assignments because it has no `predict` method. Final randomized estimators use seed 42; three-seed replication belongs to screening. GMM's final fit uses one initialization.

Birch is saved as a shallow inference copy without its recursive CF-tree links. Centers, labels, and predictions remain available, but the saved copy does not support incremental `partial_fit`. Earlier model bundles remain separate.

Loading reports missing model files. Assignment loading checks unique IDs and complete current-catalog coverage and includes only available models. It reports omissions; partial loading is not a complete four-model comparison. ARI/AMI tables compare the same track IDs. The detailed KMeans/GMM overlap skips explicitly if either model is missing, without substituting another model.

Verified final metrics for run `20261029T235248511652Z`:

| Model | k | Silhouette | CH | DB | Distortion | Gini | Smallest cluster |
|---|---:|---:|---:|---:|---:|---:|---:|
| KMeans | 21 | 0.0113 | 441.56 | 3.6203 | 2.4213 | 0.2632 | 361 |
| GaussianMixture | 14 | -0.0200 | 357.47 | 4.8473 | 2.7633 | 0.3747 | 623 |
| Ward | 19 | -0.0190 | 342.38 | 4.4286 | 2.6288 | 0.3510 | 298 |
| Birch | 20 | -0.0150 | 330.09 | 4.6286 | 2.6196 | 0.3365 | 378 |

CH, DB, distortion, Gini, and assignments use all catalog tracks; silhouette is sampled on 1,000. Set `FULL_SILHOUETTE_SAMPLE_SIZE=None` for the more expensive exact score. KMeans/GMM membership ARI is 0.1146. Near-zero or negative silhouette indicates limited separation; neither internal metrics nor differing partitions establish perceptual recommendation quality. These saved measurements do not certify GMM convergence.

## Plots and experiment records

Every `show_figure` call displays a PNG inline and saves PNG (1,000 DPI), SVG, and PDF to `outputs/<RUN_ID>/plots/`, printing the absolute paths. For example, `show_figure(fig, 'feature_set_overlap')` saves that basename in all three formats. Setup must define `PLOTS_DIR`; the helper reports an error if it is absent. This notebook does not use `_plots`.

Important per-run records include:

- `selection.json`, `selected_features.csv`, feature-selection comparison/search/overlap tables, and development/confirmation track IDs.
- `confirmation_comparison.csv`, including the actual preservation outcome even for an authorized fixed-candidate experiment.
- `clustering_scores_by_seed.csv`, `clustering_results.csv`, and `clustering_evaluation_tracks.csv`.
- `clustering_elbow_candidates.csv`, `elbow_selected_models.csv`, and `final_model_selections.csv`.
- `simple_elbow_model_results.csv`, `simple_model_ari.csv`, and `simple_model_ami.csv`.

A changed feature list gets a new run folder; plots created before that switch stay in the earlier folder unless their cells are rerun. Do not mix model files, preprocessing, curves, or task banks from different representations.

## Listening-test banks

`requested_test_models` defaults to KMeans and GMM. Planning includes only models with evaluated results and loaded assignments and reports missing models. Generation skips when none are ready; a one-model export is explicitly incomplete for the requested study.

Run the section after catalog preparation, assignment loading, and full-catalog results. The current saved manifest has all four banks:

| Model | Ranking tasks | A/B tasks |
|---|---:|---:|
| KMeans, 21 clusters | 42 | 42 |
| GMM, 14 clusters | 28 | 28 |

Files are saved under `outputs/<RUN_ID>/listening_tests/<model>/<ranking|ab>/`, with parent `manifest.csv` and `coverage.csv`. Counts describe generated tasks, not participant responses.

Ranking uses the available real song nearest the empirical cluster centroid as reference, with two disjoint five-candidate panels sampled from five equal-count distance bands. Adjacent distances must differ by at least 5% of the pool's 5th-95th percentile span. Clusters failing coverage constraints are reported rather than relaxing them. Model-predicted ranking is a hypothesis for listeners to assess.

A/B generates typical and nearest-other-cluster comparisons per reference, randomizes positions, and keeps membership/condition/distance metadata private. Exact file hashes, identical feature vectors, and matching normalized artist/title metadata form duplicate groups; alternate recordings can remain undetected.

Distribute only `participant_tasks.json`. Keep `private_design.csv`, `manifest.csv`, and `coverage.csv` private. Verify playback and excerpt identity, inspect coverage, and divide the bank into participant blocks before distribution. These exports need adaptation in the listening-test interface and do not collect participant responses.

Data, generated outputs, models, checkpoints, environments, and IDE settings are ignored by Git. Notebook backups remain under `artifacts/notebook_backups/`. Historical experiments are retained as separate records; the current notebook summary describes the compact run above.
