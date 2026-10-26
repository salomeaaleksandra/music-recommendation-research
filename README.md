# Music recommendation research

Use `FeatureSelectionforClustering.ipynb` with the local `.venv` interpreter and this directory as the working directory. `.env` points `FMA_METADATA_DIR` to the shared `../fma_metadata` directory. Original research remains in `../archive`.

## Running the notebook

Run imports, rendering, configuration and paths first. Choose **Start a new run** to recompute experiments, or **Load saved results** to reuse `SAVED_RUN_ID`. Skip expensive cells manually. There is no automatic resume mode.

Feature selection compares unsupervised methods on a common representation, checks preservation/stability and confirms the chosen subset on separate tracks. The completed selection contains 788 features. The cached k=7..40 curves provide elbow choices: K-Means 20, GMM 18, Ward 19, Birch 20. Other metrics remain available for inspection.

## Four plain model PKLs

The final section is organized as short notebook cells: one fit cell and one save cell per model. Saved models can be loaded directly, without fitting. The model directory is:

```text
models/<run_id>/simple_elbow/
    KMeans_k20.pkl
    GaussianMixture_k18.pkl
    Ward_k19.pkl
    Birch_k20.pkl
    <model>_k<count>_assignments.csv
    preprocessing.pkl
```

The four model files are ordinary sklearn estimators, not bundles. `preprocessing.pkl` contains the fitted imputer/scaler and ordered feature names. Assignments retain membership without requiring prediction, including for Ward. The catalog feature-loading cell can reuse the existing trusted K-Means bundle's selected/scaled vectors, avoiding large feature-table reads and duplication.

Birch is saved as a shallow inference copy without its recursive CF-tree links. Centers, labels and predictions are preserved, and the original fitted estimator is unchanged. The saved copy does not support incremental `partial_fit`. Earlier bundles and the failed Birch file remain untouched.

Final models use all 25,000 tracks. Silhouette is sampled by default; set `FULL_SILHOUETTE_SAMPLE_SIZE=None` for exact all-track silhouette at greater cost. CH, DB, distortion, Gini and assignments use all tracks. GMM uses full covariance and three initializations; its full-data fitting can take substantial time. Skip its fit cell when the PKL already exists.

## Analysis and listening tests

Load assignment CSVs to compare models quickly. Inline dataframes show model results, ARI/AMI membership agreement and cluster intersections. Choose one or two models explicitly in `test_models` after reviewing quality, membership differences and task workload.

Ranking tasks use the available song nearest each cluster's empirical centroid as a reference and two disjoint panels of five distance-spanning within-cluster candidates. Initial order is randomized. A private CSV records expected ranks; participant JSON contains only task/reference/candidate IDs. Coverage reports clusters too small for the requested panels. The user-test interface can consume these files after adaptation to drag ranking.

Audio is resolved by track ID. File existence does not verify playback or that an excerpt matches the feature-extraction run. Distribute the task bank across participant blocks; two panels for 20 clusters means 40 ranking tasks rather than 40 individual drag operations.

Data, models, generated outputs, notebook checkpoints, environments and IDE settings remain ignored by Git. Notebook snapshots are retained under `artifacts/notebook_backups/`.
