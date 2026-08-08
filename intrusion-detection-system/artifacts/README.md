# Model artifacts

Generated model bundles are intentionally excluded from Git. Serialized preprocessing pipelines can contain raw categorical vocabulary copied from training traffic, and runtime audit logs can contain network telemetry.

To run the API, place a reviewed deployment bundle in `artifacts/se_dwnet_edge_iiotset_random_holdout/` containing:

- `se_dwnet_model.keras`
- `preprocessing_pipeline.pkl`
- `final_features.txt`

You can use another directory or filename by setting `TON_IOT_ARTIFACT_DIR` and `TON_IOT_MODEL_FILENAME`. Audit every bundle for credentials, tokens, payload samples, internal addresses, and personal paths before distributing it. Prefer a private artifact store or a release asset with appropriate access controls for production models.
