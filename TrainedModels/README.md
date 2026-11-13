# Local Model Artifacts & Versioning

Doc2Graph relies on lightweight classical ML models (scikit-learn + rapidfuzz) to run efficiently on CPU without transformer or GPU dependencies.

## Canonical Model Paths

The canonical models loaded by the pipeline are defined in `Configuration/Settings.yaml` (and overridable via environment variables in `KnowledgeGraphApp/Settings.py`):

1. **Relation Scorer**: `TrainedModels/RelationScorer.joblib`
   - Fitted scikit-learn pipeline with `feature_version: 2`.
   - Evaluates dependency parse triples and entity type pairs.
   - Trained on triples the rule extractor produces on Re-DocRED training documents, labelled against the gold relations; scored on held-out development documents. Report: `RelationScorer.training.json`.
   - Retrain via:
     ```bash
     python Tools/TrainRelationScorer.py
     ```

2. **Entity Resolver**: `TrainedModels/EntityResolver.joblib`
   - RapidFuzz string distance, token set, and acronym matching classifier for intra- and cross-document coreference.
   - Trained on Re-DocRED alias clusters (names of the same entity vs. names of different entities). The merge threshold is tuned on held-out documents; copy the reported value to `resolution_threshold` in `Configuration/Settings.yaml`. Report: `EntityResolver.training.json`.
   - Retrain via:
     ```bash
     python Tools/TrainEntityResolver.py
     ```

The hand-written files in `Datasets/EvalGold/` (about 20 rows each) are too small to train on. The trainers only use them as a sanity check.

## Artifact Versioning & Distribution

- Dead or legacy binaries (such as `.v1.joblib` checkpoints) are omitted from version control and Docker builds.
- When scaling or distributing larger weight sets, model artifacts can be tracked via Git LFS, DVC, or pulled at container startup via an initialization script.
