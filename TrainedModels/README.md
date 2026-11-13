# Local Model Artifacts & Versioning

Doc2Graph relies on lightweight classical ML models (scikit-learn + rapidfuzz) to run efficiently on CPU without transformer or GPU dependencies.

## Canonical Model Paths

The canonical models loaded by the pipeline are defined in `Configuration/Settings.yaml` (and overridable via environment variables in `KnowledgeGraphApp/Settings.py`):

1. **Relation Scorer**: `TrainedModels/RelationScorer.joblib`
   - Fitted scikit-learn pipeline with `feature_version: 2`.
   - Evaluates dependency parse triples and entity type pairs.
   - Retrain via:
     ```bash
     python -c "from KnowledgeGraphApp.Extraction.RelationScorer import RelationScorer; print(RelationScorer.train('Datasets/EvalGold/RelationsGold.jsonl'))"
     ```

2. **Entity Resolver**: `TrainedModels/EntityResolver.joblib`
   - RapidFuzz string distance, token set, and acronym matching classifier for intra- and cross-document coreference.
   - Retrain via:
     ```bash
     python -c "from KnowledgeGraphApp.GraphStorage.EntityResolver import EntityResolver; print(EntityResolver.train('Datasets/EvalGold/ResolutionGold.jsonl'))"
     ```

## Artifact Versioning & Distribution

- Dead or legacy binaries (such as `.v1.joblib` checkpoints) are omitted from version control and Docker builds.
- When scaling or distributing larger weight sets, model artifacts can be tracked via Git LFS, DVC, or pulled at container startup via an initialization script.
