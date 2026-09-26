# Source audit

Source notebook SHA-256 identifiers:

- `steinifrank.ipynb`: `aaf7472f3e6fcb82272e5d434f32affc6f5129693e7dc93fce32b076897409a9`
- `33ee33ee.ipynb`: `146a5ed512be5c918706a95e1dcbd03b01bb9f9b5154d7dfbcf2f287d78304e1`
- `loraforge.ipynb`: `5f0739190f39fa0dbc63fe18dd485633c6506cdfaf313d8063ebe162f9e69c64`

## Incorporated concepts

- `steinifrank.ipynb`: staged optimization, emphasis on final-answer supervision, separate retention/replay examples, memory provenance, real adapter audits and fail-closed packaging.
- `loraforge.ipynb`: configurable rank/alpha/dropout, module selection, small checkpoints, and phase progress logging.
- `33ee33ee.ipynb`: the workbench concept and configurable loss emphasis.

## Corrected or excluded behavior

- The original `loraforge.ipynb` defaults to packaging an existing seed adapter and exits without training. Standalone Forge always optimizes; it never reports a copied seed as a trained result.
- `33ee33ee.ipynb` uses a small character-level network, then fabricates larger model-shaped tensors for packaging. Standalone Forge exports only the actual PEFT adapter attached to the selected base model.
- Rank is never changed by editing metadata to claim different tensor dimensions.
- No fallback to a supposedly high-scoring seed after nonfinite loss.
- No fabricated target score, multilingual vote agreement, or inferred confidence labels. Duplicated answers are not independent evidence.
- No automatic competition-data scan, arbitrary wheel execution, hidden synthetic data fallback, manual monkeypatch declaring quantizers trainable, or competition-specific harness coupling.
- Source memory mechanisms are reduced to an explicit provenance/split ledger plus user-selected replay files. Progressive retrieval, KL distillation, automatic reasoning compression and model-specific manual LoRA wrappers are not implemented in v1.

The application is a fresh modular implementation grounded in those concepts. Original notebooks are unchanged. CUDA NF4 is implemented against public PEFT/Transformers APIs but remains hardware-unverified in this build.
