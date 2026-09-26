# LoRA Forge Implementation Plan

Goal: deliver a standalone, real PEFT fine-tuner derived from the supplied notebooks.
Architecture: pure-Python data/config core, lazy-loaded ML engine, CLI, and loopback HTTP dashboard. Native implementation in this session.
Spec: DESIGN.md.

Global constraints: no fabricated tensors or scores; no seed-only success; no silent input fallback; no notebook or competition dependency; preserve final answers and validation isolation.

Review focus: conflicting data labels; prompt/validation overlap; EOS equal to padding; resume after cancellation; failed training leaving a misleading export.

1. Data/config: write failing unittest contracts, implement strict configuration and normalization/splits/token masks, run all tests. Files: loraforge/config.py, data.py, tests/test_core.py.
2. Engine/export: test a locally generated tiny model's real parameter updates, frozen base weights, saved/reloaded adapter and resumed execution. Implement checkpointing, evaluation, weighted loss, optional NF4, and truthful status. Files: engine.py, artifacts.py, smoke.py, tests/test_training.py.
3. Application: validate CLI errors and HTTP lifecycle, implement dashboard and shell-free worker management. Files: cli.py, server.py, web/index.html, tests/test_server.py.
4. Delivery: write launchers, presets, data examples and source audit; run the suite, smoke, HTTP checks and package build. Save a ZIP containing all application files and verification evidence.

Ruling: the user's explicit build request authorizes implementation; no extra design approval is needed under the session's autonomy instructions.
