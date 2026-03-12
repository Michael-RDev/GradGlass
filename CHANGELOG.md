# Changelog

All notable changes to GradGlass are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [1.0.0] — 2026-03-11

First public release.

### Added
- `Run` — training run tracker with PyTorch, Keras/TF, scikit-learn, XGBoost, and LightGBM support
- `GradGlass.run()` / singleton `gg` — one-line run creation
- Gradient and activation capture via forward/backward hooks
- Automatic checkpointing (`run.checkpoint()`, `run.checkpoint_every()`)
- Per-step metric logging (`run.log()`, `run.log_batch()`)
- Data leakage detection engine with 8 statistical checks (`run.check_leakage()`)
- Built-in analysis suite with 50+ automated tests across artifact, model, metrics, gradient, activation, prediction, data, distributed, and reproducibility categories
- Diff engine — layer-level weight diff with Frobenius norm, cosine similarity, and percent-changed metrics
- React 18 + Vite dashboard shipped inside the wheel — served at `http://localhost:8432` with zero external dependencies
- Dashboard pages: Runs, Story Mode, Eval Lab, Behavior Explorer, Root Cause Map, Diff Explorer, Architecture, Checkpoints, Gradients, Analysis, Leakage
- `gradglass` CLI: `serve`, `ls`, `open`, `monitor`, `analyze`
- `@test` decorator for registering custom analysis tests
- Full PyTorch DataLoader leakage helper (`run.check_leakage_from_loaders()`)
- XGBoost and LightGBM native callback integration
- Keras callback (`run.keras_callback()`)
- `run.monitor()` / `run.serve()` — live in-training dashboard
- Freeze-layer recommendation based on gradient activity
- SHAP-proxy and LIME-proxy attribution tests

### Fixed
- `LeakageDetector` subsampling is now reproducible via `random_state` parameter
- Server startup no longer relies on a fixed `time.sleep(1.0)`; uses a socket readiness poll instead
- Git commit hash is captured once at `Run.__init__` time rather than on every `log()` / `checkpoint()` call
- Dashboard confusion matrix colours now use inline CSS instead of dynamically generated Tailwind class names that were not included in the production bundle
- Dashboard top bar now shows the live server host/port instead of the hardcoded string `localhost:8432`
