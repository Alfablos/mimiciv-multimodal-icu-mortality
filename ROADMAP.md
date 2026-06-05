# Roadmap Ideas

* ~~Basic orchestration~~

* Baseline model (Decision Tree/Logistic Regression and CNN with binary classification) for standalone tabular-only and visual-only predictions.

* Full orchestration:
    1. Orchestration also compares to baseline model
    2. Model deploy (platform: to be defined)
    3. Model test (test set, shadow testing, Kubernetes + traffic mirroring?)
    4. Revert to previous model if performance on the test set is not better than that on the validation test, log situation

* Monitoring

* Dashboard + explanatory LLM (monitoring system connector) + trusted frontend components

* GCP/AWS training

* Federated training

* Serving and ONNX/Burn inference (Maturin + PyO3 for Rust inference engine):
    1. Add `mmim-serve` to load promoted models from MLflow by alias.
    2. Export trained PyTorch models to ONNX and log them with `mlflow.onnx` (ONNX + pyfunc) instead of using pytorch + pyfunc (current behavior).
    3. Add parity checks (quality gate!) between PyTorch, ONNX Runtime, and Burn inference: guarantees that we're running the intended model.
    4. Use Burn (ONNX) as an inference backend

  Python is the interface with the client and MLFlow, Rust is for the inference backend only.

  Advantages:
    * CPU inference performance gain (needs benchmarking)
    * GPU inference performance gain (needs benchmarking)
    * CPU/WebGPU/WASM portability: if a frontend is ever built and the model is small enough it could even run in the browser (if weights are open source) via WASM (WebGPU) without much coding effort, feasibility depends on model size, browser performance (not Italian public health), preprocessing parity, and model artifact security.
    * The inference backend could be ported in a project where a CLI/Desktop App takes in data and runs inference, decoupling from python
    * The inference runtime is more predictable and constrained (no python dependencies for the inference backend)
