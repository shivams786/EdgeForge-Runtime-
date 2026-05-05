# Build And Run

## Configure With ONNX Runtime

```bash
cmake -S . -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build
```

`ONNXRUNTIME_ROOT` must contain `include/onnxruntime_cxx_api.h` and a library directory with `libonnxruntime.so`, `libonnxruntime.dylib`, or `onnxruntime.lib`.

## Configure Without ONNX Runtime

```bash
cmake -S . -B build -DEDGE_AI_ENABLE_ONNXRUNTIME=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```

This mode builds parser, scheduler, logger, and metrics tests. Inference calls return a clear backend-disabled failure.

## Generate Assets

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r python_tools/requirements.txt
python python_tools/export_model_to_onnx.py
python python_tools/generate_sample_input.py
```

## Run CLI

```bash
./build/edge_ai_runtime --model models/sample_model.onnx --input sample_inputs/input_vector.txt --runner onnx
./build/edge_ai_runtime --jobs configs/sample_jobs.json --schedule fifo --metrics
./build/edge_ai_runtime --jobs configs/sample_jobs.json --schedule priority --metrics
```
