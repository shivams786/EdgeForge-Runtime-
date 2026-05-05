# Edge AI Inference Orchestration Runtime

A production-style C++17 runtime for loading ONNX models, scheduling inference jobs, executing them through ONNX Runtime, and recording structured performance metrics. The project is intentionally small enough to study but shaped like an SDK component: separated interfaces, runner factory, CMake build, CLI, tests, Python model tooling, and Linux-first behavior with Windows-friendly guards.

## Why This Exists

Edge AI systems need more than a model file. They need loading, backend selection, job orchestration, error handling, observability, and build discipline. This repository demonstrates the kind of runtime substrate used in Qualcomm/NVIDIA-style SDK and workflow systems, with room to extend into QNN, SNPE, TensorRT, LiteRT, or vendor-specific execution providers.

## Features

- ONNX Runtime C++ backend through `OnnxModelRunner`
- Plugin-ready `IModelRunner` interface and factory
- FIFO and priority scheduling
- Numeric input parsing from text, CSV, or JSON arrays
- JSONL inference logs at `logs/inference_metrics.jsonl`
- Metrics summary for latency, throughput, failure rate, and memory usage
- Manual CLI parser with single and batch job modes
- CMake library/executable split
- Python helpers for ONNX export, input generation, benchmarking, and quantization
- Self-contained unit tests for scheduler, parser, logger, and metrics

## Architecture

```text
CLI -> Job Loader -> InferenceScheduler -> ModelRunnerFactory -> OnnxModelRunner
       |                    |                         |
       v                    v                         v
  RuntimeConfig       RuntimeLogger            ONNX Runtime C++ API
                            |
                            v
                    MetricsCollector
```

## Install ONNX Runtime

Download a Linux or Windows ONNX Runtime release from Microsoft, unpack it, and pass its root to CMake:

```bash
cmake -S . -B build -DONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.18.1
cmake --build build
```

For local parser/scheduler development without ONNX Runtime installed:

```bash
cmake -S . -B build -DEDGE_AI_ENABLE_ONNXRUNTIME=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```

## Python Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r python_tools/requirements.txt
python python_tools/export_model_to_onnx.py
python python_tools/generate_sample_input.py
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r python_tools\requirements.txt
python python_tools\export_model_to_onnx.py
python python_tools\generate_sample_input.py
```

## Run

Single inference:

```bash
./build/edge_ai_runtime --model models/sample_model.onnx --input sample_inputs/input_vector.txt --runner onnx
```

Batch FIFO:

```bash
./build/edge_ai_runtime --jobs configs/sample_jobs.json --schedule fifo --metrics
```

Batch priority:

```bash
./build/edge_ai_runtime --jobs configs/sample_jobs.json --schedule priority --metrics
```

Benchmark:

```bash
python python_tools/benchmark_runtime.py --exe ./build/edge_ai_runtime --jobs configs/sample_jobs.json --runs 10
```

## Example Output

```text
job_002: success, latency_ms=0.142, total_runtime_ms=3.981, output_size=3
job_001: success, latency_ms=0.118, total_runtime_ms=3.612, output_size=3
Runtime Metrics
  total_jobs: 2
  successful_jobs: 2
  failed_jobs: 0
  average_latency_ms: 0.130
  min_latency_ms: 0.118
  max_latency_ms: 0.142
  approximate_throughput_jobs_per_sec: 263.492
  failure_rate: 0.000
  memory_usage_mb: 18.734
```

## Tests

```bash
cmake -S . -B build -DEDGE_AI_ENABLE_ONNXRUNTIME=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```

## Future Improvements

- Runner cache to avoid reloading models per job
- Dynamic shared-library plugin discovery
- Execution provider selection for CUDA, TensorRT, QNN, or DirectML
- Batched inference and request coalescing
- Thread-pool scheduler with cancellation and deadlines
- p50/p95/p99 latency histograms
- OpenTelemetry export and structured config parsing
- Zero-copy input buffers and memory pool integration

## Resume Bullets

- Built a C++-based edge AI inference runtime that loads ONNX models, executes inference jobs, and logs latency, throughput, memory usage, and failure metrics.
- Designed a plugin-style model-runner interface with ONNX Runtime integration, enabling future support for QNN, SNPE, TensorRT, or LiteRT backends.
- Implemented FIFO and priority-based inference scheduling for multiple model execution tasks.
- Added a CLI for single-model inference, batch job execution, scheduling selection, and runtime metrics reporting.
- Integrated CMake-based builds, structured runtime logging, Python model-export tools, and unit tests for scheduler, metrics, logging, and tensor parsing.

## Interview Explanation

This project models the control plane around edge inference. The CLI creates one or more jobs, the scheduler determines execution order, the factory chooses a backend runner, and the ONNX runner owns model loading and inference through ONNX Runtime. Every attempt is logged as JSONL and folded into a metrics collector, so failures remain observable instead of disappearing into control flow.
