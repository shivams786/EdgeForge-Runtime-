# Architecture

The runtime is split into a reusable C++ library and a thin CLI executable. The library owns scheduling, model-runner interfaces, logging, metrics, tensor parsing, and ONNX Runtime integration.

## Components

- `IModelRunner`: backend interface for model loading and inference.
- `OnnxModelRunner`: concrete ONNX Runtime C++ implementation.
- `InferenceScheduler`: stores jobs and executes them using FIFO or priority policy.
- `RuntimeLogger`: appends structured JSONL records for every inference attempt.
- `MetricsCollector`: aggregates success/failure counts, latency, throughput, and memory usage.
- `TensorUtils`: reads flat numeric tensors from text, CSV, or JSON array files.
- `CliParser`: validates command-line modes and loads batch jobs.

## Execution Model

The current scheduler executes jobs synchronously. This keeps behavior deterministic and easy to debug while leaving a clear path to a thread-pool scheduler later.

## Error Handling

Each job is wrapped in exception handling. Missing models, missing inputs, unsupported runner types, parse failures, and ONNX Runtime exceptions become failed `InferenceResult` objects. The logger records both success and failure.

