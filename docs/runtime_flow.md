# Runtime Flow

```text
CLI
  |
  v
Config and Job Input
  |
  v
InferenceScheduler
  |
  v
ModelRunnerFactory
  |
  v
OnnxModelRunner
  |
  v
ONNX Runtime Session
  |
  v
InferenceResult
  |
  +--> RuntimeLogger JSONL
  |
  +--> MetricsCollector Summary
```

Single inference creates one synthetic job named `job_single`. Batch mode reads `configs/sample_jobs.json`, inserts all jobs into the scheduler, and executes according to `fifo` or `priority`.

