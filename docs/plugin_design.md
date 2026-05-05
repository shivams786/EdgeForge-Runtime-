# Plugin Design

The runtime is plugin-ready through `IModelRunner` and `createModelRunner`.

```cpp
class IModelRunner {
public:
    virtual ~IModelRunner() = default;
    virtual void loadModel(const std::string& model_path) = 0;
    virtual InferenceResult runInference(const std::string& job_id, const std::vector<float>& input_data) = 0;
    virtual std::string getModelName() const = 0;
    virtual bool isLoaded() const = 0;
};
```

Version 1 uses a static factory:

```cpp
std::unique_ptr<IModelRunner> createModelRunner(const std::string& runner_type);
```

To add a new runner:

1. Create a class such as `TensorRtModelRunner`, `QnnModelRunner`, `SnpeModelRunner`, or `LiteRtModelRunner`.
2. Implement `loadModel`, `runInference`, `getModelName`, and `isLoaded`.
3. Register the runner type in the factory.
4. Add CMake options for backend SDK include and library paths.
5. Add CLI/config documentation for the new `runner_type`.

Future dynamic plugin loading can move factory registration behind shared-library entry points such as:

```cpp
extern "C" IModelRunner* create_runner();
extern "C" const char* runner_type();
```

