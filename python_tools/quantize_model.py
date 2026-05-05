#!/usr/bin/env python3
"""Quantize the sample ONNX model to INT8 using ONNX Runtime tooling."""

from pathlib import Path


def main() -> None:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise SystemExit(
            "ONNX Runtime quantization tools are missing. Install dependencies with:\n"
            "  pip install -r python_tools/requirements.txt\n"
            "Some versions also require: pip install onnxruntime-tools"
        ) from exc

    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "models" / "sample_model.onnx"
    output_path = repo_root / "models" / "sample_model_int8.onnx"
    if not model_path.exists():
        raise SystemExit("Missing models/sample_model.onnx. Run python_tools/export_model_to_onnx.py first.")

    quantize_dynamic(str(model_path), str(output_path), weight_type=QuantType.QInt8)
    print(f"Wrote quantized model: {output_path}")


if __name__ == "__main__":
    main()

