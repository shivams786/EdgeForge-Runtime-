#!/usr/bin/env python3
"""Export a tiny PyTorch feed-forward model to ONNX."""

from pathlib import Path

import torch


class TinyEdgeClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 3),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_path = repo_root / "models" / "sample_model.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(7)
    model = TinyEdgeClassifier().eval()
    dummy_input = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["probabilities"],
        dynamic_axes={"input": {0: "batch"}, "probabilities": {0: "batch"}},
    )
    print(f"Exported ONNX model: {output_path}")


if __name__ == "__main__":
    main()

