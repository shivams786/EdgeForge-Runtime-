#!/usr/bin/env python3
"""Generate a float vector matching the sample model input shape [1, 4]."""

from pathlib import Path

import numpy as np


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_path = repo_root / "sample_inputs" / "input_vector.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    values = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    output_path.write_text(" ".join(f"{value:.6f}" for value in values) + "\n", encoding="utf-8")
    print(f"Wrote sample input: {output_path}")


if __name__ == "__main__":
    main()

