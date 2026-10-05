#!/usr/bin/env python3
"""Write each checkpoint's weights as an .npz beside it: the torch-free file the page's numpy forward loads."""

import sys
from pathlib import Path

import torch

from filters.torch_ops import export_npz


def main():
    for checkpoint in map(Path, sys.argv[1:]):
        export_npz(torch.load(checkpoint, map_location="cpu", weights_only=True), checkpoint.with_suffix(".npz"))


if __name__ == "__main__":
    main()
