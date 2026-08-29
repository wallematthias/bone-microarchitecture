"""Command-line entry points for bone-microarchitecture."""

from __future__ import annotations

import argparse

from .batch import run_microarchitecture_batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bone-microarchitecture")
    commands = parser.add_subparsers(dest="command", required=True)
    batch = commands.add_parser("run-batch", help="write Microarchitecture derivatives for a dataset")
    batch.add_argument("dataset_root")
    batch.add_argument("--spacing", nargs=3, type=float, help="array-axis voxel spacing for .npy inputs")
    batch.add_argument("--no-common-region", action="store_true")
    batch.add_argument("--thickness-method", default="hildebrand")
    batch.add_argument("--thickness-backend", default="auto")
    args = parser.parse_args(argv)
    if args.command == "run-batch":
        run_microarchitecture_batch(
            args.dataset_root,
            spacing=tuple(args.spacing),
            use_common_region=not args.no_common_region,
            thickness_method=args.thickness_method,
            thickness_backend=args.thickness_backend,
        )
    return 0
