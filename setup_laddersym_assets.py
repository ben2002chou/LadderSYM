#!/usr/bin/env python3
"""Download LadderSym datasets/checkpoints and write a ready-to-source env file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATASETS = {
    "coco": {
        "repo_id": "ben2002chou/CocoChorales-E",
        "folder_name": "CocoChorales-E",
        "split_json": "split.json",
        "root_env": "LADDERSYM_COCO_ROOT",
        "split_env": "LADDERSYM_COCO_SPLIT_JSON",
    },
    "maestro": {
        "repo_id": "ben2002chou/MAESTRO-E",
        "folder_name": "MAESTRO-E",
        "split_json": "maestro-v3.0.0.json",
        "root_env": "LADDERSYM_MAESTRO_ROOT",
        "split_env": "LADDERSYM_MAESTRO_SPLIT_JSON",
    },
}

CHECKPOINTS = {
    "coco_prompted": "checkpoints/cocochorales/prompted/model.ckpt",
    "coco_unprompted": "checkpoints/cocochorales/unprompted/model.ckpt",
    "maestro_prompted": "checkpoints/maestro/prompted/model.ckpt",
    "maestro_unprompted": "checkpoints/maestro/unprompted/model.ckpt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set up LadderSym datasets and checkpoints from Hugging Face Hub."
    )
    parser.add_argument(
        "--datasets",
        choices=["all", "coco", "maestro", "none"],
        default="all",
        help="Which datasets to download.",
    )
    parser.add_argument(
        "--checkpoints",
        choices=["all", "prompted", "unprompted", "none"],
        default="all",
        help="Which official checkpoints to download.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Local parent directory for datasets.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="pretrained/laddersym",
        help="Local parent directory for checkpoints.",
    )
    parser.add_argument(
        "--env-file",
        default=".env.laddersym",
        help="Path to write export statements for dataset paths.",
    )
    parser.add_argument(
        "--skip-env-file",
        action="store_true",
        help="Do not write an env file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only; do not download or write files.",
    )
    return parser.parse_args()


def selected_datasets(selection: str) -> list[str]:
    if selection == "none":
        return []
    if selection == "all":
        return ["coco", "maestro"]
    return [selection]


def selected_checkpoints(selection: str) -> list[str]:
    if selection == "none":
        return []
    if selection == "all":
        return list(CHECKPOINTS.keys())
    if selection == "prompted":
        return ["coco_prompted", "maestro_prompted"]
    return ["coco_unprompted", "maestro_unprompted"]


def write_env_file(env_path: Path, lines: list[str], dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] Would write env file: {env_path}")
        for line in lines:
            print(line)
        return
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote env file: {env_path}")


def main() -> int:
    args = parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    env_path = Path(args.env_file).expanduser().resolve()

    dataset_keys = selected_datasets(args.datasets)
    checkpoint_keys = selected_checkpoints(args.checkpoints)

    failures: list[str] = []
    env_lines: list[str] = []
    snapshot_download_fn = None
    hf_hub_download_fn = None

    if not args.dry_run and (dataset_keys or checkpoint_keys):
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError:
            print(
                "huggingface_hub is required for downloads. "
                "Install dependencies with: pip install -r requirements.txt",
                file=sys.stderr,
            )
            return 2
        snapshot_download_fn = snapshot_download
        hf_hub_download_fn = hf_hub_download

    for key in dataset_keys:
        cfg = DATASETS[key]
        local_dir = data_dir / cfg["folder_name"]
        print(f"Dataset [{key}]: {cfg['repo_id']} -> {local_dir}")
        if not args.dry_run:
            try:
                snapshot_download_fn(
                    repo_id=cfg["repo_id"],
                    repo_type="dataset",
                    local_dir=str(local_dir),
                )
            except Exception as exc:  # pragma: no cover
                failures.append(f"dataset:{key}: {exc}")
                continue
        env_lines.append(f'export {cfg["root_env"]}="{local_dir}"')
        env_lines.append(
            f'export {cfg["split_env"]}="${cfg["root_env"]}/{cfg["split_json"]}"'
        )
        env_lines.append("")

    for key in checkpoint_keys:
        filename = CHECKPOINTS[key]
        print(f"Checkpoint [{key}]: {filename} -> {checkpoint_dir}")
        if args.dry_run:
            continue
        try:
            hf_hub_download_fn(
                repo_id="ben2002chou/laddersym-checkpoints",
                filename=filename,
                local_dir=str(checkpoint_dir),
            )
        except Exception as exc:  # pragma: no cover
            failures.append(f"checkpoint:{key}: {exc}")

    coco_prompted = checkpoint_dir / CHECKPOINTS["coco_prompted"]
    if checkpoint_keys and coco_prompted.exists():
        env_lines.append(
            'export LADDERSYM_MAESTRO_PRETRAINED_CKPT="'
            f'{coco_prompted}"'
        )
        env_lines.append("")

    if not args.skip_env_file and env_lines:
        write_env_file(env_path, env_lines, args.dry_run)
        print(f"Next step: source {env_path}")

    if failures:
        print("\nSetup completed with errors:", file=sys.stderr)
        for msg in failures:
            print(f"- {msg}", file=sys.stderr)
        return 1

    print("\nSetup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
