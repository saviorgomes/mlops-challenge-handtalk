from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.common import ensure_dir, read_json, utc_now_iso, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts_dir", required=True, help="Diretório base com os runs versionados")
    parser.add_argument("--run_id", required=True, help="Run aprovado que deve ser publicado")
    parser.add_argument(
        "--publish_dir",
        default="",
        help="Diretório de publicação. Se vazio, usa <artifacts_dir>/published",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    run_dir = artifacts_dir / args.run_id
    metadata_path = run_dir / "metadata.json"
    metrics_path = run_dir / "metrics.json"

    if not run_dir.exists():
        raise FileNotFoundError(f"Run não encontrado: {run_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json não encontrado para o run {args.run_id}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json não encontrado para o run {args.run_id}")

    metadata = read_json(metadata_path)
    metrics = read_json(metrics_path)

    if metadata.get("status") != "approved":
        raise ValueError(f"Run {args.run_id} não está aprovado para publicação.")

    publish_root = Path(args.publish_dir) if args.publish_dir else artifacts_dir / "published"
    published_run_dir = ensure_dir(publish_root / args.run_id)
    registry_dir = ensure_dir(publish_root / "registry")

    manifest = {
        "stage": "publish",
        "status": "published",
        "run_id": args.run_id,
        "published_at": utc_now_iso(),
        "source_run_dir": str(run_dir.as_posix()),
        "metadata_path": str(metadata_path.as_posix()),
        "metrics_path": str(metrics_path.as_posix()),
        "export_dir": metadata.get("export_dir"),
        "git_sha": metadata.get("git_sha", "unknown"),
        "metric_value": metadata.get("metric_value", metrics.get("val_token_accuracy", 0.0)),
        "threshold": metadata.get("threshold", 0.0),
    }

    write_json(published_run_dir / "published.json", manifest)
    write_json(registry_dir / "latest.json", manifest)

    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
