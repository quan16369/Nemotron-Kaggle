from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--out", default="submission.zip")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")

    files = sorted(path for path in adapter_dir.iterdir() if path.is_file())
    names = [path.name for path in files]
    if "adapter_config.json" not in names:
        raise FileNotFoundError("adapter_config.json is required for submission packaging")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=path.name)

    print(json.dumps(
        {
            "submission_zip": str(out_path),
            "file_count": len(files),
            "files": names,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
