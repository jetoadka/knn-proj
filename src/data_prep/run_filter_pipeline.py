"""Data prep: run serial filter stages from a JSON config.

Stages:
  - quality
  - face_detector
  - low_detail

Usage:
    python -m src.data_prep.run_filter_pipeline --config configs/filter_pipeline.example.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from src.data_prep.common_dataset_ops import materialize_dataset_from_kept_manifest


def _run(cmd: list[str]):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _stage_output_dir(output_root_dir: Path, index: int, name: str) -> Path:
    return output_root_dir / f"{index:02d}_{name}"


def main():
    parser = argparse.ArgumentParser(description="Run serial filtering pipeline from config")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    input_dataset_dir = Path(cfg["input_dataset_dir"])
    output_root_dir = Path(cfg["output_root_dir"])
    splits = cfg.get("splits", ["train", "dev", "test"])
    stages = cfg.get("stages", [])

    output_root_dir.mkdir(parents=True, exist_ok=True)
    current_dataset = input_dataset_dir
    stage_reports = []

    for i, stage in enumerate(stages, start=1):
        if not stage.get("enabled", True):
            continue
        stype = stage["type"]
        sname = stage.get("name", stype)
        params = stage.get("params", {})
        out_dir = _stage_output_dir(output_root_dir, i, sname)
        tmp_dir = output_root_dir / f"{i:02d}_{sname}_tmp"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        if stype == "quality":
            tmp_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "-m",
                "src.data_prep.filter_people_gator",
                "--aligned-dir",
                str(current_dataset / "aligned_112"),
                "--output-dir",
                str(tmp_dir),
                "--splits",
                *splits,
                "--drop-ratio-per-library",
                str(params.get("drop_ratio_per_library", 0.05)),
                "--min-per-identity",
                str(params.get("min_per_identity", 1)),
            ]
            max_per_identity = int(params.get("max_per_identity", 0))
            if max_per_identity > 0:
                cmd.extend(["--max-per-identity", str(max_per_identity)])
            _run(cmd)
            materialize_dataset_from_kept_manifest(current_dataset, tmp_dir / "kept_manifest.jsonl", out_dir)

            # preserve quality stage artifacts
            meta = out_dir / "_metadata" / "quality_stage"
            meta.mkdir(parents=True, exist_ok=True)
            for name in ("kept_manifest.jsonl", "rejected_manifest.jsonl", "filter_report.json"):
                p = tmp_dir / name
                if p.exists():
                    shutil.copy2(p, meta / name)
            shutil.rmtree(tmp_dir)

        elif stype == "face_detector":
            cmd = [
                sys.executable,
                "-m",
                "src.data_prep.filter_by_face_detector",
                "--input-dataset-dir",
                str(current_dataset),
                "--output-dataset-dir",
                str(out_dir),
                "--drop-rate",
                str(params.get("drop_rate", 0.05)),
                "--splits",
                *splits,
            ]
            if params.get("copy_dropped", False):
                cmd.append("--copy-dropped")
            if "scale_factor" in params:
                cmd.extend(["--scale-factor", str(params["scale_factor"])])
            if "min_neighbors" in params:
                cmd.extend(["--min-neighbors", str(params["min_neighbors"])])
            if "min_size" in params:
                cmd.extend(["--min-size", str(params["min_size"])])
            if "min_face_confidence" in params:
                cmd.extend(["--min-face-confidence", str(params["min_face_confidence"])])
            if params.get("keep_no_face", False):
                cmd.append("--keep-no-face")
            _run(cmd)

        elif stype == "low_detail":
            cmd = [
                sys.executable,
                "-m",
                "src.data_prep.filter_by_low_detail",
                "--input-dataset-dir",
                str(current_dataset),
                "--output-dataset-dir",
                str(out_dir),
                "--drop-rate",
                str(params.get("drop_rate", 0.03)),
                "--splits",
                *splits,
            ]
            if params.get("copy_dropped", False):
                cmd.append("--copy-dropped")
            if "downscale_factor" in params:
                cmd.extend(["--downscale-factor", str(params["downscale_factor"])])
            _run(cmd)

        else:
            raise ValueError(f"Unsupported stage type: {stype}")

        current_dataset = out_dir
        stage_reports.append({"stage": sname, "type": stype, "output_dataset_dir": str(out_dir)})

    pipeline_summary = {
        "config_path": str(args.config),
        "input_dataset_dir": str(input_dataset_dir),
        "final_dataset_dir": str(current_dataset),
        "stages_executed": stage_reports,
    }
    (output_root_dir / "pipeline_summary.json").write_text(json.dumps(pipeline_summary, indent=2))
    print(f"Done. Final dataset: {current_dataset}")
    print(f"Pipeline summary: {output_root_dir / 'pipeline_summary.json'}")


if __name__ == "__main__":
    main()
