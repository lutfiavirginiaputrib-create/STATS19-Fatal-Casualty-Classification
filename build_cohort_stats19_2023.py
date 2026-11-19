#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STATS19 2023 — Cohort reconstruction and audit
Based on the cohort-reconstruction logic in the author's notebook:
Cell [3b] "REKONSTRUKSI COHORT DARI 3 CSV MENTAH DfT".

Purpose
-------
1. Read the 2023 DfT collision, vehicle, and casualty CSV files.
2. Reconstruct a casualty-level table using LEFT JOINs:
   casualty -> collision on accident/collision index
   casualty -> vehicle on accident/collision index + vehicle_reference
3. Check duplicate keys and unmatched casualty records.
4. Export the reconstructed cohort and audit tables.
5. Optionally compare casualty keys with the existing analysis dataset
   (e.g. Dataset_Final_.csv).

Important
---------
- This script does NOT invent or select a new analytical subset.
- The full reconstructed cohort is exported. If an existing analysis file
  is supplied, the script reports key overlap/coverage; it does not infer
  why rows were selected into that subset.
- The CSV inputs must be the intended DfT 2023 files used for the study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


DEFAULT_FILES = {
    "collision": "dft-road-casualty-statistics-collision-2023.csv",
    "vehicle": "dft-road-casualty-statistics-vehicle-2023.csv",
    "casualty": "dft-road-casualty-statistics-casualty-2023.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    # Keep record identifiers as strings to avoid changing their formatting.
    dtype = {
        "collision_index": "string",
        "accident_index": "string",
        "collision_ref_no": "string",
        "accident_reference": "string",
    }
    return pd.read_csv(path, low_memory=False, dtype=dtype)


def identify_collision_id(frame: pd.DataFrame) -> str:
    if "collision_index" in frame.columns:
        return "collision_index"
    if "accident_index" in frame.columns:
        return "accident_index"
    raise KeyError(
        "Tidak menemukan kolom collision_index atau accident_index "
        "pada file collision."
    )


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Kolom wajib tidak ditemukan pada {label}: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rekonstruksi dan audit cohort casualty STATS19 2023."
    )
    parser.add_argument("--data-dir", required=True,
                        help="Folder yang berisi tiga CSV mentah DfT.")
    parser.add_argument("--output-dir", default="cohort_output",
                        help="Folder keluaran (default: cohort_output).")
    parser.add_argument("--analysis-file", default=None,
                        help="Opsional: file Dataset_Final_ yang ingin dibandingkan.")
    parser.add_argument("--collision-file", default=DEFAULT_FILES["collision"])
    parser.add_argument("--vehicle-file", default=DEFAULT_FILES["vehicle"])
    parser.add_argument("--casualty-file", default=DEFAULT_FILES["casualty"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "collision": data_dir / args.collision_file,
        "vehicle": data_dir / args.vehicle_file,
        "casualty": data_dir / args.casualty_file,
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"File {name} tidak ditemukan: {path}")

    print("Membaca tiga file mentah DfT...")
    collision = read_csv(paths["collision"])
    vehicle = read_csv(paths["vehicle"])
    casualty = read_csv(paths["casualty"])

    collision_id = identify_collision_id(collision)
    vehicle_id = identify_collision_id(vehicle)
    casualty_id = identify_collision_id(casualty)
    if not (collision_id == vehicle_id == casualty_id):
        raise ValueError(
            "Nama kolom ID kecelakaan berbeda antarfile: "
            f"collision={collision_id}, vehicle={vehicle_id}, casualty={casualty_id}"
        )

    require_columns(vehicle, [collision_id, "vehicle_reference"], "vehicle")
    require_columns(casualty, [collision_id, "vehicle_reference",
                               "casualty_reference", "casualty_severity"], "casualty")
    require_columns(collision, [collision_id], "collision")

    # File provenance: actual file metadata and checksum at execution time.
    provenance = []
    for name, path in paths.items():
        stat = path.stat()
        provenance.append({
            "table": name,
            "file": path.name,
            "size_bytes": stat.st_size,
            "modified_time_local": datetime.fromtimestamp(
                stat.st_mtime
            ).isoformat(timespec="seconds"),
            "sha256": sha256_file(path),
        })
    pd.DataFrame(provenance).to_csv(
        output_dir / "A0_raw_file_provenance.csv", index=False
    )

    # Duplicate-key and unmatched-record checks, following the notebook.
    vehicle_keys = vehicle[[collision_id, "vehicle_reference"]].drop_duplicates()
    collision_keys = collision[[collision_id]].drop_duplicates()

    casualty_vehicle_check = casualty.merge(
        vehicle_keys,
        on=[collision_id, "vehicle_reference"],
        how="left",
        indicator=True,
    )
    casualty_collision_check = casualty.merge(
        collision_keys,
        on=collision_id,
        how="left",
        indicator=True,
    )

    # Reconstruct casualty-level table using the same LEFT JOIN sequence
    # as notebook Cell [3b].
    reconstructed = (
        casualty.merge(
            collision,
            on=collision_id,
            how="left",
            suffixes=("", "_col"),
        )
        .merge(
            vehicle,
            on=[collision_id, "vehicle_reference"],
            how="left",
            suffixes=("", "_veh"),
        )
    )

    if len(reconstructed) != len(casualty):
        raise AssertionError(
            "Jumlah baris berubah setelah LEFT JOIN. Periksa duplikasi key "
            "pada tabel collision/vehicle sebelum memakai hasil."
        )

    flow = [
        {"stage": "Raw collision table", "n_rows": len(collision),
         "n_collisions": collision[collision_id].nunique()},
        {"stage": "Raw vehicle table", "n_rows": len(vehicle),
         "n_collisions": vehicle[collision_id].nunique()},
        {"stage": "Raw casualty table", "n_rows": len(casualty),
         "n_collisions": casualty[collision_id].nunique()},
        {"stage": "Duplicate casualty keys",
         "n_rows": int(casualty.duplicated(
             [collision_id, "casualty_reference"]
         ).sum()), "n_collisions": None},
        {"stage": "Duplicate vehicle keys",
         "n_rows": int(vehicle.duplicated(
             [collision_id, "vehicle_reference"]
         ).sum()), "n_collisions": None},
        {"stage": "Casualties without matching collision",
         "n_rows": int((casualty_collision_check["_merge"] == "left_only").sum()),
         "n_collisions": None},
        {"stage": "Casualties without matching vehicle",
         "n_rows": int((casualty_vehicle_check["_merge"] == "left_only").sum()),
         "n_collisions": None},
        {"stage": "Casualty-level table after left joins (full)",
         "n_rows": len(reconstructed),
         "n_collisions": reconstructed[collision_id].nunique()},
        {"stage": "Fatal by standard casualty_severity (full)",
         "n_rows": int((pd.to_numeric(
             casualty["casualty_severity"], errors="coerce"
         ) == 1).sum()), "n_collisions": None},
    ]
    pd.DataFrame(flow).to_csv(
        output_dir / "A4_cohort_flow_from_raw.csv", index=False
    )

    # Schema comparison: normalize known DfT field-name variants as in notebook.
    rename_map = {
        "collision_index": "accident_index",
        "collision_ref_no": "accident_reference",
        "collision_year": "accident_year",
    }
    raw_columns = {rename_map.get(column, column)
                   for column in reconstructed.columns}
    schema_rows = [
        {
            "item": "Kolom di file analisis, tidak ada di data mentah",
            "n": None,
            "columns": "",
        },
        {
            "item": "Kolom di data mentah, tidak ada di file analisis",
            "n": len(raw_columns),
            "columns": ", ".join(sorted(raw_columns)),
        },
    ]

    # Optional comparison with the existing analysis file.
    if args.analysis_file:
        analysis_path = Path(args.analysis_file).expanduser().resolve()
        if not analysis_path.is_file():
            raise FileNotFoundError(
                f"Analysis file tidak ditemukan: {analysis_path}"
            )
        analysis = read_csv(analysis_path)
        analysis_id = identify_collision_id(analysis)
        if analysis_id != "accident_index" and collision_id != analysis_id:
            if collision_id == "collision_index" and "accident_index" in analysis.columns:
                pass
            else:
                raise ValueError(
                    "Kolom ID kecelakaan pada analysis file tidak dapat "
                    "dicocokkan dengan file mentah."
                )
        require_columns(analysis, ["casualty_reference"], "analysis file")

        raw_id_values = reconstructed[collision_id].astype("string")
        analysis_id_col = (
            "accident_index" if "accident_index" in analysis.columns
            else analysis_id
        )
        raw_keys = set(zip(
            raw_id_values.fillna(""),
            reconstructed["casualty_reference"].astype("string").fillna("")
        ))
        analysis_keys = set(zip(
            analysis[analysis_id_col].astype("string").fillna(""),
            analysis["casualty_reference"].astype("string").fillna("")
        ))
        intersection = raw_keys & analysis_keys

        match = [
            {"item": "Kunci casualty di file analisis", "n": len(analysis_keys)},
            {"item": "Kunci casualty di data mentah hasil join", "n": len(raw_keys)},
            {"item": "Kunci yang sama (irisan)", "n": len(intersection)},
            {"item": "Kunci file analisis yang tidak ada di data mentah",
             "n": len(analysis_keys - raw_keys)},
            {"item": "Cakupan file analisis terhadap data mentah (%)",
             "n": (100 * len(intersection) / len(raw_keys)) if raw_keys else None},
        ]
        pd.DataFrame(match).to_csv(
            output_dir / "A6_key_match_analysis_vs_raw.csv", index=False
        )

        reconstructed["in_subset"] = [
            key in analysis_keys for key in zip(
                raw_id_values.fillna(""),
                reconstructed["casualty_reference"].astype("string").fillna("")
            )
        ]
        coverage_tables = []
        for column, dimension in [
            ("police_force", "police_force"),
            ("casualty_severity", "casualty_severity (1=fatal)"),
            ("casualty_class", "casualty_class"),
            ("number_of_vehicles", "number_of_vehicles"),
        ]:
            if column in reconstructed.columns:
                part = (
                    reconstructed.groupby(column, dropna=False)
                    .agg(raw_n=("in_subset", "size"),
                         in_subset=("in_subset", "sum"))
                    .reset_index()
                    .rename(columns={column: "value"})
                )
                part["coverage_pct"] = (
                    100 * part["in_subset"] / part["raw_n"]
                )
                part.insert(0, "dimension", dimension)
                coverage_tables.append(part)
        if coverage_tables:
            pd.concat(coverage_tables, ignore_index=True).to_csv(
                output_dir / "A7_coverage_by_dimension.csv", index=False
            )

        schema_rows[0] = {
            "item": "Kolom di file analisis, tidak ada di data mentah",
            "n": len(set(analysis.columns) - {"row_id"} - raw_columns),
            "columns": ", ".join(sorted(
                (set(analysis.columns) - {"row_id"}) - raw_columns
            )),
        }
        schema_rows[1] = {
            "item": "Kolom di data mentah, tidak ada di file analisis",
            "n": len(raw_columns - (set(analysis.columns) - {"row_id"})),
            "columns": ", ".join(sorted(
                raw_columns - (set(analysis.columns) - {"row_id"})
            )),
        }
    pd.DataFrame(schema_rows).to_csv(
        output_dir / "A5_schema_raw_vs_analysis.csv", index=False
    )

    # Save reconstructed full casualty-level table.
    output_cohort = output_dir / "STATS19_2023_reconstructed_casualty_cohort.csv"
    reconstructed.to_csv(output_cohort, index=False)

    summary = {
        "input_files": {name: str(path) for name, path in paths.items()},
        "collision_id_column": collision_id,
        "collision_rows": len(collision),
        "vehicle_rows": len(vehicle),
        "casualty_rows": len(casualty),
        "reconstructed_rows": len(reconstructed),
        "reconstructed_unique_collisions": int(reconstructed[collision_id].nunique()),
        "output_cohort": str(output_cohort),
        "output_directory": str(output_dir),
        "analysis_file_compared": args.analysis_file,
        "note": (
            "Full casualty-level left-join cohort. This output is not "
            "automatically equivalent to the analytical subset in the paper."
        ),
    }
    (output_dir / "cohort_run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSelesai.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nPeriksa file A4_cohort_flow_from_raw.csv dan A6/A7 bila tersedia.")
    print("Jangan menyatakan hasil ini mereproduksi subset analisis paper "
          "sebelum aturan pemilihan subset berhasil diverifikasi.")


if __name__ == "__main__":
    main()
