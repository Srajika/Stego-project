"""Exercise dataset creation with generated images in a temporary project."""

import csv
import hashlib
import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image

from analysis.build_dataset import MANIFEST_COLUMNS, build_dataset


def _hash_files(paths: list[Path]) -> dict[Path, str]:
    return {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _seed_manifest(path: Path) -> None:
    entries = (
        ("dark", "shared.png", "dark/shared.png", "dark-source", "dark licence", "dark notes"),
        ("photos", "shared.png", "photos/shared.png", "photo-source", "photo licence", "photo notes"),
        ("photos", "z.jpg", "photos/z.jpg", "large-source", "large licence", "large notes"),
        ("dark", "removed.png", "dark/removed.png", "old-source", "old licence", "old notes"),
    )
    with path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for category, filename, original_path, source, licence, notes in entries:
            row = {column: "" for column in MANIFEST_COLUMNS}
            row.update(
                {
                    "category": category,
                    "original_filename": filename,
                    "original_path": original_path,
                    "source": source,
                    "licence": licence,
                    "notes": notes,
                }
            )
            writer.writerow(row)


def run_test() -> None:
    with tempfile.TemporaryDirectory(prefix="build-dataset-test-") as temp_dir:
        project_root = Path(temp_dir)
        raw_root = project_root / "images" / "raw"
        dark_dir = raw_root / "dark"
        photos_dir = raw_root / "photos"
        dataset_root = project_root / "images" / "dataset"
        dark_dir.mkdir(parents=True)
        photos_dir.mkdir(parents=True)

        # Two differently encoded files render to the same RGB pixels. The
        # second also checks alpha compositing onto white and duplicate ordering.
        base = np.full((48, 64, 3), 255, dtype=np.uint8)
        base[8:40, 12:52] = (25, 90, 145)
        Image.fromarray(base, mode="RGB").save(dark_dir / "shared.png")

        rgba = np.empty((48, 64, 4), dtype=np.uint8)
        rgba[:, :, :3] = (1, 2, 3)
        rgba[:, :, 3] = 0
        opaque = np.any(base != 255, axis=2)
        rgba[opaque, :3] = base[opaque]
        rgba[opaque, 3] = 255
        Image.fromarray(rgba, mode="RGBA").save(photos_dir / "shared.png")

        # EXIF orientation 6 turns this 2000x1000 source into 1000x2000 before
        # the 1600-pixel bound is applied, producing 800x1600.
        large = np.zeros((1000, 2000, 3), dtype=np.uint8)
        large[:, :, 0] = np.arange(2000, dtype=np.uint16)[None, :] % 256
        large[:, :, 1] = np.arange(1000, dtype=np.uint16)[:, None] % 256
        exif = Image.Exif()
        exif[274] = 6
        Image.fromarray(large, mode="RGB").save(
            photos_dir / "z.jpg", format="JPEG", quality=95, exif=exif
        )

        # This is an unreadable AVIF stand-in so the test needs no AVIF codec.
        (photos_dir / "unsupported.avif").write_bytes(b"not an image")
        source_paths = sorted(path for path in raw_root.rglob("*") if path.is_file())
        raw_before = _hash_files(source_paths)

        dataset_root.mkdir(parents=True)
        _seed_manifest(dataset_root / "manifest.csv")
        with redirect_stdout(io.StringIO()):
            build_dataset(project_root)

        # Simulate edits made by the dataset owner after the first run.
        manifest_path = dataset_root / "manifest.csv"
        with manifest_path.open("r", encoding="utf-8", newline="") as manifest_file:
            rows = list(csv.DictReader(manifest_file))
        for row in rows:
            if row["original_path"] == "dark/shared.png":
                row["source"] = "edited dark source"
            elif row["original_path"] == "photos/shared.png":
                row["notes"] = "edited photo notes"
        with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
            writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        # Existing generated files are overwritten in place on a rerun.
        Image.new("RGB", (64, 48), (0, 0, 0)).save(
            dataset_root / "dark" / "dark_01.png", format="PNG"
        )
        result = build_dataset(project_root)

        with manifest_path.open("r", encoding="utf-8", newline="") as manifest_file:
            manifest_rows = list(csv.DictReader(manifest_file))
        by_path = {row["original_path"]: row for row in manifest_rows}

        assert len(manifest_rows) == 5
        assert by_path["dark/shared.png"]["new_filename"] == "dark_01.png"
        assert by_path["photos/shared.png"]["new_filename"] == ""
        assert by_path["photos/shared.png"]["duplicate_of"] == "dark/dark_01.png"
        assert by_path["photos/z.jpg"]["new_filename"] == "photos_01.png"
        assert by_path["dark/shared.png"]["source"] == "edited dark source"
        assert by_path["photos/shared.png"]["source"] == "photo-source"
        assert by_path["photos/shared.png"]["notes"] == "edited photo notes"
        assert by_path["photos/shared.png"]["licence"] == "photo licence"
        assert by_path["dark/shared.png"]["notes"] == "dark notes"
        assert by_path["photos/z.jpg"]["original_size"] == "1000x2000"
        assert by_path["photos/z.jpg"]["new_size"] == "800x1600"
        assert by_path["photos/z.jpg"]["usable_for_main_studies"] == "True"
        assert by_path["dark/removed.png"]["source"] == "old-source"
        assert by_path["dark/removed.png"]["licence"] == "old licence"
        assert by_path["dark/removed.png"]["notes"] == "old notes"
        assert by_path["photos/shared.png"]["original_filename"] == "shared.png"
        assert by_path["dark/shared.png"]["original_filename"] == "shared.png"
        assert "unsupported.avif" in result["skipped"][0][0]
        assert result["totals"] == {
            "images": 2,
            "unusable": 1,
            "duplicates": 1,
            "skipped": 1,
        }

        saved = sorted(dataset_root.rglob("*.png"))
        assert [path.relative_to(dataset_root).as_posix() for path in saved] == [
            "dark/dark_01.png",
            "photos/photos_01.png",
        ]
        with Image.open(dataset_root / "dark" / "dark_01.png") as image:
            assert np.array_equal(np.asarray(image.convert("RGB")), base)
        with Image.open(dataset_root / "photos" / "photos_01.png") as image:
            assert image.size == (800, 1600)
        assert _hash_files(source_paths) == raw_before

        print(
            "Test assertions: 3 generated images; duplicate skipped without a "
            "number gap; same-name category metadata kept separate; alpha, EXIF, "
            "resize, rerun and orphan metadata preservation, and raw-file safety verified."
        )


if __name__ == "__main__":
    run_test()
    print("test_build_dataset.py passed")
