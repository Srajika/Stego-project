"""Small end-to-end compression study check using generated temporary images."""

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.compression_study import CSV_COLUMNS, run_study


def run_test() -> None:
    with tempfile.TemporaryDirectory(prefix="compression-study-test-") as temp_dir:
        root = Path(temp_dir)
        dataset = root / "images" / "dataset"
        for category in ("generated_a", "generated_b"):
            (dataset / category).mkdir(parents=True)

        rng = np.random.default_rng(3401)
        for index, category in enumerate(("generated_a", "generated_b")):
            # 512x512 gives 1,365 DCT payload bits, above the 1,000-bit frame.
            pixels = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
            pixels[:, :, 0] = (pixels[:, :, 0] // 2 + index * 35).astype(np.uint8)
            Image.fromarray(pixels, mode="RGB").save(
                dataset / category / f"{category}_01.png", format="PNG"
            )

        manifest_path = dataset / "manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=(
                    "new_filename",
                    "category",
                    "usable_for_main_studies",
                ),
            )
            writer.writeheader()
            for category in ("generated_a", "generated_b"):
                writer.writerow(
                    {
                        "new_filename": f"{category}_01.png",
                        "category": category,
                        "usable_for_main_studies": "True",
                    }
                )

        result = run_study(root, quick=True)
        assert result["csv_path"].is_file()
        assert result["run_info_path"].is_file()
        assert all(path.is_file() for path in result["plot_paths"])
        with result["csv_path"].open("r", encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        assert len(rows) == 2 * (4 * (1 + 2 * 2) + 2 * 2)
        assert set(rows[0]) == set(CSV_COLUMNS)
        no_compression = [
            row for row in rows
            if row["record_type"] == "stego" and row["jpeg_quality"] == "none"
        ]
        cover_rows = [row for row in rows if row["record_type"] == "cover_baseline"]
        assert len(no_compression) == 2 * 4
        assert len(cover_rows) == 2 * 2 * 2
        for row in no_compression:
            assert row["exact_recovery"] in ("True", "False")
        for row in cover_rows:
            assert row["cover_reconstruction_psnr"]
            assert row["cover_reconstruction_ssim"]


if __name__ == "__main__":
    run_test()
