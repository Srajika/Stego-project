"""Build a normalized, deduplicated image dataset from ``images/raw``.

Pass the project root to :func:`build_dataset`. All generated PNGs and the
manifest are confined to ``<root>/images/dataset``.
"""

import csv
import hashlib
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageOps


MAX_SIDE = 1600
MIN_DCT_CAPACITY_BITS = 1500
MANIFEST_COLUMNS = (
    "new_filename",
    "category",
    "original_filename",
    "original_path",
    "original_format",
    "original_size",
    "new_size",
    "lsb_capacity_bits",
    "dct_capacity_bits",
    "usable_for_main_studies",
    "duplicate_of",
    "source",
    "licence",
    "notes",
)
METADATA_COLUMNS = ("source", "licence", "notes")


def _resolved(path: Path) -> Path:
    return path.resolve(strict=False)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _assert_dataset_path(path: Path, dataset_root: Path) -> Path:
    """Resolve and assert that a path stays inside the dataset directory."""
    resolved_path = _resolved(path)
    assert _is_within(resolved_path, dataset_root), (
        f"Refusing to write outside images/dataset/: {resolved_path}"
    )
    return resolved_path


def _normalize_relative_path(value: str) -> str:
    """Normalize manifest paths to POSIX separators for stable matching."""
    return PurePosixPath(value.replace("\\", "/")).as_posix()


def _load_existing_metadata(
    manifest_path: Path, dataset_root: Path
) -> tuple[dict[str, dict[str, str]], set[str], dict[str, dict[str, str]]]:
    """Read metadata and names of PNG outputs previously owned by this script."""
    _assert_dataset_path(manifest_path, dataset_root)
    assert not manifest_path.is_symlink(), (
        f"Refusing to read a manifest symlink: {manifest_path}"
    )
    metadata_by_original_path: dict[str, dict[str, str]] = {}
    owned_outputs: set[str] = set()
    existing_rows_by_original_path: dict[str, dict[str, str]] = {}
    if not manifest_path.exists():
        return metadata_by_original_path, owned_outputs, existing_rows_by_original_path
    if not manifest_path.is_file():
        raise IsADirectoryError(f"Manifest is not a file: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        for row in csv.DictReader(manifest_file):
            category = row.get("category", "") or ""
            filename = row.get("original_filename", "") or ""
            original_path = row.get("original_path", "") or ""
            key = _normalize_relative_path(
                original_path if original_path else f"{category}/{filename}"
            )
            existing_row = {
                column: row.get(column, "") or "" for column in MANIFEST_COLUMNS
            }
            existing_row["original_path"] = key
            existing_rows_by_original_path[key] = existing_row
            metadata_by_original_path[key] = {
                column: row.get(column, "") or "" for column in METADATA_COLUMNS
            }

            new_filename = (row.get("new_filename", "") or "").replace("\\", "/")
            if new_filename:
                relative_output = (
                    new_filename
                    if "/" in new_filename
                    else f"{category}/{new_filename}"
                )
                owned_outputs.add(_normalize_relative_path(relative_output))

    return metadata_by_original_path, owned_outputs, existing_rows_by_original_path


def _metadata_for(
    original_path: str, metadata_by_original_path: dict[str, dict[str, str]]
) -> dict[str, str]:
    return metadata_by_original_path.get(
        original_path, {column: "" for column in METADATA_COLUMNS}
    )


def _open_as_rgb(path: Path) -> tuple[Image.Image, str, tuple[int, int]]:
    """Apply EXIF orientation and white alpha compositing before RGB conversion."""
    with Image.open(path) as opened:
        original_format = opened.format or ""
        has_transparency = "A" in opened.getbands() or "transparency" in opened.info
        oriented = ImageOps.exif_transpose(opened)
        oriented.load()
        original_size = oriented.size

        if has_transparency:
            rgba = oriented.convert("RGBA")
            white_background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(white_background, rgba).convert("RGB")
        else:
            rgb = oriented.convert("RGB")

    return rgb, original_format, original_size


def _save_png_inside_dataset(
    image: Image.Image,
    target_path: Path,
    dataset_root: Path,
    owned_outputs: set[str],
    relative_output: str,
) -> None:
    """Atomically save a PNG without following or replacing an unowned path."""
    _assert_dataset_path(target_path, dataset_root)
    assert not target_path.is_symlink(), (
        f"Refusing to write through a symlink: {target_path}"
    )
    if target_path.exists() and relative_output not in owned_outputs:
        raise FileExistsError(
            f"Refusing to overwrite a PNG not listed in the existing manifest: "
            f"{target_path}"
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    _assert_dataset_path(target_path.parent, dataset_root)
    assert not target_path.parent.is_symlink(), (
        f"Refusing to write through a category symlink: {target_path.parent}"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target_path.stem}.", suffix=".tmp", dir=target_path.parent
    )
    temporary_path = Path(temporary_name)
    _assert_dataset_path(temporary_path, dataset_root)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            image.save(temporary_file, format="PNG")
        os.replace(temporary_path, target_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_manifest(
    manifest_path: Path, rows: list[dict[str, str]], dataset_root: Path
) -> None:
    """Atomically write the refreshed manifest inside the dataset directory."""
    _assert_dataset_path(manifest_path, dataset_root)
    assert not manifest_path.is_symlink(), (
        f"Refusing to replace a manifest symlink: {manifest_path}"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".manifest.", suffix=".tmp", dir=dataset_root
    )
    temporary_path = Path(temporary_name)
    _assert_dataset_path(temporary_path, dataset_root)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _print_manifest_preview(manifest_path: Path) -> None:
    with manifest_path.open("r", encoding="utf-8", newline="") as manifest_file:
        preview_rows = []
        for index, row in enumerate(csv.reader(manifest_file)):
            if index >= 11:  # Header plus at most 10 data rows.
                break
            preview_rows.append(row)

    from io import StringIO

    preview = StringIO(newline="")
    csv.writer(preview, lineterminator="\n").writerows(preview_rows)
    print("\nFirst 10 manifest rows (including header):")
    print(preview.getvalue().rstrip())


def build_dataset(root: str | Path) -> dict[str, Any]:
    """Create the normalized dataset under ``root/images/dataset``.

    Categories are immediate subdirectory names under ``images/raw``. Files
    are processed in sorted category and filename order. Duplicate images are
    identified from SHA-256 hashes of their EXIF-corrected, white-composited
    RGB pixel bytes. The returned summary also makes temporary-root testing
    straightforward.
    """
    project_root = Path(root).resolve()
    images_root = project_root / "images"
    raw_root = images_root / "raw"
    dataset_root = images_root / "dataset"

    # Reject symlinked roots so the lexical images/dataset path is also the
    # resolved destination and every write remains inside this exact tree.
    assert _resolved(images_root) == images_root, (
        f"Refusing a symlinked images directory: {images_root}"
    )
    assert _resolved(raw_root) == raw_root, (
        f"Refusing a symlinked raw directory: {raw_root}"
    )
    assert _resolved(dataset_root) == dataset_root, (
        f"Refusing a symlinked dataset directory: {dataset_root}"
    )
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw image directory does not exist: {raw_root}")

    dataset_root.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_root / "manifest.csv"
    (
        metadata_by_original_path,
        owned_outputs,
        existing_rows_by_original_path,
    ) = _load_existing_metadata(manifest_path, dataset_root)

    category_dirs = sorted(
        (path for path in raw_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    for category_dir in category_dirs:
        assert _resolved(category_dir) == category_dir, (
            f"Refusing a symlinked category directory: {category_dir}"
        )

    category_names = [path.name for path in category_dirs]
    summary: dict[str, dict[str, int]] = {
        category: {"images": 0, "unusable": 0, "duplicates": 0, "skipped": 0}
        for category in category_names
    }
    rows: list[dict[str, str]] = []
    skipped: list[tuple[str, str]] = []
    first_by_hash: dict[str, dict[str, str]] = {}
    counters = {category: 0 for category in category_names}

    input_files = [
        (category_dir.name, path)
        for category_dir in category_dirs
        for path in sorted(category_dir.iterdir(), key=lambda item: item.name)
        if path.is_file()
    ]

    for category, source_path in input_files:
        original_path = source_path.relative_to(raw_root).as_posix()
        display_path = f"images/raw/{original_path}"
        metadata = _metadata_for(original_path, metadata_by_original_path)
        source_resolved = _resolved(source_path)
        if not _is_within(source_resolved, raw_root):
            reason = "source path resolves outside images/raw/"
            skipped.append((display_path, reason))
            summary[category]["skipped"] += 1
            rows.append(
                {
                    "new_filename": "",
                    "category": category,
                    "original_filename": source_path.name,
                    "original_path": original_path,
                    "original_format": "",
                    "original_size": "",
                    "new_size": "",
                    "lsb_capacity_bits": "",
                    "dct_capacity_bits": "",
                    "usable_for_main_studies": "False",
                    "duplicate_of": "",
                    **metadata,
                }
            )
            continue

        try:
            rgb, original_format, original_size = _open_as_rgb(source_path)
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            skipped.append((display_path, reason))
            summary[category]["skipped"] += 1
            rows.append(
                {
                    "new_filename": "",
                    "category": category,
                    "original_filename": source_path.name,
                    "original_path": original_path,
                    "original_format": "",
                    "original_size": "",
                    "new_size": "",
                    "lsb_capacity_bits": "",
                    "dct_capacity_bits": "",
                    "usable_for_main_studies": "False",
                    "duplicate_of": "",
                    **metadata,
                }
            )
            continue

        pixel_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
        output_image = rgb.copy()
        if max(output_image.size) > MAX_SIDE:
            output_image.thumbnail(
                (MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS
            )

        width, height = output_image.size
        lsb_capacity = width * height * 3
        dct_capacity = (width // 8) * (height // 8) // 3
        usable = dct_capacity >= MIN_DCT_CAPACITY_BITS
        original_size_text = f"{original_size[0]}x{original_size[1]}"
        new_size_text = f"{width}x{height}"

        if pixel_hash in first_by_hash:
            first = first_by_hash[pixel_hash]
            summary[category]["duplicates"] += 1
            print(
                f"DUPLICATE: {display_path} matches "
                f"images/raw/{first['original_path']}; "
                f"skipped (kept {first['dataset_path']})."
            )
            rows.append(
                {
                    "new_filename": "",
                    "category": category,
                    "original_filename": source_path.name,
                    "original_path": original_path,
                    "original_format": original_format,
                    "original_size": original_size_text,
                    "new_size": new_size_text,
                    "lsb_capacity_bits": str(lsb_capacity),
                    "dct_capacity_bits": str(dct_capacity),
                    "usable_for_main_studies": str(usable),
                    "duplicate_of": first["dataset_path"],
                    **metadata,
                }
            )
            continue

        counters[category] += 1
        new_filename = f"{category}_{counters[category]:02d}.png"
        relative_output = f"{category}/{new_filename}"
        target_path = dataset_root / category / new_filename
        _save_png_inside_dataset(
            output_image,
            target_path,
            dataset_root,
            owned_outputs,
            relative_output,
        )
        first_by_hash[pixel_hash] = {
            "original_path": original_path,
            "dataset_path": relative_output,
        }
        summary[category]["images"] += 1
        if not usable:
            summary[category]["unusable"] += 1
        owned_outputs.add(relative_output)
        rows.append(
            {
                "new_filename": new_filename,
                "category": category,
                "original_filename": source_path.name,
                "original_path": original_path,
                "original_format": original_format,
                "original_size": original_size_text,
                "new_size": new_size_text,
                "lsb_capacity_bits": str(lsb_capacity),
                "dct_capacity_bits": str(dct_capacity),
                "usable_for_main_studies": str(usable),
                "duplicate_of": "",
                **metadata,
            }
        )

    current_original_paths = {row["original_path"] for row in rows}
    orphan_rows = [
        existing_rows_by_original_path[original_path]
        for original_path in sorted(existing_rows_by_original_path)
        if original_path not in current_original_paths
    ]
    rows.extend(orphan_rows)
    _write_manifest(manifest_path, rows, dataset_root)

    print("\nSummary by category")
    print("category | images saved | unusable | duplicates | skipped")
    total = {"images": 0, "unusable": 0, "duplicates": 0, "skipped": 0}
    for category in category_names:
        counts = summary[category]
        print(
            f"{category} | {counts['images']} | {counts['unusable']} | "
            f"{counts['duplicates']} | {counts['skipped']}"
        )
        for key in total:
            total[key] += counts[key]
    print(
        f"TOTAL | {total['images']} | {total['unusable']} | "
        f"{total['duplicates']} | {total['skipped']}"
    )
    print(f"Duplicates found: {total['duplicates']}")
    if orphan_rows:
        print(f"Prior manifest rows preserved for missing raw files: {len(orphan_rows)}")

    print("\nSkipped files:")
    if skipped:
        for display_path, reason in skipped:
            print(f"{display_path}: {reason}")
    else:
        print("None")

    print(f"\nManifest: images/dataset/manifest.csv ({len(rows)} rows)")
    _print_manifest_preview(manifest_path)
    return {"summary": summary, "totals": total, "skipped": skipped, "rows": rows}


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    build_dataset(project_root)


if __name__ == "__main__":
    main()
