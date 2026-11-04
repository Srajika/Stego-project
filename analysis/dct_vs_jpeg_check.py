"""Compare DCT and LSB payload survival after Pillow JPEG recompression."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from stego.common import build_payload, bytes_to_bits
from stego.dct import EMBED_QUALITY, capacity_bits, extract as dct_extract, hide as dct_hide
from stego.dct import read_majority_bits as read_dct_bits
from stego.lsb import extract as lsb_extract, hide as lsb_hide
from stego.tamper import make_tag


JPEG_QUALITIES = (100, 90, 75, 70, 50)
MESSAGE = "JPEG survival test"


def _bit_error_rate(expected: str, observed: str) -> float:
    """Return differing payload bits divided by original payload bit length."""
    differing = sum(left != right for left, right in zip(expected, observed))
    differing += abs(len(expected) - len(observed))
    return differing / len(expected)


def _save_jpeg(source_path: Path, output_path: Path, save_quality: int) -> None:
    """Use the same explicit Pillow JPEG settings for both embedding methods."""
    with Image.open(source_path) as image:
        image.convert("RGB").save(
            output_path,
            format="JPEG",
            quality=save_quality,
            subsampling=0,
            optimize=False,
            progressive=False,
        )


def main() -> None:
    """Print statuses and payload BER for DCT and LSB after JPEG saving."""
    project_root = Path(__file__).resolve().parent.parent
    cover_path = project_root / "images" / "small.png"
    if not cover_path.is_file():
        raise FileNotFoundError(f"Required test image is missing: {cover_path}")

    with Image.open(cover_path) as image:
        width, height = image.size
    payload_bits = bytes_to_bits(
        build_payload(MESSAGE, make_tag(MESSAGE.encode("utf-8"), password=None))
    )
    print(f"DCT capacity: {capacity_bits(width, height)} payload bits")
    print(f"LSB capacity: {width * height * 3} payload bits")
    print(
        "JPEG save settings: Pillow quality varies by row; subsampling=0, "
        "optimize=False, progressive=False (same settings for DCT and LSB)."
    )
    print(f"DCT extraction always uses EMBED_QUALITY={EMBED_QUALITY}; JPEG quality is never passed to extract().")
    print("Quality | DCT status | DCT BER | LSB status | LSB BER")
    print("--------|------------|---------|------------|--------")

    with tempfile.TemporaryDirectory() as temp_dir:
        scratch = Path(temp_dir)
        dct_png = scratch / "dct_stego.png"
        lsb_png = scratch / "lsb_stego.png"
        dct_hide(str(cover_path), str(dct_png), MESSAGE)
        lsb_hide(str(cover_path), str(lsb_png), MESSAGE)

        for save_quality in JPEG_QUALITIES:
            dct_jpeg = scratch / f"dct_q{save_quality}.jpg"
            lsb_jpeg = scratch / f"lsb_q{save_quality}.jpg"
            _save_jpeg(dct_png, dct_jpeg, save_quality)
            _save_jpeg(lsb_png, lsb_jpeg, save_quality)

            dct_result = dct_extract(str(dct_jpeg), quality=EMBED_QUALITY)
            dct_observed = read_dct_bits(
                str(dct_jpeg), bit_count=len(payload_bits), quality=EMBED_QUALITY
            )
            lsb_result = lsb_extract(str(lsb_jpeg))
            with Image.open(lsb_jpeg) as image:
                lsb_values = np.array(image.convert("RGB")).reshape(-1)
            lsb_observed = "".join(str(int(bit)) for bit in (lsb_values[:len(payload_bits)] & 1))

            print(
                f"{save_quality:>7} | {dct_result['status']:<10} | "
                f"{_bit_error_rate(payload_bits, dct_observed):.6f} | "
                f"{lsb_result['status']:<10} | {_bit_error_rate(payload_bits, lsb_observed):.6f}"
            )


if __name__ == "__main__":
    main()
