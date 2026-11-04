"""Assert-style checks and actual p-value report for the LSB detector."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from stego.common import STOP_MARKER, TAG_BYTES
from stego.dct import hide as dct_hide
from stego.detect import detect, detect_prefix_curve, detect_sequential
from stego.lsb import hide as lsb_hide
from stego.tamper import make_tag


def _report(label: str, result: dict) -> None:
    """Print the observed p-value and decision without tuning expectations."""
    print(
        f"{label}: p_value={result['p_value']:.12g}, "
        f"statistic={result['statistic']:.12g}, "
        f"pairs_used={result['pairs_used']}, "
        f"pairs_skipped={result['pairs_skipped']}, "
        f"degrees_of_freedom={result['degrees_of_freedom']}, "
        f"verdict={result['verdict']}"
    )


def _assert_result(result: dict) -> None:
    """Check the detector's documented result fields and numeric values."""
    assert set(("statistic", "p_value", "pairs_used", "pairs_skipped", "degrees_of_freedom", "verdict", "score")) <= set(result)
    assert result["statistic"] >= 0.0
    assert np.isfinite(result["p_value"])
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["pairs_used"] + result["pairs_skipped"] == 128
    assert result["degrees_of_freedom"] == result["pairs_used"] - 1
    assert result["score"] == result["p_value"] * 100.0


def _calibration_regression(scratch: Path, cover_path: Path) -> None:
    """Check the full-Pearson mean is close to the mean degrees of freedom."""
    crop_sizes = [(120, 100), (256, 192), (512, 384), (800, 600)]
    crop_rng = np.random.default_rng(8675309)
    statistics = []
    degrees_of_freedom = []

    with Image.open(cover_path) as source:
        cover = source.convert("RGB")
    image_width, image_height = cover.size

    for index in range(30):
        width, height = crop_sizes[index % len(crop_sizes)]
        left = int(crop_rng.integers(0, image_width - width + 1))
        top = int(crop_rng.integers(0, image_height - height + 1))
        crop = np.asarray(
            cover.crop((left, top, left + width, top + height)), dtype=np.uint8
        )
        flat = crop.reshape(-1)
        random_bits = np.random.default_rng(20261104 + index).integers(
            0, 2, size=flat.size, dtype=np.uint8
        )
        randomized = (flat & np.uint8(254)) | random_bits
        image_path = scratch / f"calibration_{index:02d}.png"
        Image.fromarray(randomized.reshape(crop.shape)).save(image_path)
        result = detect(image_path)
        statistics.append(result["statistic"])
        degrees_of_freedom.append(result["degrees_of_freedom"])

    mean_statistic = float(np.mean(statistics))
    mean_dof = float(np.mean(degrees_of_freedom))
    ratio = mean_statistic / mean_dof
    print(
        f"30-crop calibration regression: mean_statistic={mean_statistic:.6f}, "
        f"mean_dof={mean_dof:.6f}, ratio={ratio:.6f}"
    )
    assert 0.9 < ratio < 1.1


def run_tests() -> None:
    """Report p-values for clean, randomized, LSB, and DCT images."""
    project_root = Path(__file__).resolve().parent.parent
    cover_path = project_root / "images" / "small.png"
    assert cover_path.is_file(), f"Required test image is missing: {cover_path}"

    with tempfile.TemporaryDirectory() as temp_dir:
        scratch = Path(temp_dir)

        clean_result = detect(cover_path)
        _assert_result(clean_result)
        _report("clean images/small.png", clean_result)

        # Randomize each channel's LSB with a fixed seed, preserving its upper bits.
        with Image.open(cover_path) as image:
            randomized = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        flat = randomized.reshape(-1)
        random_bits = np.random.default_rng(20261104).integers(
            0, 2, size=flat.size, dtype=np.uint8
        )
        randomized_flat = (flat & np.uint8(254)) | random_bits
        randomized_path = scratch / "random_lsb.png"
        Image.fromarray(randomized_flat.reshape(randomized.shape)).save(randomized_path)
        random_result = detect(randomized_path)
        _assert_result(random_result)
        assert random_result["p_value"] > 0.01
        _report("randomized LSBs (seed=20261104)", random_result)

        plain_path = scratch / "plain_lsb.png"
        lsb_hide(str(cover_path), str(plain_path), "short message")
        plain_result = detect(plain_path)
        _assert_result(plain_result)
        _report("plain-mode LSB (short message)", plain_result)

        password_path = scratch / "password_lsb.png"
        lsb_hide(
            str(cover_path),
            str(password_path),
            "short message",
            password="detector test password",
        )
        password_result = detect(password_path)
        _assert_result(password_result)
        _report("password-mode LSB (short message)", password_result)

        dct_path = scratch / "dct_stego.png"
        dct_hide(str(cover_path), str(dct_path), "short message")
        dct_result = detect(dct_path)
        _assert_result(dct_result)
        _report("DCT stego (short message)", dct_result)

        curve = detect_prefix_curve(plain_path)
        assert len(curve["curve"]) == 19
        assert curve["curve"][0]["values_used"] >= 1000
        assert curve["curve"][-1]["values_used"] == curve["values_total"]
        assert curve["curve"][-1]["p_value"] == plain_result["p_value"]

        clean_sequential = detect_sequential(cover_path)
        assert clean_sequential["high_prefix_count"] == 0
        assert clean_sequential["estimated_payload_end_rate"] is None
        print(
            "clean sequential: "
            f"high_prefix_count={clean_sequential['high_prefix_count']}, "
            f"verdict={clean_sequential['verdict']}"
        )

        # Match the analysis script's 50% uniformly random byte payload in plain mode.
        with Image.open(cover_path) as image:
            uniform_pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        uniform_flat = uniform_pixels.reshape(-1)
        target_payload_bytes = len(uniform_flat) * 50 // 800
        random_message = np.random.default_rng(20261104 + 50).bytes(
            target_payload_bytes - len(STOP_MARKER) - TAG_BYTES
        )
        tag = make_tag(random_message, None, None)
        payload = random_message + STOP_MARKER + tag.encode("ascii")
        payload_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        uniform_flat[:len(payload_bits)] = (
            uniform_flat[:len(payload_bits)] & np.uint8(254)
        ) | payload_bits
        uniform_path = scratch / "uniform_bytes_plain_50.png"
        Image.fromarray(uniform_flat.reshape(uniform_pixels.shape)).save(uniform_path)
        uniform_sequential = detect_sequential(uniform_path)
        assert uniform_sequential["high_prefix_count"] > 0
        assert uniform_sequential["estimated_payload_end_rate"] is not None
        assert 45.0 <= uniform_sequential["estimated_payload_end_rate"] <= 55.0
        print(
            "plain uniform-byte 50% sequential: "
            f"high_prefix_count={uniform_sequential['high_prefix_count']}, "
            f"longest_high_prefix_rate={uniform_sequential['longest_high_prefix_rate']:.2f}%, "
            f"estimated_payload_end_rate={uniform_sequential['estimated_payload_end_rate']:.2f}%, "
            f"verdict={uniform_sequential['verdict']}"
        )

        _calibration_regression(scratch, cover_path)


if __name__ == "__main__":
    run_tests()
    print("test_detect.py passed")
