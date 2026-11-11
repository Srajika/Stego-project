"""Flask client checks using generated images and isolated temporary files."""

import io
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app import create_app


def _png_bytes(size: tuple[int, int] = (512, 512)) -> bytes:
    """Build a repeatable RGB cover large enough for every DCT demo payload."""
    width, height = size
    rows, columns = np.indices((height, width))
    pixels = np.stack(
        (
            (columns * 3 + rows * 5) % 256,
            (columns * 7 + rows * 11 + 41) % 256,
            (columns * 13 + rows * 17 + 83) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    return output.getvalue()


class FlaskDemoTests(unittest.TestCase):
    """Check the visible routes, extraction states, and temp-file lifecycle."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="stego-web-test-")
        self.temp_root = Path(self.temp.name)
        self.app = create_app(
            {"TESTING": True, "TEMP_DIR": str(self.temp_root)}
        )
        self.client = self.app.test_client()
        self.cover = _png_bytes()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _hide(
        self,
        method: str,
        password: str | None = None,
        message: str = "A hidden UTF-8 note: café.",
    ) -> bytes:
        form = {
            "method": method,
            "message": message,
            "cover": (io.BytesIO(self.cover), "cover.png"),
        }
        if password is not None:
            form["password"] = password
        response = self.client.post(
            "/hide", data=form, content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertIn(b"Maximum message size", response.data)
        self.assertIn(
            f"Submitted message: <strong>{len(message.encode('utf-8'))} UTF-8 bytes</strong>".encode(),
            response.data,
        )
        match = re.search(rb'href="(/download/[0-9a-f]{32})"', response.data)
        self.assertIsNotNone(match)
        download = self.client.get(match.group(1).decode("ascii"))
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download.mimetype, "image/png")
        return download.data

    def _extract(self, image: bytes, method: str, password: str | None = None, **options):
        form = {
            "method": method,
            "stego_image": (io.BytesIO(image), "stego.png"),
            **options,
        }
        if password is not None:
            form["password"] = password
        return self.client.post(
            "/extract", data=form, content_type="multipart/form-data"
        )

    def test_round_trip_all_four_methods_and_clean_uploads(self) -> None:
        cases = (
            ("lsb_plain", None, "CHECKSUM MATCHES"),
            ("lsb_password", "demo password", "INTACT"),
            ("dct_robust", "demo password", "INTACT"),
            ("dct_balanced", None, "CHECKSUM MATCHES"),
        )
        for method, password, status in cases:
            with self.subTest(method=method):
                image = self._hide(method, password)
                result = self._extract(image, method, password)
                self.assertEqual(result.status_code, 200)
                self.assertIn(status.encode(), result.data)
                self.assertIn("A hidden UTF-8 note: café.".encode(), result.data)
                self.assertEqual(list(self.temp_root.iterdir()), [])

    def test_wrong_password_finds_no_shuffled_lsb_message(self) -> None:
        image = self._hide("lsb_password", "correct password")
        result = self._extract(image, "lsb_password", "wrong password")
        self.assertIn(b"NO MESSAGE FOUND", result.data)

    def test_extracted_user_text_is_html_escaped(self) -> None:
        message = "hello <script>alert(1)</script>"
        image = self._hide("lsb_plain", message=message)
        result = self._extract(image, "lsb_plain")
        self.assertIn(b"&lt;script&gt;alert(1)&lt;/script&gt;", result.data)
        self.assertNotIn(b"<script>", result.data)

    def test_dct_without_password_reads_message_with_checksum_mismatch(self) -> None:
        image = self._hide("dct_robust", "tag password")
        result = self._extract(image, "dct_robust")
        self.assertIn(b"CHECKSUM MISMATCH", result.data)
        self.assertIn("A hidden UTF-8 note: café.".encode(), result.data)

    def test_oversized_message_is_rejected_with_byte_capacity(self) -> None:
        tiny = _png_bytes((8, 8))
        response = self.client.post(
            "/hide",
            data={
                "method": "lsb_plain",
                "message": "x",
                "cover": (io.BytesIO(tiny), "tiny.png"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"allows at most 0 message bytes", response.data)
        self.assertEqual(list(self.temp_root.iterdir()), [])

    def test_non_image_upload_is_rejected(self) -> None:
        response = self.client.post(
            "/hide",
            data={
                "method": "lsb_plain",
                "message": "hello",
                "cover": (io.BytesIO(b"not an image"), "fake.png"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"not a valid PNG or JPEG", response.data)
        self.assertEqual(list(self.temp_root.iterdir()), [])

    def test_stop_marker_is_rejected_after_capacity_is_shown(self) -> None:
        response = self.client.post(
            "/hide",
            data={
                "method": "lsb_plain",
                "message": "blocked ##### marker",
                "cover": (io.BytesIO(self.cover), "cover.png"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"cannot contain the stop marker", response.data)
        self.assertIn(b"Maximum message size", response.data)
        self.assertEqual(list(self.temp_root.iterdir()), [])

    def test_jpeg_quality_70_breaks_lsb_but_quality_50_dct_survives(self) -> None:
        lsb_image = self._hide("lsb_plain")
        lsb_result = self._extract(
            lsb_image,
            "lsb_plain",
            simulate_jpeg="yes",
            jpeg_quality="70",
        )
        self.assertNotIn(b"CHECKSUM MATCHES", lsb_result.data)
        self.assertNotIn(b"A hidden UTF-8 note: caf", lsb_result.data)

        dct_image = self._hide("dct_robust", "jpeg password")
        dct_result = self._extract(
            dct_image,
            "dct_robust",
            "jpeg password",
            simulate_jpeg="yes",
            jpeg_quality="70",
        )
        self.assertEqual(dct_result.status_code, 200)
        self.assertIn(b"INTACT", dct_result.data)
        self.assertIn("A hidden UTF-8 note: café.".encode(), dct_result.data)

    def test_random_pixel_tamper_reports_corruption(self) -> None:
        image = self._hide("lsb_plain")
        response = self._extract(
            image,
            "lsb_plain",
            simulate_tamper="yes",
            tamper_count="50000",
        )
        self.assertIn(b"TAMPERED OR CORRUPTED", response.data)
        self.assertIn(b"Pixels edited: 50000", response.data)

    def test_detect_page_reports_verdict_and_corrected_dct_p_value(self) -> None:
        response = self.client.post(
            "/detect",
            data={"image": (io.BytesIO(self.cover), "cover.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Verdict", response.data)
        self.assertIn(b"Copy-corrected p-value", response.data)
        self.assertIn(b"quality 75 only", response.data)
        self.assertEqual(list(self.temp_root.iterdir()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
