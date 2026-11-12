"""Flask client checks using generated images and isolated temporary files."""

import io
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app import create_app
from webapp_helpers import method_capacity


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


def _flat_png_bytes(size: tuple[int, int] = (512, 512)) -> bytes:
    """Build a clean flat cover whose LSB histogram is not balanced by chance."""
    width, height = size
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[:] = (100, 150, 200)
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    return output.getvalue()


def _even_random_png_bytes(size: tuple[int, int] = (1024, 1024)) -> bytes:
    """Build a larger, seeded cover with textured pixels and clean zero LSBs."""
    width, height = size
    rng = np.random.default_rng(4242)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    pixels &= np.uint8(254)
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

    def _request(self, method: str, *args, **kwargs):
        """Read and close each test response, including its multipart input stream."""
        response = getattr(self.client, method)(*args, **kwargs)
        input_stream = response.request.input_stream
        try:
            response.get_data()
        finally:
            response.close()
            if input_stream is not None:
                input_stream.close()
        return response

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
        response = self._request(
            "post",
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
        download = self._request("get", match.group(1).decode("ascii"))
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
        return self._request(
            "post",
            "/extract", data=form, content_type="multipart/form-data"
        )

    def _detector_demo(self, method: str, fill_rate: int) -> bytes:
        response = self._request(
            "post",
            "/hide",
            data={
                "method": method,
                "action": "detector_demo",
                "demo_fill_rate": str(fill_rate),
                "cover": (io.BytesIO(self.cover), "cover.png"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertIn(
            b"Demonstration payload: random bytes, NOT a readable message; Extract will not show text.",
            response.data,
        )
        match = re.search(rb'href="(/download/[0-9a-f]{32})"', response.data)
        self.assertIsNotNone(match)
        download = self._request("get", match.group(1).decode("ascii"))
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download.mimetype, "image/png")
        self.assertEqual(list(self.temp_root.iterdir()), [])
        return download.data

    def _detect(self, image: bytes):
        response = self._request(
            "post",
            "/detect",
            data={"image": (io.BytesIO(image), "detector-input.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertEqual(list(self.temp_root.iterdir()), [])
        return response

    def _section(self, response, heading: bytes, next_heading: bytes) -> bytes:
        body = response.data.split(heading, 1)[1]
        return body.split(next_heading, 1)[0]

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
        response = self._request(
            "post",
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
        response = self._request(
            "post",
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
        response = self._request(
            "post",
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
        response = self._request(
            "post",
            "/detect",
            data={"image": (io.BytesIO(self.cover), "cover.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Verdict", response.data)
        self.assertIn(b"Copy-corrected p-value", response.data)
        self.assertIn(b"quality 75 only", response.data)
        self.assertIn(
            b"falsely flags 2 of 23 (about 9%) of clean full covers and 12% (512x384 crops) to 22% (256x192 crops) of small crops",
            response.data,
        )
        self.assertEqual(list(self.temp_root.iterdir()), [])

    def test_detector_demo_payload_contract_and_temp_cleanup(self) -> None:
        cases = (
            ("lsb_plain", 10),
            ("lsb_password", 100),
            ("dct_balanced", 100),
        )
        for method, fill_rate in cases:
            with self.subTest(method=method):
                response = self._request(
                    "post",
                    "/hide",
                    data={
                        "method": method,
                        "action": "detector_demo",
                        "demo_fill_rate": str(fill_rate),
                        "cover": (io.BytesIO(self.cover), "cover.png"),
                    },
                    content_type="multipart/form-data",
                )
                self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
                payload_match = re.search(
                    rb"Framed payload: <strong>(\d+) bytes</strong>; one-bit fraction: <strong>([0-9.]+)%</strong>",
                    response.data,
                )
                rate_match = re.search(
                    rb"actual framed payload rate: <strong>([0-9.]+)%</strong>",
                    response.data,
                )
                self.assertIsNotNone(payload_match)
                self.assertIsNotNone(rate_match)
                capacity = method_capacity(512, 512, method)["capacity_bits"]
                expected_bytes = min(capacity * fill_rate // 800, capacity // 8)
                payload_bytes = int(payload_match.group(1))
                one_bit_fraction = float(payload_match.group(2)) / 100
                actual_rate = float(rate_match.group(1))
                self.assertEqual(payload_bytes, expected_bytes)
                self.assertGreaterEqual(actual_rate, fill_rate - 0.5)
                self.assertLessEqual(actual_rate, fill_rate)
                self.assertGreater(one_bit_fraction, 0.40)
                self.assertLess(one_bit_fraction, 0.60)

                match = re.search(rb'href="(/download/[0-9a-f]{32})"', response.data)
                self.assertIsNotNone(match)
                self.assertEqual(len(list(self.temp_root.iterdir())), 1)
                download = self._request("get", match.group(1).decode("ascii"))
                self.assertEqual(download.status_code, 200)
                self.assertEqual(list(self.temp_root.iterdir()), [])

    def test_clean_image_is_not_flagged_by_whole_image_lsb(self) -> None:
        response = self._detect(_even_random_png_bytes())
        whole_image = self._section(
            response, b"<h3>LSB chi-square</h3>", b"<h3>Sequential LSB"
        )
        self.assertIn(b"no evidence of LSB embedding", whole_image)

    def test_plain_lsb_10_percent_demo_triggers_sequential_detector(self) -> None:
        self.cover = _even_random_png_bytes()
        image = self._detector_demo("lsb_plain", 10)
        response = self._detect(image)
        sequential = self._section(
            response, b"<h3>Sequential LSB chi-square", b"<h3>DCT parity"
        )
        self.assertIn(b"embedding likely", sequential)
        self.assertRegex(
            sequential,
            rb"Estimated payload end: <strong>(?!not estimated)[0-9.]+%</strong>",
        )

    def test_plain_lsb_100_percent_demo_triggers_whole_image_detector(self) -> None:
        image = self._detector_demo("lsb_plain", 100)
        response = self._detect(image)
        whole_image = self._section(
            response, b"<h3>LSB chi-square</h3>", b"<h3>Sequential LSB"
        )
        self.assertIn(b"embedding likely", whole_image)

    def test_shuffled_lsb_100_percent_demo_triggers_whole_image_detector(self) -> None:
        image = self._detector_demo("lsb_password", 100)
        response = self._detect(image)
        whole_image = self._section(
            response, b"<h3>LSB chi-square</h3>", b"<h3>Sequential LSB"
        )
        self.assertIn(b"embedding likely", whole_image)

    def test_dct_100_percent_demo_is_flagged_by_copy_corrected_value(self) -> None:
        image = self._detector_demo("dct_balanced", 100)
        response = self._detect(image)
        dct_card = response.data.split(b"<h3>DCT parity", 1)[1].split(b"</article>", 1)[0]
        self.assertIn(b"Copy-corrected verdict: <strong>DCT embedding likely</strong>", dct_card)

    def test_short_plain_message_is_not_flagged_as_lsb_embedding(self) -> None:
        self.cover = _even_random_png_bytes()
        image = self._hide("lsb_plain", message="Short note.")
        response = self._detect(image)
        whole_image = self._section(
            response, b"<h3>LSB chi-square</h3>", b"<h3>Sequential LSB"
        )
        self.assertIn(b"no evidence of LSB embedding", whole_image)


if __name__ == "__main__":
    unittest.main(verbosity=2)
