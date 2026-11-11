"""Local-only Flask demonstration for the project's steganography methods."""

from __future__ import annotations

import io
import re
import tempfile
from pathlib import Path

from flask import (
    Flask,
    abort,
    render_template,
    request,
    send_file,
    url_for,
)

from stego import detect as lsb_detect
from stego import detect_dct, dct, lsb
from webapp_helpers import (
    MAX_UPLOAD_BYTES,
    METHODS,
    STATUS_LABELS,
    cleanup_old_files,
    copy_corrected_p_value,
    expire_file_later,
    get_method,
    method_capacity,
    method_items,
    random_path,
    remove_file,
    save_jpeg_copy,
    save_tampered_copy,
    save_verified_upload,
    quality_metrics,
)

STATUS_CLASSES = {
    "intact": "green",
    "tampered or corrupted": "red",
    "no message found": "yellow",
    "checksum matches": "cyan",
    "checksum mismatch": "pink",
}


def create_app(test_config: dict | None = None) -> Flask:
    """Build the demo app and allow tests to use an isolated temp directory."""
    app = Flask(__name__)
    app.config.from_mapping(
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
        TEMP_DIR=str(Path(tempfile.gettempdir()) / "stego-flask-demo"),
    )
    if test_config:
        app.config.update(test_config)

    def method_context(method_key: str) -> dict:
        selected = get_method(method_key)
        return {
            "methods": method_items(),
            "selected_method_key": method_key,
            "selected_method": selected,
            "password_mode": selected["password_mode"],
        }

    @app.before_request
    def prepare_temp_folder() -> None:
        """Create the temp folder and expire abandoned files before each request."""
        temp_root = Path(app.config["TEMP_DIR"])
        temp_root.mkdir(parents=True, exist_ok=True)
        cleanup_old_files(temp_root, max_age_seconds=3600)

    @app.errorhandler(413)
    def request_too_large(_error):
        """Show a friendly message when the 10 MB request limit is exceeded."""
        return render_template(
            "error.html", error="Uploads must be 10 MB or smaller."
        ), 413

    @app.errorhandler(404)
    def page_not_found(_error):
        """Render a simple not-found page without exposing server details."""
        return render_template("error.html", error="That page or download was not found."), 404

    @app.errorhandler(500)
    def server_error(_error):
        """Keep unexpected server errors friendly and free of stack traces."""
        return render_template(
            "error.html", error="The demo could not finish that request. Please try again."
        ), 500

    # Home route: introduce the methods and make the local-only safety boundary clear.
    @app.get("/")
    def home():
        return render_template("home.html")

    # Hide route: validate a cover, report byte capacity, embed the selected message, and offer a PNG.
    @app.route("/hide", methods=["GET", "POST"])
    def hide_page():
        method_key = request.values.get("method", "lsb_plain")
        if method_key not in METHODS:
            return render_template(
                "error.html", error="Choose one of the listed embedding methods."
            ), 400

        context = method_context(method_key)
        capacity = None
        message_size = None
        error_message = None
        result = None
        temp_root = Path(app.config["TEMP_DIR"])
        temporary_files: list[Path] = []
        output_path = None
        keep_output = False

        if request.method == "GET":
            return render_template(
                "hide.html", **context, capacity=None, message_size=None,
                error=None, result=None
            )

        message = request.form.get("message", "")
        password = request.form.get("password", "") or None
        message_size = len(message.encode("utf-8"))
        try:
            if context["password_mode"] == "required" and not password:
                raise ValueError("Enter a password for password-shuffled LSB.")

            cover_path, dimensions = save_verified_upload(
                request.files.get("cover"), temp_root
            )
            temporary_files.append(cover_path)
            capacity = method_capacity(dimensions[0], dimensions[1], method_key)
            if "#####" in message:
                raise ValueError("Messages cannot contain the stop marker #####.")
            if message_size > capacity["max_message_bytes"]:
                raise ValueError(
                    f"This message is {message_size} UTF-8 bytes; this image "
                    f"allows at most {capacity['max_message_bytes']} message bytes "
                    "after the 69-byte marker and tag."
                )

            output_path = random_path(temp_root, ".png")
            if context["selected_method"]["kind"] == "lsb":
                lsb.hide(
                    str(cover_path), str(output_path), message,
                    password=password if context["password_mode"] == "required" else None,
                )
            else:
                dct.hide(
                    str(cover_path), str(output_path), message,
                    password=password,
                    quality=context["selected_method"]["quality"],
                )
            metrics = quality_metrics(cover_path, output_path)
            keep_output = True
            expire_file_later(output_path)
            result = {
                "metrics": metrics,
                "description": context["selected_method"]["description"],
                "download_url": url_for("download_image", token=output_path.stem),
            }
        except ValueError as error:
            error_message = str(error)
        except Exception:
            app.logger.exception("Hide request failed")
            error_message = "The image could not be processed. Check the file and try again."
        finally:
            for path in temporary_files:
                remove_file(path)
            if not keep_output:
                remove_file(output_path)

        return render_template(
            "hide.html", **context, capacity=capacity, message_size=message_size,
            error=error_message, result=result,
        ), 400 if error_message and isinstance(error_message, str) else 200

    # Extract route: optionally recompress or tamper with a validated stego image, then verify its payload.
    @app.route("/extract", methods=["GET", "POST"])
    def extract_page():
        method_key = request.values.get("method", "lsb_plain")
        if method_key not in METHODS:
            return render_template(
                "error.html", error="Choose one of the listed extraction methods."
            ), 400
        context = method_context(method_key)
        error_message = None
        result = None
        temporary_files: list[Path] = []
        temp_root = Path(app.config["TEMP_DIR"])
        if request.method == "GET":
            return render_template("extract.html", **context, error=None, result=None)

        try:
            source_path, _dimensions = save_verified_upload(
                request.files.get("stego_image"), temp_root
            )
            temporary_files.append(source_path)
            active_path = source_path
            simulated_jpeg = request.form.get("simulate_jpeg") == "yes"
            jpeg_quality = 70
            if simulated_jpeg:
                try:
                    jpeg_quality = int(request.form.get("jpeg_quality", "70"))
                except ValueError as error:
                    raise ValueError("Choose a JPEG quality from 30 to 100.") from error
                if not 30 <= jpeg_quality <= 100:
                    raise ValueError("Choose a JPEG quality from 30 to 100.")
                active_path = save_jpeg_copy(active_path, temp_root, jpeg_quality)
                temporary_files.append(active_path)

            tamper_count = 0
            simulated_tamper = request.form.get("simulate_tamper") == "yes"
            if simulated_tamper:
                try:
                    requested_count = int(request.form.get("tamper_count", "10000"))
                except ValueError as error:
                    raise ValueError("Enter a whole number of pixels to flip.") from error
                if requested_count < 1:
                    raise ValueError("Choose at least one pixel to tamper with.")
                active_path, tamper_count = save_tampered_copy(
                    active_path, temp_root, requested_count
                )
                temporary_files.append(active_path)

            password = request.form.get("password", "") or None
            if context["selected_method"]["kind"] == "lsb":
                extracted = lsb.extract(str(active_path), password=password)
            else:
                extracted = dct.extract(
                    str(active_path), password=password,
                    quality=context["selected_method"]["quality"],
                )
            raw_status = extracted["status"]
            display_status = raw_status
            if simulated_tamper and raw_status in {
                "no message found", "checksum mismatch"
            }:
                display_status = "tampered or corrupted"
            result = {
                "status": display_status,
                "status_label": STATUS_LABELS.get(display_status, display_status.upper()),
                "status_class": STATUS_CLASSES.get(display_status, "red"),
                "message": extracted.get("message"),
                "simulated_jpeg": simulated_jpeg,
                "jpeg_quality": jpeg_quality,
                "simulated_tamper": simulated_tamper,
                "tamper_count": tamper_count,
                "tamper_note": (
                    "The simulated edits changed the payload or prevented it from being read."
                    if simulated_tamper and display_status == "tampered or corrupted"
                    else "The simulated edits did not change the integrity result."
                    if simulated_tamper
                    else None
                ),
            }
        except ValueError as error:
            error_message = str(error)
        except Exception:
            app.logger.exception("Extract request failed")
            error_message = "The image could not be read. Check the file and try again."
        finally:
            for path in temporary_files:
                remove_file(path)

        return render_template(
            "extract.html", **context, error=error_message, result=result
        ), 400 if error_message else 200

    # Detect route: run both LSB checks and the quality-75 DCT parity detector on one validated image.
    @app.route("/detect", methods=["GET", "POST"])
    def detect_page():
        if request.method == "GET":
            return render_template("detect.html", error=None, result=None)

        temp_root = Path(app.config["TEMP_DIR"])
        image_path = None
        error_message = None
        result = None
        try:
            image_path, _dimensions = save_verified_upload(
                request.files.get("image"), temp_root
            )
            lsb_result = lsb_detect.detect(str(image_path))
            sequential = lsb_detect.detect_sequential(str(image_path), steps=20)
            final_prefix = sequential["curve"][-1] if sequential["curve"] else {}
            dct_result = detect_dct.detect_dct_parity(str(image_path))
            result = {
                "lsb": lsb_result,
                "sequential": sequential,
                "sequential_statistic": final_prefix.get("statistic", float("nan")),
                "sequential_p_value": final_prefix.get("p_value", float("nan")),
                "dct": dct_result,
                "dct_corrected_p_value": copy_corrected_p_value(
                    dct_result["statistic"]
                ),
            }
        except ValueError as error:
            error_message = str(error)
        except Exception:
            app.logger.exception("Detect request failed")
            error_message = "The image could not be analyzed. Check the file and try again."
        finally:
            remove_file(image_path)

        return render_template(
            "detect.html", error=error_message, result=result
        ), 400 if error_message else 200

    # Download route: stream a one-time PNG by opaque token, then immediately delete its temp file.
    @app.get("/download/<token>")
    def download_image(token: str):
        if not re.fullmatch(r"[0-9a-f]{32}", token):
            abort(404)
        output_path = Path(app.config["TEMP_DIR"]) / f"{token}.png"
        if not output_path.is_file():
            abort(404)
        payload = output_path.read_bytes()
        remove_file(output_path)
        return send_file(
            io.BytesIO(payload), mimetype="image/png", as_attachment=True,
            download_name="stego.png",
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
