"""Plain and password-seeded least-significant-bit image hiding."""

import numpy as np
from PIL import Image

from .common import bits_to_bytes, build_payload, bytes_to_bits, parse_payload
from .tamper import (
    STATUS_CHECKSUM_MISMATCH,
    STATUS_EMBEDDED,
    STATUS_NO_MESSAGE,
    STATUS_TAMPERED,
    derive_keys,
    make_tag,
    verify,
)


def _position_order(order_key: bytes, count: int) -> np.ndarray:
    """Make a repeatable shuffled channel order from a derived secret key."""
    seed = int.from_bytes(order_key, byteorder="big")
    return np.random.default_rng(seed).permutation(count)


def hide(in_path: str, out_path: str, message: str, password: str | None = None) -> dict:
    """Embed a message in RGB channel LSBs and save a lossless PNG."""
    hmac_key = None
    if password is not None:
        order_key, hmac_key = derive_keys(password)
    message_bytes = message.encode("utf-8")
    tag = make_tag(message_bytes, password, hmac_key)
    bits = bytes_to_bits(build_payload(message, tag))

    with Image.open(in_path) as source:
        pixels = np.array(source.convert("RGB"))
    flat = pixels.reshape(-1).copy()
    if len(bits) > flat.size:
        raise ValueError(
            f"Message is too large for this image (needs {len(bits)} bits; "
            f"image holds {flat.size} bits)"
        )
    positions = np.arange(flat.size) if password is None else _position_order(order_key, flat.size)
    selected = positions[:len(bits)]
    flat[selected] = (flat[selected] & 254) | np.fromiter((int(bit) for bit in bits), dtype=np.uint8)
    Image.fromarray(flat.reshape(pixels.shape)).save(out_path, format="PNG")
    return {"status": STATUS_EMBEDDED, "path": out_path, "payload_bits": len(bits)}


def extract(path: str, password: str | None = None) -> dict:
    """Extract and verify a payload; plain checksums are not authentication."""
    if password is not None:
        order_key, hmac_key = derive_keys(password)
    else:
        order_key = hmac_key = None
    with Image.open(path) as image:
        flat = np.array(image.convert("RGB")).reshape(-1)
    positions = np.arange(flat.size) if password is None else _position_order(order_key, flat.size)
    ordered_bits = (flat[positions] & 1).astype(np.uint8)
    bit_string = "".join(str(bit) for bit in ordered_bits)
    # Pixel channel counts may leave 1-7 trailing bits that cannot form a byte.
    bit_string = bit_string[:len(bit_string) // 8 * 8]
    raw = bits_to_bytes(bit_string)
    parsed = parse_payload(raw)
    if parsed is None:
        return {"status": STATUS_NO_MESSAGE, "message": None}

    message_bytes, tag = parsed
    try:
        message = message_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return {"status": STATUS_TAMPERED, "message": None, "tag": tag}
    status = verify(message, tag, password, hmac_key)
    if status in (STATUS_CHECKSUM_MISMATCH, STATUS_TAMPERED):
        return {"status": status, "message": message, "tag": tag}
    return {"status": status, "message": message, "tag": tag}
