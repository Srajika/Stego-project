"""Shared bit and payload helpers for the steganography methods."""

STOP_MARKER = b"#####"
TAG_BYTES = 64


def bytes_to_bits(data: bytes) -> str:
    """Convert bytes to a string of bits, preserving leading zeroes."""
    return "".join(f"{byte:08b}" for byte in data)


def bits_to_bytes(bits: str) -> bytes:
    """Convert complete groups of eight bits back to bytes."""
    if len(bits) % 8:
        raise ValueError("Bit data length must be a multiple of 8")
    if any(bit not in "01" for bit in bits):
        raise ValueError("Bit data may contain only '0' and '1'")
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def build_payload(message: str, tag: str) -> bytes:
    """Build payload bytes from UTF-8 message, stop marker, and 64-char tag."""
    message_bytes = message.encode("utf-8")
    if STOP_MARKER in message_bytes:
        raise ValueError("Message cannot contain the stop marker '#####'")
    if len(tag) != TAG_BYTES or any(char not in "0123456789abcdef" for char in tag):
        raise ValueError("Tag must be 64 lowercase hexadecimal characters")
    return message_bytes + STOP_MARKER + tag.encode("ascii")


def parse_payload(payload: bytes) -> tuple[bytes, str] | None:
    """Return message bytes and tag when a complete marked payload is found."""
    marker_at = payload.find(STOP_MARKER)
    if marker_at < 0:
        return None
    tag_start = marker_at + len(STOP_MARKER)
    tag_bytes = payload[tag_start:tag_start + TAG_BYTES]
    if len(tag_bytes) != TAG_BYTES:
        return None
    try:
        tag = tag_bytes.decode("ascii")
    except UnicodeDecodeError:
        return None
    return payload[:marker_at], tag
