"""Password-derived keys and payload integrity checks."""

import hashlib
import hmac

STATUS_EMBEDDED = "embedded"
STATUS_INTACT = "intact"
STATUS_TAMPERED = "tampered or corrupted"
STATUS_NO_MESSAGE = "no message found"
STATUS_CHECKSUM_MATCHES = "checksum matches"
STATUS_CHECKSUM_MISMATCH = "checksum mismatch"

PBKDF2_ITERATIONS = 100_000
# A fixed application salt keeps this student project simple. Production systems
# should store a unique random salt with each payload instead.
APPLICATION_SALT = b"bsc-stego-project-fixed-salt-v1"


def derive_keys(password: str) -> tuple[bytes, bytes]:
    """Derive independent embedding-order and HMAC keys from one password."""
    password_bytes = password.encode("utf-8")
    order_key = hashlib.pbkdf2_hmac(
        "sha256", password_bytes, APPLICATION_SALT + b"stego-order", PBKDF2_ITERATIONS
    )
    hmac_key = hashlib.pbkdf2_hmac(
        "sha256", password_bytes, APPLICATION_SALT + b"stego-hmac", PBKDF2_ITERATIONS
    )
    return order_key, hmac_key


def make_tag(message_bytes: bytes, password: str | None, hmac_key: bytes | None = None) -> str:
    """Make an HMAC tag for keyed mode or an unkeyed checksum for plain mode."""
    if password is None:
        return hashlib.sha256(message_bytes).hexdigest()
    if hmac_key is None:
        _, hmac_key = derive_keys(password)
    return hmac.new(hmac_key, message_bytes, hashlib.sha256).hexdigest()


def verify(message: str, tag: str, password: str | None = None,
           hmac_key: bytes | None = None) -> str:
    """Return integrity status for message text and its stored 64-char tag.

    A plain SHA-256 checksum only detects accidental corruption and is NOT
    authentication, because anyone can recompute it. Only password mode (HMAC)
    provides integrity protection against deliberate changes.
    """
    try:
        expected = make_tag(message.encode("utf-8"), password, hmac_key)
    except (UnicodeEncodeError, ValueError):
        return STATUS_TAMPERED if password is not None else STATUS_CHECKSUM_MISMATCH
    matches = hmac.compare_digest(expected, tag)
    if password is None:
        return STATUS_CHECKSUM_MATCHES if matches else STATUS_CHECKSUM_MISMATCH
    return STATUS_INTACT if matches else STATUS_TAMPERED
