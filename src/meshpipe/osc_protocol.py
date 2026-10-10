"""Small float-only OSC codec shared by studio and Blender integration."""

import math
import struct

MAX_PACKET_BYTES = 8192


def _text(value: str) -> bytes:
    encoded = value.encode("utf-8") + b"\0"
    return encoded + b"\0" * (-len(encoded) % 4)


def osc_message(address: str, values: tuple[float, ...]) -> bytes:
    if not address.startswith("/") or "\0" in address or len(address) > 128:
        raise ValueError("Invalid OSC address")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("OSC values must be finite")
    return (
        _text(address)
        + _text("," + "f" * len(values))
        + struct.pack(">" + "f" * len(values), *values)
    )


def _read_text(data: bytes, offset: int) -> tuple[str, int]:
    end = data.find(b"\0", offset)
    if end < 0:
        raise ValueError("Truncated OSC string")
    try:
        text = data[offset:end].decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Invalid OSC string encoding") from error
    following = (end + 4) & ~3
    if following > len(data) or any(data[end:following]):
        raise ValueError("Invalid OSC padding")
    return text, following


def decode_bundle(data: bytes) -> dict[str, tuple[float, ...]]:
    if not 16 <= len(data) <= MAX_PACKET_BYTES or data[:8] != b"#bundle\0":
        raise ValueError("Invalid OSC bundle")
    messages = {}
    offset = 16
    while offset < len(data):
        if offset + 4 > len(data):
            raise ValueError("Truncated OSC bundle")
        length = struct.unpack_from(">I", data, offset)[0]
        offset += 4
        if not 8 <= length <= len(data) - offset or length % 4:
            raise ValueError("Invalid OSC message length")
        message = data[offset : offset + length]
        offset += length
        address, cursor = _read_text(message, 0)
        tags, cursor = _read_text(message, cursor)
        if (
            not address.startswith("/meshpipe/")
            or len(address) > 128
            or not tags.startswith(",")
            or set(tags[1:]) - {"f"}
        ):
            raise ValueError("Unsupported OSC address or value type")
        count = len(tags) - 1
        if count > 16 or cursor + count * 4 != len(message):
            raise ValueError("Invalid OSC value count")
        values = struct.unpack_from(">" + "f" * count, message, cursor)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Nonfinite OSC value")
        messages[address] = values
        if len(messages) > 100:
            raise ValueError("Too many OSC messages")
    return messages
