"""Opt-in, loopback-only OSC transport. Never transmits images."""

import socket
import struct
import time

import numpy as np

from meshpipe.osc_protocol import osc_message
from meshpipe.pipeline import Frame
from meshpipe.pose import head_angles


def packet(frame: Frame, scores: dict[str, float], sequence: int) -> bytes:
    messages = [
        osc_message("/meshpipe/frame", (float(sequence), frame.timestamp_ms / 1000)),
        osc_message("/meshpipe/tracked", (float(frame.vertices is not None),)),
    ]
    if frame.vertices is not None:
        messages.append(osc_message("/meshpipe/rotation", head_angles(frame.vertices)))
        if frame.landmarks is not None:
            messages.append(osc_message("/meshpipe/position", tuple(frame.landmarks[1])))
        for name, value in scores.items():
            if len(name) > 80 or not name.isidentifier():
                raise ValueError("Invalid expression channel")
            messages.append(
                osc_message(f"/meshpipe/blendshape/{name}", (float(np.clip(value, 0, 1)),))
            )
    result = b"#bundle\0" + struct.pack(">Q", 1)
    result += b"".join(struct.pack(">I", len(message)) + message for message in messages)
    if len(result) > 8192:
        raise ValueError("OSC frame exceeds packet limit")
    return result


class LocalStream:
    def __init__(self, port: int = 9000, max_fps: float = 30):
        if not 1024 <= port <= 65535:
            raise ValueError("OSC port must be between 1024 and 65535")
        if not 1 <= max_fps <= 120:
            raise ValueError("OSC rate must be between 1 and 120")
        self.target = ("127.0.0.1", port)
        self.interval = 1 / max_fps
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.last_sent = float("-inf")
        self.sequence = 0
        self.dropped = 0

    def send(self, frame: Frame, scores: dict[str, float], clock: float | None = None) -> bool:
        clock = time.monotonic() if clock is None else clock
        if clock - self.last_sent < self.interval:
            return False
        payload = packet(frame, scores, self.sequence)
        self.last_sent = clock
        self.sequence += 1
        try:
            self.socket.sendto(payload, self.target)
            return True
        except BlockingIOError:
            self.dropped += 1
            return False

    def close(self) -> None:
        self.socket.close()
