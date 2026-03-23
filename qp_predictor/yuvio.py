from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


class YUVReader420:
    def __init__(self, path: str, width: int, height: int, bit_depth: int = 8):
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.bit_depth = int(bit_depth)
        if self.bit_depth != 8:
            raise NotImplementedError("当前代码仅实现 8-bit YUV420。")
        self.y_size = self.width * self.height
        self.uv_size = (self.width // 2) * (self.height // 2)
        self.frame_size = self.y_size + 2 * self.uv_size

    def read_y(self, frame_idx: int) -> np.ndarray:
        offset = int(frame_idx) * self.frame_size
        with self.path.open("rb") as f:
            f.seek(offset)
            y = f.read(self.y_size)
        if len(y) != self.y_size:
            raise EOFError(f"Cannot read frame {frame_idx} from {self.path}")
        arr = np.frombuffer(y, dtype=np.uint8).reshape(self.height, self.width)
        return arr.copy()


def resize_y(img_y: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    pil = Image.fromarray(img_y, mode="L")
    pil = pil.resize((int(out_w), int(out_h)), resample=Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)
