"""Fuentes de fotogramas. OpenCV solo abre y lee la cámara."""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np


class CameraSource:
    """Captura BGR. Cada lectura se copia porque OpenCV puede reutilizar el buffer."""

    def __init__(self) -> None:
        self._cap: cv2.VideoCapture | None = None
        self.index = 0
        self.width = 0
        self.height = 0
        self.last_error = ""

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def is_synthetic(self) -> bool:
        return False

    def open(self, index: int = 0, width: int = 640, height: int = 480) -> bool:
        self.release()
        self.index = index
        self.last_error = ""
        backends: list[int] = []
        if sys.platform == "win32":
            backends.append(cv2.CAP_DSHOW)
        backends.append(cv2.CAP_ANY)
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                continue
            self._cap = cap
            self.width = int(frame.shape[1])
            self.height = int(frame.shape[0])
            return True
        self.last_error = (
            "No se pudo abrir la cámara. "
            "Revisa el permiso del sistema y que no la use otro programa."
        )
        return False

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        timestamp = time.perf_counter()
        if self._cap is None or not self._cap.isOpened():
            self.last_error = "La cámara no está abierta."
            return False, None, timestamp
        ok, frame = self._cap.read()
        timestamp = time.perf_counter()
        if not ok or frame is None:
            self.last_error = "La cámara dejó de entregar imágenes."
            return False, None, timestamp
        return True, frame.copy(), timestamp

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class SyntheticFrameSource:
    """Figuras generadas en memoria. No es una persona ni una captura de cámara.

    La interfaz debe mostrar esta fuente como «Demo sintética».
    """

    label = "Demo sintética"

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height
        self._t = 0

    @property
    def is_synthetic(self) -> bool:
        return True

    def reset(self) -> None:
        self._t = 0

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        frame = self._render(self._t)
        self._t += 1
        return True, frame, time.perf_counter()

    def release(self) -> None:
        return None

    def _render(self, t: int) -> np.ndarray:
        height, width = self.height, self.width
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (42, 38, 34)
        horizon = int(height * 0.78)
        frame[horizon : horizon + 3, :] = (72, 66, 60)
        cx = int(width * (0.50 + 0.30 * np.sin(t * 0.09)))
        cy = int(height * (0.46 + 0.18 * np.sin(t * 0.07 + 0.6)))
        self._disk(frame, cx, cy, max(22, width // 10), (210, 196, 168))
        cx2 = int(width * (0.52 + 0.22 * np.cos(t * 0.08)))
        cy2 = int(height * (0.42 + 0.14 * np.sin(t * 0.11)))
        self._disk(frame, cx2, cy2, max(14, width // 18), (186, 206, 214))
        return frame

    @staticmethod
    def _disk(frame: np.ndarray, cx: int, cy: int, radius: int, bgr: tuple[int, int, int]) -> None:
        height, width = frame.shape[:2]
        y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
        if y0 >= y1 or x0 >= x1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        frame[y0:y1, x0:x1][mask] = bgr
