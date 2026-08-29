"""Webcam + MediaPipe Hands (solutions API)."""

from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from awr.gestures import INDEX_TIP, PointingGate

Event = dict[str, Any]

# MediaPipe landmarks are normalized [0,1], so inference can run on a smaller
# image and still map correctly onto the full camera frame.
_MP_WIDTH = 640
_MP_HEIGHT = 360


class HandTracker:
    def __init__(
        self,
        camera_index: int = 0,
        width: int = 960,
        height: int = 540,
        model_path=None,  # unused; kept for call-site compatibility
    ) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open camera. Check permissions and that a webcam is connected."
            )

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self._gate = PointingGate()

    def read(self) -> tuple[np.ndarray, Event]:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame from camera.")

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if w > _MP_WIDTH or h > _MP_HEIGHT:
            small = cv2.resize(rgb, (_MP_WIDTH, _MP_HEIGHT), interpolation=cv2.INTER_AREA)
        else:
            small = rgb
        small.flags.writeable = False
        result = self._hands.process(small)

        event: Event = {"kind": "noHand"}
        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]
            tip = lm.landmark[INDEX_TIP]
            x = max(0, min(w - 1, int(tip.x * w)))
            y = max(0, min(h - 1, int(tip.y * h)))
            if self._gate.update(lm.landmark):
                event = {"kind": "draw", "x": x, "y": y}
                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
            else:
                event = {"kind": "penUp", "x": x, "y": y}
                cv2.circle(frame, (x, y), 8, (0, 200, 255), 2)
        else:
            # Lost hand → stop ink immediately (no trailing stroke)
            self._gate.reset()

        return frame, event

    def release(self) -> None:
        self._hands.close()
        self.cap.release()
