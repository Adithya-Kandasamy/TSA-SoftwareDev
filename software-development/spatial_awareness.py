"""
Smart Spatial Awareness System
================================
Detects objects via webcam using YOLOv8, announces position + distance,
and gives voice safety warnings.

Requirements:  pip install ultralytics opencv-python pyttsx3 numpy
"""

import cv2
import numpy as np
import pyttsx3
import threading
import time
import platform
from collections import deque
from ultralytics import YOLO

# ─── CONFIG ──────────────────────────────────────────────────────────────────

TARGET_CLASSES = {"person", "chair", "backpack", "bottle", "laptop",
                  "cell phone", "cup", "book"}

NEAR_THRESHOLD   = 0.55
MEDIUM_THRESHOLD = 0.25
CENTER_ZONE = (0.33, 0.66)
VOICE_COOLDOWN = 2.5
APPROACH_FRAMES = 4

# ─── VOICE ENGINE ────────────────────────────────────────────────────────────

class VoiceEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 165)
        self.engine.setProperty("volume", 1.0)
        self._lock = threading.Lock()
        self._last_spoken = {}
        self._queue = deque()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def say(self, text, key=None, priority=False):
        now = time.time()
        k = key or text
        with self._lock:
            last = self._last_spoken.get(k, 0)
            if now - last < VOICE_COOLDOWN and not priority:
                return
            self._last_spoken[k] = now
            if priority:
                self._queue.appendleft(text)
            else:
                self._queue.append(text)

    def _worker(self):
        while True:
            if self._queue:
                text = self._queue.popleft()
                self.engine.say(text)
                self.engine.runAndWait()
            else:
                time.sleep(0.05)


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def get_position(cx, fw):
    ratio = cx / fw
    if ratio < CENTER_ZONE[0]:
        return "left"
    elif ratio > CENTER_ZONE[1]:
        return "right"
    return "center"


def get_distance(box_h, fh):
    ratio = box_h / fh
    if ratio > NEAR_THRESHOLD:
        return "near"
    elif ratio > MEDIUM_THRESHOLD:
        return "medium"
    return "far"


def box_area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


# ─── APPROACH TRACKER ────────────────────────────────────────────────────────

class ApproachTracker:
    def __init__(self):
        self._history = {}

    def update(self, track_id, area):
        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=APPROACH_FRAMES + 1)
        self._history[track_id].append(area)
        hist = self._history[track_id]
        if len(hist) < APPROACH_FRAMES:
            return False
        return all(hist[i] < hist[i + 1] for i in range(len(hist) - 1))

    def cleanup(self, active_ids):
        dead = [k for k in self._history if k not in active_ids]
        for k in dead:
            del self._history[k]


# ─── DRAW OVERLAY ────────────────────────────────────────────────────────────

COLORS = {
    "near":   (0, 60, 255),
    "medium": (0, 165, 255),
    "far":    (0, 200, 80),
}

def draw_overlay(frame, label, pos, dist, x1, y1, x2, y2, warning=False):
    color = COLORS[dist]
    if warning:
        color = (0, 0, 255)

    thickness = 3 if warning else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    tag = f"{label} | {pos} | {dist}"
    if warning:
        tag = f"!! {tag} APPROACHING !!"

    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, tag, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    if warning:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)


def draw_hud(frame, detections):
    h, w = frame.shape[:2]
    y = h - 10
    for text in reversed(detections):
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1,
                    cv2.LINE_AA)
        y -= 20


# ─── OPEN CAMERA (Mac-safe) ──────────────────────────────────────────────────

def open_camera():
    is_mac = platform.system() == "Darwin"

    # On Mac, use AVFoundation backend explicitly
    backends = (
        [(cv2.CAP_AVFOUNDATION, 0), (cv2.CAP_AVFOUNDATION, 1)]
        if is_mac else
        [(cv2.CAP_ANY, 0), (cv2.CAP_ANY, 1)]
    )

    for backend, idx in backends:
        print(f"[INFO] Trying camera {idx} (backend {backend})...")
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[INFO] Camera {idx} opened successfully.")
                return cap
            cap.release()

    return None


# ─── MAIN LOOP ───────────────────────────────────────────────────────────────

def main():
    print("[INFO] Loading YOLOv8n model ...")
    model = YOLO("yolov8n.pt")

    voice = VoiceEngine()
    tracker = ApproachTracker()

    cap = open_camera()
    if cap is None:
        print("[ERROR] Cannot open any camera. Check permissions in System Settings > Privacy > Camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    win_name = "Smart Spatial Awareness System"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 540)

    print("[INFO] System running. Press Q or ESC to quit.")
    voice.say("Smart spatial awareness system active", priority=True)

    fail_count = 0

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            fail_count += 1
            print(f"[WARN] Empty frame #{fail_count}")
            if fail_count > 30:
                print("[ERROR] Too many failed frames. Exiting.")
                break
            time.sleep(0.05)
            continue

        fail_count = 0  # reset on success
        fh, fw = frame.shape[:2]

        results = model.track(frame, persist=True, verbose=False,
                              conf=0.40, iou=0.45)[0]

        active_ids = set()
        hud_lines = []

        if results.boxes is not None and len(results.boxes):
            boxes   = results.boxes.xyxy.cpu().numpy()
            confs   = results.boxes.conf.cpu().numpy()
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)
            track_ids = (results.boxes.id.cpu().numpy().astype(int)
                         if results.boxes.id is not None
                         else list(range(len(boxes))))

            for box, conf, cls_id, tid in zip(boxes, confs, cls_ids, track_ids):
                label = model.names[cls_id]
                if label not in TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) / 2
                bh = y2 - y1

                pos  = get_position(cx, fw)
                dist = get_distance(bh, fh)
                area = box_area(box)

                active_ids.add(tid)
                approaching = tracker.update(tid, area)

                if approaching and dist in ("near", "medium"):
                    msg = f"Warning: {label} approaching"
                    voice.say(msg, key=f"warn_{tid}", priority=True)
                    draw_overlay(frame, label, pos, dist,
                                 x1, y1, x2, y2, warning=True)
                else:
                    if dist == "near" and pos == "center":
                        voice.say(f"Obstacle ahead: {label}", key=f"obs_{tid}")
                    else:
                        voice.say(f"{label}, {pos}, {dist}", key=f"loc_{tid}")
                    draw_overlay(frame, label, pos, dist, x1, y1, x2, y2)

                hud_lines.append(
                    f"{label}: {pos}  {dist}"
                    + ("  !! APPROACHING" if approaching else "")
                )

        tracker.cleanup(active_ids)
        draw_hud(frame, hud_lines)

        cv2.putText(frame, "SPATIAL AWARENESS  [Q = quit]",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(win_name, frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:
            break

        try:
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Shutting down.")


if __name__ == "__main__":
    main()