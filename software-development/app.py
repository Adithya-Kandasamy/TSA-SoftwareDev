"""
Guardian Eye v2.1 — TSA Software Development
=============================================
Multi-model spatial intelligence system for accessibility.

Detection stack:
  • YOLO-World v2s  — open-vocabulary detection (80+ custom classes)
  • YOLOv8s fallback — COCO-80 real-time detection
  • 3×3 spatial zone grid — maps the scene into nine named regions
  • Per-track Kalman-inspired movement analysis — approach/recede/stable
  • Collision prediction — on-course + distance threshold
  • Projectile detection — fast-moving objects reclassified as threats
  • Scene description engine — natural-language environment summaries
  • Session statistics — live counters for judge showcase

Delivery:
  • Browser WebRTC camera → Flask inference → annotated JPEG + JSON
  • Web Speech API — priority voice alerts with distance milestones
  • Professional dark UI — designed for competition showcase
"""

# ─────────────────────────────── IMPORTS ────────────────────────────────────

import cv2
import numpy as np
import base64
import json
import threading
import time
import subprocess
import shutil
from collections import deque
from flask import Flask, request, render_template_string

try:
    from ultralytics import YOLOWorld
    _HAS_WORLD = True
except ImportError:
    _HAS_WORLD = False

from ultralytics import YOLO

# ─────────────────────────────── CONFIG ─────────────────────────────────────

PORT = 5001

# Zone thresholds (feet)
NEAR_FT   = 4
MEDIUM_FT = 7

# Collision trigger: within this distance AND actively approaching on-course
COLLISION_FT   = 3
# Point-blank: always a collision, no trajectory check needed
POINT_BLANK_FT = 1.0

# Projectile: any object approaching at this fractional area growth / frame
PROJECTILE_RATE = 0.15
PROJECTILE_FT   = 4

# Movement analysis window
TRACK_WINDOW      = 12     # slightly longer window for smoother tracking
APPROACH_THRESH   = 0.03   # lower → more sensitive to slow approaches
LATERAL_THRESH    = 0.055  # wider cone → catches off-center approaches

# Focal length (px, vertical) for distance estimation
# Calibrated: hand ~0.5ft away fills ~60% of 480px frame → effective_h=0.3ft
# FOCAL = pixel_h * dist / real_h = 290 * 0.5 / 0.3 ≈ 483 → round to 500
FOCAL_PX = 500

# ── Object taxonomy ──────────────────────────────────────────────────────────

# Effective visible heights in feet (calibrated for typical webcam framing)
HEIGHTS = {
    # ── People & animals ──────────────────────────────────────────────
    "person": 2.0, "baby": 1.0, "child": 3.0,
    "dog": 1.0, "cat": 0.7, "bird": 0.4, "horse": 3.5,
    "cow": 3.0, "sheep": 2.0, "bear": 3.5,
    "elephant": 8.0, "zebra": 5.0, "giraffe": 10.0,
    # ── Vehicles ─────────────────────────────────────────────────────
    "bicycle": 2.5, "car": 2.8, "motorcycle": 2.5,
    "bus": 5.0, "truck": 4.5, "boat": 3.0, "train": 5.0,
    "airplane": 10.0, "skateboard": 0.40, "scooter": 3.5,
    # ── Weapons / hazards ────────────────────────────────────────────
    "knife": 0.5, "scissors": 0.4, "gun": 0.45, "firearm": 0.45,
    "baseball bat": 2.5,
    # ── Everyday items ───────────────────────────────────────────────
    "pencil": 0.55, "pen": 0.45,
    "cell phone": 0.45, "laptop": 0.75, "keyboard": 0.10,
    "mouse": 0.20, "remote": 0.50, "book": 0.75,
    "backpack": 1.5, "handbag": 0.9, "suitcase": 1.5, "umbrella": 1.5,
    "bottle": 0.75, "cup": 0.35, "wine glass": 0.55, "bowl": 0.30,
    "fork": 0.50, "spoon": 0.40, "banana": 0.40, "apple": 0.30,
    "orange": 0.30, "pizza": 0.20, "donut": 0.15, "sandwich": 0.20,
    "broccoli": 0.25, "carrot": 0.30, "hot dog": 0.20, "cake": 0.25,
    "chair": 2.5, "couch": 2.0, "bed": 2.0, "dining table": 2.5,
    "toilet": 2.0, "tv": 1.5, "microwave": 0.80, "oven": 2.0,
    "refrigerator": 5.0, "sink": 2.0, "clock": 0.50, "vase": 0.80,
    "toothbrush": 0.50, "hair drier": 0.50, "teddy bear": 0.80,
    "potted plant": 1.0, "sports ball": 0.40,
    "surfboard": 2.0, "tennis racket": 1.5,
    "toaster": 0.40, "bench": 1.0,
    "tie": 1.0, "frisbee": 0.30, "skis": 4.0, "snowboard": 1.5,
    "kite": 0.5, "baseball glove": 0.40,
    # ── Traffic / navigation ─────────────────────────────────────────
    "traffic light": 2.5, "stop sign": 2.0,
    "fire hydrant": 1.5, "parking meter": 1.5,
    # ── Accessibility hazards (extended) ─────────────────────────────
    "stairs": 3.0, "steps": 2.5, "escalator": 7.0, "elevator": 7.0,
    "door": 6.5, "handrail": 3.5, "railing": 3.5,
    "curb": 0.5, "ramp": 1.0,
    "fire": 2.0, "smoke": 2.0, "flame": 2.0,
    "cable": 0.3, "wire": 0.3, "cord": 0.3,
    "puddle": 0.3, "wet floor": 1.5,
    "broken glass": 0.3,
    "fire extinguisher": 1.5,
    "trash can": 2.5,
    "wheelchair": 3.5, "crutches": 3.5, "cane": 3.0,
    "crosswalk": 0.5,
    "power outlet": 0.5,
}

DANGER_SET = {
    "person", "baby", "child",
    "car", "truck", "bus", "motorcycle", "bicycle", "airplane", "train", "scooter",
    "knife", "scissors", "gun", "firearm", "baseball bat",
    "dog", "bear", "horse", "elephant", "cow",
    "fire", "smoke", "flame",
    "stairs", "steps", "escalator",
    "broken glass",
    "cable", "wire", "cord",
    "traffic light", "stop sign",
}

ALL_CLASSES = set(HEIGHTS.keys())
BENIGN_SET  = ALL_CLASSES - DANGER_SET

# Objects that might be thrown
THROWABLE = {
    "sports ball", "bottle", "backpack", "handbag",
    "book", "cup", "bowl", "apple", "orange", "banana",
    "frisbee", "kite", "snowboard",
}

# Class list used when running YOLO-World (open-vocabulary)
WORLD_CLASSES = sorted(ALL_CLASSES)

# ─────────────────────────────── ZONES ──────────────────────────────────────
# The frame is divided into a 3×3 grid.
# Zone ID = row+col, e.g. "MC" = middle-centre (straight ahead).

ZONE_SPEECH = {
    "TL": "upper left",   "TC": "above you",       "TR": "upper right",
    "ML": "to your left", "MC": "straight ahead",   "MR": "to your right",
    "BL": "lower left",   "BC": "below you",        "BR": "lower right",
}

def classify_zone(cx_n: float, cy_n: float) -> str:
    col = "L" if cx_n < 0.33 else ("R" if cx_n > 0.67 else "C")
    row = "T" if cy_n < 0.33 else ("B" if cy_n > 0.67 else "M")
    return row + col

def zone_label(z: str) -> str:
    return ZONE_SPEECH.get(z, "nearby")

# ─────────────────────── MOVEMENT TRACKER ───────────────────────────────────

class Tracker:
    """Per-object motion analysis over a sliding window of frames."""

    def __init__(self):
        self._hist = {}   # tid → deque[(cx_n, cy_n, area, dist_ft)]
        self._lock = threading.Lock()

    def update(self, tid: int, cx_n: float, cy_n: float,
               area: float, dist_ft: float) -> dict:
        with self._lock:
            if tid not in self._hist:
                self._hist[tid] = deque(maxlen=TRACK_WINDOW + 1)
            self._hist[tid].append((cx_n, cy_n, area, dist_ft))
            h = self._hist[tid]

            if len(h) < 3:
                # Even on first frames, flag point-blank immediately
                return {"movement": "stable", "on_course": False,
                        "collision": dist_ft <= POINT_BLANK_FT,
                        "approach_rate": 0.0}

            old_area = h[0][2]
            cur_area = h[-1][2]
            approach_rate = ((cur_area - old_area) / (old_area * len(h))
                             if old_area > 0 else 0.0)

            cxs = [f[0] for f in h]
            cx_mean = sum(cxs) / len(cxs)
            lateral_var = sum((x - cx_mean) ** 2 for x in cxs) / len(cxs)

            if approach_rate > APPROACH_THRESH:
                movement = "approaching"
            elif approach_rate < -APPROACH_THRESH:
                movement = "receding"
            else:
                movement = "stable"

            on_course = (movement == "approaching"
                         and lateral_var < LATERAL_THRESH)

            # Collision fires when:
            #   a) actively on-course and within COLLISION_FT, OR
            #   b) point-blank (≤ POINT_BLANK_FT) regardless of trajectory —
            #      at that distance the object IS a collision hazard no matter what
            collision = (
                (on_course and dist_ft <= COLLISION_FT)
                or dist_ft <= POINT_BLANK_FT
            )

            return {
                "movement":      movement,
                "on_course":     on_course,
                "collision":     collision,
                "approach_rate": approach_rate,
            }

    def cleanup(self, active: set):
        with self._lock:
            for k in [k for k in self._hist if k not in active]:
                del self._hist[k]

# ─────────────────────── SCENE DESCRIBER ────────────────────────────────────

class SceneDescriber:
    """
    Generates a concise natural-language summary of the current scene,
    suitable for a periodic voice announcement.
    """

    def describe(self, dets: list) -> str:
        if not dets:
            return "Environment clear."

        collisions = [d for d in dets if d["collision"]]
        threats    = [d for d in dets if not d["collision"]
                      and not d["benign"]
                      and d["zone"] in ("near", "medium")]
        approaching = [d for d in dets if d["movement"] == "approaching"
                       and not d["collision"] and not d["benign"]]
        ambient     = [d for d in dets if d["benign"] and not d["collision"]]

        parts = []

        for d in collisions[:2]:
            ft = int(d["dist_ft"])
            parts.append(
                f"{d['label'].capitalize()} — collision imminent, "
                f"{ft} foot{'s' if ft != 1 else ''} {zone_label(d['grid_zone'])}"
            )

        for d in threats[:2]:
            ft  = int(d["dist_ft"])
            mv  = "approaching" if d["movement"] == "approaching" else "nearby"
            parts.append(
                f"{d['label'].capitalize()} {mv} {zone_label(d['grid_zone'])}, "
                f"{ft} feet"
            )

        if approaching and not collisions and not threats:
            d  = approaching[0]
            ft = int(d["dist_ft"])
            parts.append(
                f"{d['label'].capitalize()} moving closer — "
                f"{ft} feet, {zone_label(d['grid_zone'])}"
            )

        if ambient:
            unique = list(dict.fromkeys(d["label"] for d in ambient))[:4]
            if len(unique) == 1:
                parts.append(f"{unique[0].capitalize()} detected nearby")
            elif len(unique) == 2:
                parts.append(f"{unique[0].capitalize()} and {unique[1]} in view")
            else:
                parts.append(
                    f"{', '.join(u.capitalize() for u in unique[:-1])} "
                    f"and {unique[-1]} in environment"
                )

        if not parts:
            n = len(dets)
            parts.append(f"{n} object{'s' if n != 1 else ''} in view")

        return ". ".join(parts) + "."

# ─────────────────────── HELPERS ────────────────────────────────────────────

def estimate_dist_ft(label: str, pixel_h: int, frame_h: int = 480,
                     y1: int = None, y2: int = None) -> float:
    real_h = HEIGHTS.get(label, 2.0)
    if pixel_h < 1:
        return 99.0
    dist = (real_h * FOCAL_PX) / pixel_h
    # Edge-clipping correction: if the bounding box is clipped by the top or
    # bottom of the frame, the true object is taller than pixel_h shows → it's
    # closer than the formula calculates.  Only large fill-fractions trip this
    # so a distant tall object whose bbox stays within frame isn't affected.
    if y1 is not None and y2 is not None:
        clipped = (y1 <= 3) or (y2 >= frame_h - 3)
        if clipped:
            dist = min(dist, 2.0)   # object is definitely within ~2 ft
    return min(round(dist, 1), 99.0)

def get_dist_zone(dist_ft: float) -> str:
    if dist_ft <= NEAR_FT:   return "near"
    if dist_ft <= MEDIUM_FT: return "medium"
    return "far"

def box_area(x1, y1, x2, y2):
    return (x2 - x1) * (y2 - y1)

# ─────────────────────── DRAW ───────────────────────────────────────────────

NEON      = (0, 255, 100)    # BGR neon green — benign
COLORS    = {
    "near":   (20,  20, 245),
    "medium": (0,  180, 255),
    "far":    (0,  220,  80),
}
COLLISION_COLOR = (10, 10, 240)

def draw_overlay(frame, label, dist_zone, movement, collision, dist_ft,
                 x1, y1, x2, y2, is_benign: bool):
    """
    Minimal overlay:
      - L-corner ticks (neon green for benign, distance-colour for threats)
      - Small pill tag (neon accent bar for benign, colour bar for threats)
      - Pulsing double ring for collision threats
      - No filled bounding boxes
    """
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    arm = max(8, (x2 - x1) // 6)

    if is_benign:
        color, thick = NEON, 1
    else:
        color = COLLISION_COLOR if collision else COLORS[dist_zone]
        thick = 3 if collision else 2

    # Corner ticks
    corners = [
        ((x1, y1+arm), (x1, y1), (x1+arm, y1)),
        ((x2-arm, y1), (x2, y1), (x2, y1+arm)),
        ((x1, y2-arm), (x1, y2), (x1+arm, y2)),
        ((x2-arm, y2), (x2, y2), (x2, y2-arm)),
    ]
    for p in corners:
        cv2.line(frame, p[0], p[1], color, thick)
        cv2.line(frame, p[1], p[2], color, thick)
    cv2.circle(frame, (cx, cy), 3 if is_benign else (5 if collision else 3), color, -1)

    # Collision rings — triple concentric for urgency
    if collision:
        r = max(x2-x1, y2-y1) // 2 + 10
        cv2.circle(frame, (cx, cy), r,    COLLISION_COLOR, 3)
        cv2.circle(frame, (cx, cy), r+10, (40, 40, 220),   2)
        cv2.circle(frame, (cx, cy), r+20, (20, 20, 160),   1)
        # Red X at center
        s = 6
        cv2.line(frame, (cx-s, cy-s), (cx+s, cy+s), COLLISION_COLOR, 2)
        cv2.line(frame, (cx+s, cy-s), (cx-s, cy+s), COLLISION_COLOR, 2)

    # Tag — plain neon green text, dark outline for legibility on any background
    mv_sym = {"approaching": "▲", "receding": "▼", "stable": "–"}.get(movement, "")
    if collision:
        tag = f"!! {label}  {dist_ft:.0f}ft"
    else:
        tag = f"{label}  {mv_sym}  {dist_ft:.0f}ft"
    font, sc, fw_ = cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1
    (tw, th), _  = cv2.getTextSize(tag, font, sc, fw_)
    tx = max(0, min(cx - tw // 2, frame.shape[1] - tw))
    ty = max(th + 6, cy - 18)

    # Four-direction dark outline → neon text pops on any background
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cv2.putText(frame, tag, (tx+dx, ty+dy), font, sc, (0, 0, 0), fw_+1, cv2.LINE_AA)
    cv2.putText(frame, tag, (tx, ty), font, sc, NEON, fw_, cv2.LINE_AA)

# ─────────────────────────────── HTML ───────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Guardian Eye — Spatial Intelligence</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#020b16;--panel:#040e1c;--card:#071422;--border:#0d2035;
  --text:#cce4f7;--muted:#27435e;--accent:#00d4ff;--g:#39ff14;
  --danger:#ff3355;--warn:#ff9500;--safe:#00e676;--r:8px;
}
html,body{height:100%;overflow:hidden;background:var(--bg);color:var(--text);
  font-family:'Segoe UI',system-ui,sans-serif;font-size:14px}
body{display:grid;grid-template-rows:52px 1fr 26px}
main{display:grid;grid-template-columns:1fr 330px;overflow:hidden}

/* HEADER */
header{
  background:var(--panel);border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  padding:0 20px;gap:14px;
  box-shadow:0 2px 24px rgba(0,212,255,.05);
}
.logo{display:flex;align-items:center;gap:10px}
.logo-mark{
  width:34px;height:34px;border-radius:9px;
  background:linear-gradient(135deg,#00d4ff 0%,#0060ff 100%);
  display:flex;align-items:center;justify-content:center;font-size:17px;
  box-shadow:0 0 16px rgba(0,212,255,.4);flex-shrink:0;
}
.logo-text h1{font-size:.95rem;font-weight:800;letter-spacing:.5px}
.logo-text p{font-size:.5rem;color:var(--muted);text-transform:uppercase;letter-spacing:1.8px;margin-top:1px}
.hstats{display:flex;gap:22px;align-items:center}
.hs{text-align:center;min-width:38px}
.hs-v{font-size:1.2rem;font-weight:800;color:var(--accent);
  line-height:1;font-variant-numeric:tabular-nums;font-family:'Courier New',monospace}
.hs-l{font-size:.48rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-top:2px}
#hs-danger .hs-v{color:var(--danger)}
#hs-fps   .hs-v{color:var(--safe)}
#hs-sess  .hs-v{color:var(--warn)}
.hdivider{width:1px;height:28px;background:var(--border)}
.model-badge{
  background:rgba(0,212,255,.06);border:1px solid rgba(0,212,255,.14);
  border-radius:6px;padding:4px 10px;font-size:.56rem;
  color:rgba(0,212,255,.7);letter-spacing:.6px;text-transform:uppercase;
}
.status-pill{
  display:flex;align-items:center;gap:6px;
  background:rgba(0,230,118,.06);border:1px solid rgba(0,230,118,.2);
  border-radius:20px;padding:4px 12px;
  font-size:.58rem;color:var(--safe);letter-spacing:.8px;font-weight:700;
}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--safe);animation:blink 1.5s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.08}}

/* VIDEO COLUMN */
.vcol{display:grid;grid-template-rows:1fr 114px;border-right:1px solid var(--border);overflow:hidden}
.vwrap{position:relative;background:#000;overflow:hidden}
#raw-video{position:absolute;width:1px;height:1px;opacity:0;pointer-events:none}
#canvas{width:100%;height:100%;object-fit:contain;display:block}

/* zone overlay on video (3x3 grid lines) */
#zone-overlay{
  position:absolute;inset:0;pointer-events:none;z-index:2;
  display:grid;grid-template-columns:repeat(3,1fr);
  grid-template-rows:repeat(3,1fr);
}
.zo{border:1px solid rgba(255,255,255,.03);transition:background .25s}
.zo.zn{background:rgba(255,51,85,.1)!important;border-color:rgba(255,51,85,.3)!important}
.zo.zm{background:rgba(255,149,0,.07)!important;border-color:rgba(255,149,0,.2)!important}

/* danger flash ring */
#danger-flash{
  position:absolute;inset:0;pointer-events:none;z-index:5;
  border:0 solid var(--danger);transition:border-width .07s;
}
#danger-flash.on{border-width:6px;animation:fb .32s infinite alternate}
@keyframes fb{from{border-color:var(--danger)}to{border-color:rgba(255,51,85,.15)}}

/* collision banner */
#danger-banner{
  position:absolute;bottom:14px;left:50%;transform:translateX(-50%);
  display:none;z-index:10;min-width:280px;text-align:center;
  background:rgba(20,4,8,.88);backdrop-filter:blur(10px);
  color:var(--danger);font-weight:900;font-size:1.15rem;letter-spacing:2.5px;
  padding:11px 30px;border-radius:12px;
  border:2px solid rgba(255,51,85,.6);
  animation:gb .38s ease-in-out infinite alternate;
}
#danger-banner.on{display:block}
@keyframes gb{
  from{box-shadow:0 0 18px rgba(255,51,85,.5),inset 0 0 20px rgba(255,51,85,.1)}
  to  {box-shadow:0 0 60px rgba(255,51,85,.85),inset 0 0 30px rgba(255,51,85,.15)}
}
#danger-banner small{display:block;font-size:.68rem;font-weight:500;
  margin-top:3px;opacity:.85;letter-spacing:.5px;color:var(--text)}

/* camera denied */
#cam-error{
  position:absolute;inset:0;display:none;z-index:20;
  background:rgba(2,11,22,.96);
  flex-direction:column;align-items:center;justify-content:center;gap:12px;
  font-size:.85rem;color:var(--muted);
}
#cam-error span{font-size:2rem}

/* ZONE BAR */
.zonebar{
  background:var(--panel);border-top:1px solid var(--border);
  padding:7px 12px;display:flex;flex-direction:column;gap:5px;overflow:hidden;
}
.zone-hdr{display:flex;align-items:center;justify-content:space-between}
.zone-title{font-size:.5rem;color:var(--muted);text-transform:uppercase;letter-spacing:1.6px}
.zone-grid{
  display:grid;grid-template-columns:repeat(3,1fr);
  grid-template-rows:repeat(3,1fr);gap:3px;flex:1;min-height:0;
}
.zc{
  background:rgba(255,255,255,.012);border:1px solid var(--border);
  border-radius:5px;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:1px;
  padding:2px 3px;transition:all .22s;min-height:16px;
}
.zc.zn{border-color:var(--danger)!important;background:rgba(255,51,85,.1)!important;box-shadow:0 0 5px rgba(255,51,85,.2)}
.zc.zm{border-color:var(--warn)!important;background:rgba(255,149,0,.07)!important}
.zc.zf{border-color:var(--safe)!important;background:rgba(0,230,118,.04)!important}
.zc.za{border-color:var(--g)!important;background:rgba(57,255,20,.02)!important}
.zc-lbl{font-size:.4rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.zc-cnt{font-size:.5rem;color:var(--text);text-align:center;line-height:1.3;
  word-break:break-word;white-space:pre-line;max-width:100px}

/* SIDEBAR */
.side{display:flex;flex-direction:column;background:var(--panel);overflow:hidden}
.sbs{border-bottom:1px solid var(--border);padding:10px 12px;flex-shrink:0}
.sbs.grow{flex:1;overflow:hidden;display:flex;flex-direction:column}
.sbt{font-size:.5rem;color:var(--muted);text-transform:uppercase;
  letter-spacing:1.5px;margin-bottom:6px;display:flex;align-items:center;gap:5px}
.sbt-icon{font-size:.7rem}

/* SCENE */
#scene-desc{
  font-size:.73rem;line-height:1.55;color:var(--text);
  background:var(--card);border-radius:var(--r);padding:9px 11px;
  border-left:3px solid var(--accent);min-height:44px;transition:border-color .4s;
}
#scene-desc.danger{border-left-color:var(--danger)}
#scene-desc.warn{border-left-color:var(--warn)}

/* ALERTS */
#alog{display:flex;flex-direction:column;gap:3px;max-height:120px;overflow-y:auto}
.al{
  display:flex;align-items:flex-start;gap:7px;padding:7px 9px;
  border-radius:6px;font-size:.7rem;line-height:1.4;animation:fi .14s ease-out;
}
@keyframes fi{from{opacity:0;transform:translateX(8px)}to{opacity:1;transform:none}}
.al-d{background:rgba(255,51,85,.1);border-left:3px solid var(--danger)}
.al-w{background:rgba(255,149,0,.09);border-left:3px solid var(--warn)}
.al-i{background:rgba(0,212,255,.06);border-left:3px solid var(--accent)}
.dot{width:6px;height:6px;border-radius:50%;flex-shrink:0;margin-top:4px}
.al-d .dot{background:var(--danger)}.al-w .dot{background:var(--warn)}.al-i .dot{background:var(--accent)}
.alt{flex:1;font-weight:600}.altime{font-size:.52rem;color:var(--muted);flex-shrink:0;font-family:monospace}
.al-live{border-color:var(--danger)!important}

/* OBJECT LIST */
#olist{display:flex;flex-direction:column;gap:3px;overflow-y:auto;flex:1;padding-top:2px}
.obj{
  background:var(--card);border:1px solid var(--border);border-radius:7px;
  padding:7px 10px;display:grid;grid-template-columns:22px 1fr auto;
  align-items:start;gap:7px;transition:border-color .22s,box-shadow .22s;
}
.obj.od{border-color:rgba(255,51,85,.5);animation:throb .55s ease-in-out infinite alternate}
@keyframes throb{from{box-shadow:none}to{box-shadow:0 0 10px rgba(255,51,85,.22)}}
.obj.ow{border-color:rgba(255,149,0,.35)}
.obj.ob{opacity:.52}
.oicon{font-size:1.05rem;text-align:center;margin-top:1px}
.oname{font-size:.78rem;font-weight:700;text-transform:capitalize;line-height:1.2}
.ometa{font-size:.56rem;color:var(--muted);margin-top:2px}
.oconf{width:100%;height:2px;background:var(--border);border-radius:2px;margin-top:4px;overflow:hidden}
.oconf-bar{height:100%;border-radius:2px;transition:width .35s;background:var(--accent)}
.oconf-bar.od{background:var(--danger)}.oconf-bar.ow{background:var(--warn)}.oconf-bar.ob{background:var(--g)}
.obadges{display:flex;flex-direction:column;align-items:flex-end;gap:2px}
.bx{
  display:inline-flex;align-items:center;padding:2px 7px;border-radius:20px;
  font-size:.52rem;font-weight:700;text-transform:uppercase;letter-spacing:.4px;
}
.bn{background:rgba(255,51,85,.14);color:#ff7090;border:1px solid rgba(255,51,85,.22)}
.bm{background:rgba(255,149,0,.14);color:#ffb74d;border:1px solid rgba(255,149,0,.22)}
.bf{background:rgba(0,230,118,.08);color:#69f0ae;border:1px solid rgba(0,230,118,.18)}
.ba{background:rgba(255,51,85,.09);color:#ff8099}
.br{background:rgba(0,230,118,.07);color:#69f0ae}
.bs{background:rgba(25,45,65,.5);color:var(--muted)}
.bb{background:rgba(57,255,20,.05);color:var(--g);border:1px solid rgba(57,255,20,.16)}
.ft{font-size:.78rem;font-weight:800;font-variant-numeric:tabular-nums;
  color:var(--accent);font-family:'Courier New',monospace}
.ft.fn{color:var(--danger)}.ft.fw{color:var(--warn)}

/* VOICE BAR */
#voicebar{
  display:flex;align-items:center;gap:7px;padding-top:7px;
  font-size:.6rem;color:var(--muted);
  border-top:1px solid var(--border);margin-top:5px;flex-shrink:0;
}
#vicon{font-size:.95rem;transition:filter .2s,transform .18s}
#vicon.on{filter:drop-shadow(0 0 7px var(--accent));transform:scale(1.2)}
.vbars{display:flex;gap:2px;align-items:center;height:14px}
.vbar{width:3px;border-radius:2px;background:var(--muted);transition:height .12s}
.vbar.on{background:var(--accent);animation:vb .4s ease-in-out infinite alternate}
@keyframes vb{from{height:3px}to{height:13px}}
#vtxt{flex:1;font-style:italic;font-size:.58rem;color:var(--muted);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.vbtn{
  padding:2px 7px;border-radius:4px;font-size:.5rem;font-weight:700;
  border:1px solid var(--border);background:rgba(255,255,255,.04);
  color:var(--muted);cursor:pointer;letter-spacing:.4px;
  transition:all .15s;flex-shrink:0;
}
.vbtn:hover{background:rgba(0,212,255,.1);border-color:var(--accent);color:var(--accent)}

/* FOOTER */
footer{
  background:var(--panel);border-top:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  padding:0 16px;font-size:.52rem;color:var(--muted);letter-spacing:.4px;
}
footer span{display:flex;align-items:center;gap:5px}

/* SCROLLBARS */
::-webkit-scrollbar{width:3px}::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.empty{color:var(--muted);font-size:.68rem;text-align:center;padding:14px;opacity:.6}

</style>
</head>
<body>

<!-- HEADER -->
<header>
  <div class="logo">
    <div class="logo-mark">👁</div>
    <div class="logo-text">
      <h1>Guardian Eye</h1>
      <p>Spatial Intelligence &amp; Accessibility System</p>
    </div>
  </div>
  <div class="hstats">
    <div class="hs" id="hs-fps">
      <div class="hs-v">--</div><div class="hs-l">FPS</div>
    </div>
    <div class="hdivider"></div>
    <div class="hs">
      <div class="hs-v" id="hs-obj">0</div><div class="hs-l">Objects</div>
    </div>
    <div class="hs" id="hs-danger">
      <div class="hs-v" id="hs-thr">0</div><div class="hs-l">Threats</div>
    </div>
    <div class="hdivider"></div>
    <div class="hs" id="hs-sess">
      <div class="hs-v" id="hs-total">0</div><div class="hs-l">Session</div>
    </div>
    <div class="hdivider"></div>
    <div class="model-badge" id="model-badge">Loading…</div>
    <div class="status-pill"><div class="live-dot"></div>LIVE</div>
  </div>
</header>

<!-- MAIN -->
<main>

  <!-- VIDEO COLUMN -->
  <div class="vcol">
    <div class="vwrap">
      <video id="raw-video" autoplay playsinline muted></video>
      <canvas id="canvas"></canvas>
      <div id="zone-overlay">
        <div class="zo" id="zo-TL"></div><div class="zo" id="zo-TC"></div><div class="zo" id="zo-TR"></div>
        <div class="zo" id="zo-ML"></div><div class="zo" id="zo-MC"></div><div class="zo" id="zo-MR"></div>
        <div class="zo" id="zo-BL"></div><div class="zo" id="zo-BC"></div><div class="zo" id="zo-BR"></div>
      </div>
      <div id="danger-flash"></div>
      <div id="danger-banner">⚠ COLLISION ALERT<small id="dblabel">—</small></div>
      <div id="cam-error">
        <span>📷</span>
        Camera access required — click the camera icon in Chrome's address bar and allow access, then refresh.
      </div>
    </div>

    <!-- ZONE MAP -->
    <div class="zonebar">
      <div class="zone-hdr">
        <div class="zone-title">📍 Spatial Zone Map</div>
        <div class="zone-title" id="zone-count" style="color:var(--accent)"></div>
      </div>
      <div class="zone-grid" id="zgrid"></div>
    </div>
  </div>

  <!-- SIDEBAR -->
  <div class="side">

    <!-- SCENE DESCRIPTION -->
    <div class="sbs">
      <div class="sbt"><span class="sbt-icon">🗺</span> Scene Overview</div>
      <div id="scene-desc">Initialising camera…</div>
    </div>

    <!-- ACTIVE ALERTS -->
    <div class="sbs">
      <div class="sbt"><span class="sbt-icon">⚠</span> Active Alerts</div>
      <div id="alog"><div class="empty">No active threats</div></div>
      <div id="voicebar">
        <span id="vicon">🔊</span>
        <div class="vbars">
          <div class="vbar" id="vb1"></div>
          <div class="vbar" id="vb2"></div>
          <div class="vbar" id="vb3"></div>
          <div class="vbar" id="vb4"></div>
        </div>
        <span id="vtxt">Voice alerts ready</span>
        <button class="vbtn" onclick="testAudio()">▶ Test</button>
        <button class="vbtn" id="mute-btn" onclick="toggleMute()">🔊 Mute</button>
      </div>
    </div>

    <!-- OBJECT LIST -->
    <div class="sbs grow">
      <div class="sbt"><span class="sbt-icon">🔍</span> Detected Objects</div>
      <div id="olist"><div class="empty">Scanning environment…</div></div>
    </div>

  </div>
</main>

<!-- FOOTER -->
<footer>
  <span><div class="live-dot" style="display:inline-block"></div>Guardian Eye — TSA Software Development Showcase</span>
  <span id="lupdate">Initialising…</span>
  <span id="footer-model">Loading model…</span>
  <span style="opacity:.5">[S] describe · [M] mute</span>
</footer>

<script>
'use strict';

/* ── ICONS MAP ───────────────────────────────────────────────────────── */
const ICONS = {
  person:'🧍',baby:'👶',child:'🧒',
  car:'🚗',truck:'🚛',bus:'🚌',motorcycle:'🏍',bicycle:'🚲',
  airplane:'✈️',boat:'⛵',train:'🚆',scooter:'🛵',
  dog:'🐕',cat:'🐈',bird:'🐦',horse:'🐎',cow:'🐄',sheep:'🐑',
  bear:'🐻',elephant:'🐘',zebra:'🦓',giraffe:'🦒',
  knife:'🔪',scissors:'✂️',gun:'🔫',firearm:'🔫','baseball bat':'🏏',
  pencil:'✏️',pen:'🖊️',
  'cell phone':'📱',laptop:'💻',keyboard:'⌨️',mouse:'🖱️',
  remote:'📺',book:'📖',tv:'📺',
  backpack:'🎒',handbag:'👜',suitcase:'🧳',umbrella:'☂️',
  bottle:'🍶',cup:'☕','wine glass':'🍷',bowl:'🥣',
  fork:'🍴',spoon:'🥄',banana:'🍌',apple:'🍎',orange:'🍊',
  pizza:'🍕',donut:'🍩',sandwich:'🥪',broccoli:'🥦',carrot:'🥕',
  chair:'🪑',couch:'🛋️',bed:'🛏️','dining table':'🍽️',toilet:'🚽',
  microwave:'📦',oven:'🍳',refrigerator:'❄️',sink:'🚿',
  clock:'🕐',vase:'🏺',toothbrush:'🪥','hair drier':'💨',
  'teddy bear':'🧸','potted plant':'🪴','sports ball':'⚽',
  skateboard:'🛹',surfboard:'🏄','tennis racket':'🎾',
  bench:'🪑','traffic light':'🚦','stop sign':'🛑',
  'fire hydrant':'🚒',stairs:'🪜',door:'🚪',fire:'🔥',
  smoke:'💨',wheelchair:'♿',cable:'🔌',wire:'🔌',cord:'🔌',
  'fire extinguisher':'🧯','trash can':'🗑️',
  'broken glass':'🪟',frisbee:'🥏',skis:'⛷️',kite:'🪁',
};

/* ── BENIGN SET (JS-side mirror) ─────────────────────────────────────── */
const DANGER_LABELS = new Set([
  'person','baby','child','car','truck','bus','motorcycle','bicycle',
  'airplane','train','scooter','knife','scissors','gun','firearm',
  'baseball bat','dog','bear','horse','elephant','cow',
  'fire','smoke','flame','stairs','steps','escalator',
  'broken glass','cable','wire','cord','traffic light','stop sign',
]);

/* ── ZONE CONFIG ─────────────────────────────────────────────────────── */
const ZONE_ORDER = ['TL','TC','TR','ML','MC','MR','BL','BC','BR'];
const ZONE_NAMES = {
  TL:'Upper Left',TC:'Above',TR:'Upper Right',
  ML:'Left',MC:'Ahead',MR:'Right',
  BL:'Lower Left',BC:'Below',BR:'Lower Right',
};
const ZONE_SPEECH = {
  TL:'upper left',TC:'above you',TR:'upper right',
  ML:'to your left',MC:'straight ahead',MR:'to your right',
  BL:'lower left',BC:'below you',BR:'lower right',
};

/* ── BUILD ZONE GRID ─────────────────────────────────────────────────── */
(function(){
  const g = document.getElementById('zgrid');
  ZONE_ORDER.forEach(id=>{
    const d = document.createElement('div');
    d.className='zc'; d.id='z-'+id;
    d.innerHTML=`<div class="zc-lbl">${ZONE_NAMES[id]}</div><div class="zc-cnt" id="zc-${id}">—</div>`;
    g.appendChild(d);
  });
})();


/* ── SPEECH ENGINE (server-side TTS via macOS `say`) ─────────────────── */
// All speech is sent to /speak on the Flask server, which calls `say`.
// This bypasses all browser audio restrictions completely.

let spkMuted = false;
const spkCD  = {};
const CD_MS  = 8000;   // 8 s cooldown per key for non-priority
let   _spkActive = false;

function speak(text, key, priority=false){
  if(spkMuted) return;
  const now = Date.now();
  if(!priority && spkCD[key] && now - spkCD[key] < CD_MS) return;
  if(!priority) spkCD[key] = now;
  setVoiceActive(true);
  document.getElementById('vtxt').textContent = '"' + text + '"';
  // Fire-and-forget POST to server; server queues via macOS say
  fetch('/speak', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({text, priority})
  }).catch(()=>{});
  // Visual indicator stays on briefly
  clearTimeout(window._vTimer);
  window._vTimer = setTimeout(()=>setVoiceActive(false), 2500);
}

function toggleMute(){
  spkMuted = !spkMuted;
  const btn = document.getElementById('mute-btn');
  if(btn) btn.textContent = spkMuted ? '🔇 Unmute' : '🔊 Mute';
  document.getElementById('vtxt').textContent = spkMuted ? 'Voice muted' : 'Voice alerts ready';
  if(!spkMuted) speak('Voice unmuted.', '__mute__', true);
}

function testAudio(){
  speak('Guardian Eye voice test. Audio is working correctly.', '__test__', true);
}

function unlockVoice(){
  // Hide overlay (kept for visual cleanliness, not required for audio)
  const ov = document.getElementById('voice-unlock');
  if(ov) ov.style.display = 'none';
  speak('Guardian Eye active. Scanning environment.', '__welcome__', true);
}

function setVoiceActive(on){
  document.getElementById('vicon').classList.toggle('on', on);
  ['vb1','vb2','vb3','vb4'].forEach((id,i)=>{
    const el = document.getElementById(id);
    if(on){ el.classList.add('on'); el.style.animationDelay = (i*0.1)+'s'; }
    else el.classList.remove('on');
  });
}

/* ── DISTANCE MILESTONES ─────────────────────────────────────────────── */
const lastMile = {};
const MILES    = [20, 15, 10, 7, 5, 4, 3, 2, 1];

function checkMilestone(tid, label, dist_ft, on_course, zone, is_proj){
  if(!DANGER_LABELS.has(label) && !is_proj) return;
  const key  = tid + '_' + label;
  const prev = lastMile[key] ?? 999;
  for(const m of MILES){
    if(dist_ft <= m && prev > m){
      lastMile[key] = m;
      const phrase = buildPhrase(label, dist_ft, m, on_course, zone, is_proj);
      speak(phrase, key, m <= 4);
      break;
    }
  }
  if(dist_ft > (lastMile[key] ?? 0) + 6) delete lastMile[key];
}

function buildPhrase(label, dist_ft, m, on_course, zone, is_proj){
  const ft  = Math.round(dist_ft);
  const dir = ZONE_SPEECH[zone] || 'nearby';
  const obj = cap(label);
  if(is_proj) return `Projectile! ${obj} incoming from ${dir}. ${ft} feet.`;
  const urg = m<=2 ? 'Danger! ' : m<=4 ? 'Warning! ' : '';
  if(m<=1) return `Danger alert! ${obj} — ${ft} foot. Collision imminent!`;
  if(m<=3) return `${urg}${obj} — ${ft} feet, ${dir}.`;
  if(m<=5) return on_course
    ? `Caution. ${obj} on collision course. ${ft} feet, ${dir}.`
    : `Caution. ${obj} — ${ft} feet, ${dir}.`;
  if(m<=7) return `${obj} closing in. ${ft} feet, ${dir}.`;
  if(m<=10)return `${obj} approaching. ${ft} feet, ${dir}.`;
  return      `${obj} detected. ${ft} feet, ${dir}.`;
}

/* ── PERIODIC SCENE DESCRIPTION ─────────────────────────────────────── */
let lastSceneSpeak = 0;
const SCENE_MS = 13000;

function maybeDescribeScene(dets){
  const now = Date.now();
  if(now - lastSceneSpeak < SCENE_MS) return;
  lastSceneSpeak = now;
  speak(generateScene(dets), '__scene__', false);
}

function generateScene(dets){
  if(!dets.length) return 'Environment clear. No objects detected.';
  const cols = dets.filter(d=>d.collision);
  const thr  = dets.filter(d=>!d.collision && DANGER_LABELS.has(d.label)
                              && (d.zone==='near'||d.zone==='medium'));
  const amb  = dets.filter(d=>!DANGER_LABELS.has(d.label) && !d.collision);
  const parts = [];
  cols.slice(0,2).forEach(d=>
    parts.push(`${cap(d.label)} approaching ${ZONE_SPEECH[d.grid_zone]??'nearby'}, ${Math.round(d.dist_ft)} feet — collision imminent`)
  );
  thr.slice(0,3).forEach(d=>{
    const mv = d.movement==='approaching'?'approaching':'nearby';
    parts.push(`${cap(d.label)} ${mv} ${ZONE_SPEECH[d.grid_zone]??'nearby'}, ${Math.round(d.dist_ft)} feet`);
  });
  if(amb.length){
    const ls = [...new Set(amb.map(d=>d.label))].slice(0,3);
    parts.push(ls.length===1 ? `${ls[0]} nearby` : `${ls.slice(0,-1).join(', ')} and ${ls.at(-1)} in the environment`);
  }
  if(!parts.length){
    const n = dets.length;
    parts.push(`${n} object${n!==1?'s':''} detected around you`);
  }
  return parts.join('. ')+'.';
}

/* ── STATE ───────────────────────────────────────────────────────────── */
let processing=false, fc=0, lastT=performance.now();
let sessionTotal=0;

// Adaptive frame-skip: skip canvas draws if the server round-trip is slow
// to keep the UI from feeling frozen. We measure how long each /process call
// takes and skip every other animation frame if it's >300 ms.
let lastRTT=100, frameSkip=0, frameSkipCounter=0;

// Collision state machine: only fire when a tid *enters* collision.
// activeCollisions keeps a lastSeen timestamp so rapid on/off bouncing
// (caused by noisy approach_rate) doesn't cause flickering — we hold
// the entry for COLLISION_HOLD_MS after the last collision frame.
const activeCollisions  = new Map();   // tid → {label, dist_ft, grid_zone, lastSeen}
const collisionLog      = [];          // recent events for display (max 8)
const COLLISION_HOLD_MS = 2500;        // keep alert live for 2.5s after last collision frame
const POINT_BLANK_FT    = 1.0;         // threshold for "less than a foot" warning
const pbLastSpoke       = new Map();   // tid → timestamp, cooldown for point-blank alerts
const PB_COOLDOWN_MS    = 3000;        // re-announce point-blank every 3s at most
let   _lastAlogHtml     = '';          // diff-guard: only update DOM when content changes

/* ── KEYBOARD SHORTCUTS ──────────────────────────────────────────────── */
document.addEventListener('keydown', e=>{
  if(e.key==='s'||e.key==='S'){
    // Force immediate scene description
    lastSceneSpeak = 0;
  }
  if(e.key==='m'||e.key==='M'){
    toggleMute();
  }
});

/* ── WEBCAM ──────────────────────────────────────────────────────────── */
navigator.mediaDevices.getUserMedia({video:{width:640,height:480,facingMode:'user'},audio:false})
  .then(stream=>{
    const v = document.getElementById('raw-video');
    v.srcObject = stream;
    v.onloadedmetadata = ()=>{
      const c = document.getElementById('canvas');
      c.width = v.videoWidth||640; c.height = v.videoHeight||480;
      requestAnimationFrame(loop);
    };
  })
  .catch(()=>{
    document.getElementById('cam-error').style.display='flex';
    document.getElementById('olist').innerHTML=
      '<div style="color:var(--danger);padding:10px;font-size:.78rem">⚠ Camera denied.</div>';
  });

/* ── FETCH SESSION STATS ─────────────────────────────────────────────── */
function fetchStats(){
  fetch('/stats').then(r=>r.json()).then(s=>{
    document.getElementById('model-badge').textContent = s.model || 'YOLOv8';
    document.getElementById('footer-model').textContent =
      (s.model||'YOLOv8') + ' · ' + (s.class_count||80) + ' classes · WebRTC · Web Speech API';
    document.getElementById('hs-total').textContent = s.total_detections > 9999
      ? Math.floor(s.total_detections/1000)+'k' : s.total_detections;
  }).catch(()=>{});
}
fetchStats();
setInterval(fetchStats, 10000);

/* ── FRAME LOOP ──────────────────────────────────────────────────────── */
async function loop(){
  requestAnimationFrame(loop);

  // Adaptive skip: if RTT>250ms skip every other animation frame to reduce
  // concurrent fetches and let the CPU breathe
  if(lastRTT > 250){
    frameSkipCounter = (frameSkipCounter+1) % 2;
    if(frameSkipCounter !== 0) return;
  }

  if(processing) return;
  processing = true;
  const v  = document.getElementById('raw-video');
  const oc = document.createElement('canvas');
  oc.width=640; oc.height=480;
  oc.getContext('2d').drawImage(v,0,0,640,480);
  const b64 = oc.toDataURL('image/jpeg', .70).split(',')[1];
  const t0 = performance.now();
  try{
    const res = await fetch('/process',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({frame:b64})
    });
    const d = await res.json();
    lastRTT = performance.now() - t0;
    if(d.frame){
      const img = new Image();
      img.onload = ()=>{
        const c = document.getElementById('canvas');
        c.width=img.width; c.height=img.height;
        c.getContext('2d').drawImage(img,0,0);
      };
      img.src = 'data:image/jpeg;base64,'+d.frame;
    }
    update(d.detections||[]);
    fc++;
    const now = performance.now();
    if(now-lastT >= 1000){
      const fps = fc;
      const fpsEl = document.getElementById('hs-fps').querySelector('.hs-v');
      fpsEl.textContent = fps;
      // Color-code FPS
      fpsEl.style.color = fps >= 5 ? 'var(--safe)' : fps >= 2 ? 'var(--warn)' : 'var(--danger)';
      fc=0; lastT=now;
    }
    document.getElementById('lupdate').textContent='Last scan: '+new Date().toLocaleTimeString();
  }catch(e){
    lastRTT = 500;
    // Auto-reconnect: if server is down, retry every 3s
    document.getElementById('lupdate').textContent='Reconnecting…';
    await new Promise(r=>setTimeout(r,3000));
  }
  processing = false;
}

/* ── UI UPDATE ───────────────────────────────────────────────────────── */
function update(items){
  const threats = items.filter(d=>d.collision);
  document.getElementById('hs-obj').textContent = items.length;
  document.getElementById('hs-thr').textContent = threats.length;

  // Danger banner + flash
  const df = document.getElementById('danger-flash');
  const db = document.getElementById('danger-banner');
  if(threats.length){
    df.classList.add('on'); db.classList.add('on');
    document.getElementById('dblabel').textContent =
      threats.map(d=>`${cap(d.label)} — ${Math.round(d.dist_ft)} ft`).join(' · ');
  } else {
    df.classList.remove('on'); db.classList.remove('on');
  }

  // Scene description
  const scene = generateScene(items);
  const sd = document.getElementById('scene-desc');
  sd.textContent = scene;
  sd.className = threats.length ? 'danger' : (items.some(d=>DANGER_LABELS.has(d.label)&&d.zone==='near') ? 'warn' : '');

  // Overlay zone highlight (on video)
  ZONE_ORDER.forEach(id=>{
    document.getElementById('zo-'+id).className='zo';
  });
  items.forEach(d=>{
    const el = document.getElementById('zo-'+d.grid_zone);
    if(!el) return;
    if(d.collision||d.is_projectile) el.classList.add('zn');
    else if(DANGER_LABELS.has(d.label) && d.zone==='near' && !el.classList.contains('zn')) el.classList.add('zm');
  });

  // ── Collision state machine ───────────────────────────────────────
  const collidingNow = new Set(items.filter(d=>d.collision).map(d=>d.tid));

  const now_ms = Date.now();

  items.forEach(d=>{
    if(!d.collision) return;
    if(!activeCollisions.has(d.tid)){
      // New collision — speak once, log once
      const ft     = Math.round(d.dist_ft);
      const zsp    = ZONE_SPEECH[d.grid_zone] ?? 'nearby';
      const phrase = `Collision alert! ${cap(d.label)} — ${ft} foot${ft!==1?'s':''}, ${zsp}.`;
      speak(phrase, `col_${d.tid}`, true);
      spkCD[`col_${d.tid}`] = 0;
      const logTxt = `${ICONS[d.label]||'⚠'} ${cap(d.label)} — ${ft} ft · ${zsp}`;
      collisionLog.unshift({text:logTxt, time:ts(), tid:d.tid});
      if(collisionLog.length>8) collisionLog.pop();
      sessionTotal++;
    }
    // Always refresh lastSeen so the debounce timer resets each collision frame
    activeCollisions.set(d.tid, {
      label:d.label, dist_ft:d.dist_ft, grid_zone:d.grid_zone, lastSeen:now_ms
    });
  });

  // Only clear a collision after it has been absent for COLLISION_HOLD_MS
  // This prevents flickering when approach_rate briefly dips below threshold
  for(const [tid, v] of activeCollisions){
    if(!collidingNow.has(tid) && now_ms - v.lastSeen > COLLISION_HOLD_MS){
      activeCollisions.delete(tid);
    }
  }

  // ── Point-blank alerts (≤ 1 ft) — re-announce every 3 s ─────────────
  items.forEach(d=>{
    if(d.dist_ft > POINT_BLANK_FT) return;
    const last = pbLastSpoke.get(d.tid) ?? 0;
    if(now_ms - last > PB_COOLDOWN_MS){
      pbLastSpoke.set(d.tid, now_ms);
      speak(`Warning! ${cap(d.label)} — obstacle less than a foot away!`,
            `pb_${d.tid}`, true);
      spkCD[`pb_${d.tid}`] = 0;
    }
  });
  // Clean up point-blank cooldown for tracks that have moved away
  for(const tid of pbLastSpoke.keys()){
    if(!items.find(d=>d.tid===tid && d.dist_ft<=POINT_BLANK_FT*2))
      pbLastSpoke.delete(tid);
  }

  // Milestone announcements for non-collision threats
  items.forEach(d=>{
    if(!d.collision)
      checkMilestone(d.tid, d.label, d.dist_ft, d.on_course, d.grid_zone, d.is_projectile);
  });

  maybeDescribeScene(items);

  // ── Alert panel (only re-render DOM when content actually changes) ──
  const alog = document.getElementById('alog');
  let newAlogHtml;
  if(activeCollisions.size===0 && collisionLog.length===0){
    newAlogHtml = '<div class="empty">No active threats</div>';
  } else {
    newAlogHtml = '';
    for(const [,v] of activeCollisions){
      const ft    = v.dist_ft <= POINT_BLANK_FT ? '&lt; 1' : Math.round(v.dist_ft);
      const zsp   = ZONE_SPEECH[v.grid_zone]??'nearby';
      const ptBlk = v.dist_ft <= POINT_BLANK_FT
        ? ' style="color:#ff0"' : '';
      newAlogHtml += `<div class="al al-d">
        <div class="dot" style="animation:blink .5s infinite"></div>
        <div class="alt"${ptBlk}>${ICONS[v.label]||'⚠'} <strong>${cap(v.label)}</strong> — ${ft} ft · ${zsp}</div>
        <div class="altime" style="color:var(--danger);font-weight:800">LIVE</div></div>`;
    }
    collisionLog.slice(0,5).forEach(a=>{
      newAlogHtml += `<div class="al al-d" style="opacity:.4"><div class="dot"></div>
        <div class="alt">${a.text}</div><div class="altime">${a.time}</div></div>`;
    });
  }
  // Only touch the DOM if the rendered content actually changed
  if(newAlogHtml !== _lastAlogHtml){
    alog.innerHTML  = newAlogHtml;
    _lastAlogHtml   = newAlogHtml;
  }

  // ── Zone map ──────────────────────────────────────────────────────
  updateZones(items);

  // ── Object list ───────────────────────────────────────────────────
  if(!items.length){
    document.getElementById('olist').innerHTML='<div class="empty">No objects detected</div>';
    return;
  }
  document.getElementById('olist').innerHTML = items.map(d=>{
    const isDanger = DANGER_LABELS.has(d.label) || d.is_projectile;
    const ib  = !isDanger && !d.collision;
    const cls = d.collision ? 'od' : (isDanger && d.zone==='near') ? 'ow' : ib ? 'ob' : '';
    const dc  = d.zone==='near'?'bn':d.zone==='medium'?'bm':'bf';
    const mc  = d.movement==='approaching'?'ba':d.movement==='receding'?'br':'bs';
    const mTx = d.movement==='approaching'?'▲ Approach':d.movement==='receding'?'▼ Recede':'■ Stable';
    const ftc = d.zone==='near'?'fn':d.zone==='medium'?'fw':'';
    const meta = ib
      ? `Ambient · ${ZONE_SPEECH[d.grid_zone]??'nearby'}`
      : `${ZONE_SPEECH[d.grid_zone]??'nearby'}${d.on_course?' · on course':''}`;
    const icon = ICONS[d.label] || ICONS[d.label.split(' ')[0]] || '📦';
    const conf = Math.round((d.conf||0.5)*100);
    const barCls = d.collision?'od':(isDanger&&d.zone==='near')?'ow':ib?'ob':'';
    return `<div class="obj ${cls}">
      <div class="oicon">${icon}</div>
      <div>
        <div class="oname">${d.label}</div>
        <div class="ometa">${meta}</div>
        <div class="oconf"><div class="oconf-bar ${barCls}" style="width:${conf}%"></div></div>
      </div>
      <div class="obadges">
        <span class="ft ${ftc}">${Math.round(d.dist_ft)} ft</span>
        <span class="bx ${ib?'bb':dc}">${d.zone}</span>
        <span class="bx ${ib?'bb':mc}">${ib?'ambient':mTx}</span>
      </div>
    </div>`;
  }).join('');
}

/* ── ZONE MAP ────────────────────────────────────────────────────────── */
function updateZones(items){
  ZONE_ORDER.forEach(id=>{
    const c=document.getElementById('z-'+id);
    c.className='zc';
    document.getElementById('zc-'+id).textContent='—';
  });
  const byZone={};
  items.forEach(d=>{
    if(!d.grid_zone) return;
    (byZone[d.grid_zone]||(byZone[d.grid_zone]=[])).push(d);
  });
  let total=0;
  Object.entries(byZone).forEach(([z,zi])=>{
    const cell  = document.getElementById('z-'+z);
    const inner = document.getElementById('zc-'+z);
    if(!cell||!inner) return;
    total += zi.length;
    const thr = zi.filter(d=>DANGER_LABELS.has(d.label)||d.collision||d.is_projectile);
    const ben = zi.filter(d=>!DANGER_LABELS.has(d.label)&&!d.collision&&!d.is_projectile);
    if(thr.some(d=>d.zone==='near')||zi.some(d=>d.collision))   cell.classList.add('zn');
    else if(thr.some(d=>d.zone==='medium'))                      cell.classList.add('zm');
    else if(thr.some(d=>d.zone==='far'))                         cell.classList.add('zf');
    else if(ben.length)                                          cell.classList.add('za');
    const lbl = zi.slice(0,2).map(d=>(ICONS[d.label]||'📦')+' '+d.label).join('\n');
    inner.textContent = lbl||'—';
  });
  document.getElementById('zone-count').textContent = total ? total+' tracked' : '';
}

/* ── HELPERS ─────────────────────────────────────────────────────────── */
function cap(s){ return s.charAt(0).toUpperCase()+s.slice(1) }
function ts(){ return new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'}) }
</script>
</body>
</html>
"""


# ─────────────────────────── FLASK APP ──────────────────────────────────────

app      = Flask(__name__)
tracker  = Tracker()
describer = SceneDescriber()
model    = None

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/process", methods=["POST"])
def process():
    data  = request.get_json(force=True)
    b64   = data.get("frame", "")
    arr   = np.frombuffer(base64.b64decode(b64), np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        return _json({"detections": [], "frame": ""})

    fh, fw = frame.shape[:2]
    frame_area = fw * fh

    # ── Inference ──
    # 384px balances speed and accuracy well for webcam input on CPU.
    # YOLO-World uses this as the backbone input; CLIP text embeddings
    # are pre-computed so there's no extra per-frame overhead.
    INF_SZ = 384
    small  = cv2.resize(frame, (INF_SZ, INF_SZ))
    try:
        results = model.track(small, persist=True, verbose=False,
                               conf=0.28, iou=0.40, imgsz=INF_SZ)[0]
    except Exception:
        # Fallback: predict without ByteTrack (assigns sequential IDs)
        results = model.predict(small, verbose=False,
                                conf=0.28, iou=0.40, imgsz=INF_SZ)[0]
    sx, sy = fw / INF_SZ, fh / INF_SZ

    active = set()
    dets   = []

    if results.boxes is not None and len(results.boxes):
        boxes     = results.boxes.xyxy.cpu().numpy()
        cls_ids   = results.boxes.cls.cpu().numpy().astype(int)
        confs     = results.boxes.conf.cpu().numpy()
        track_ids = (results.boxes.id.cpu().numpy().astype(int)
                     if results.boxes.id is not None
                     else np.arange(len(boxes)))

        for box, cls_id, conf, tid in zip(boxes, cls_ids, confs, track_ids):
            # model.names works for both YOLO-World (custom vocab) and YOLOv8s (COCO)
            raw_label = model.names.get(cls_id, "")
            if not raw_label or raw_label not in ALL_CLASSES:
                continue

            x1 = int(box[0]*sx); y1 = int(box[1]*sy)
            x2 = int(box[2]*sx); y2 = int(box[3]*sy)

            # ── Pencil reclassification ──
            label = raw_label
            if label == "baseball bat":
                frac = box_area(x1, y1, x2, y2) / frame_area
                label = "pencil" if frac < 0.04 else label

            # (hand/arm reclassification removed — YOLO-World handles people correctly)

            cx_n = ((x1+x2)/2) / fw
            cy_n = ((y1+y2)/2) / fh
            dist_ft   = estimate_dist_ft(label, y2 - y1, fh, y1, y2)
            dist_zone = get_dist_zone(dist_ft)
            grid_zone = classify_zone(cx_n, cy_n)
            area      = box_area(x1, y1, x2, y2)
            tid_i     = int(tid)
            active.add(tid_i)

            mv = tracker.update(tid_i, cx_n, cy_n, area, dist_ft)

            # ── Projectile detection ──
            is_projectile = (
                mv["approach_rate"] >= PROJECTILE_RATE
                and dist_ft <= PROJECTILE_FT
                and mv["movement"] == "approaching"
            )
            collision = mv["collision"] or is_projectile
            is_benign = (label in BENIGN_SET) and not is_projectile

            display_label = label
            if is_projectile and label in BENIGN_SET:
                display_label = f"{label} (thrown)"

            draw_overlay(frame, display_label, dist_zone, mv["movement"],
                         collision, dist_ft, x1, y1, x2, y2, is_benign)

            record_detection(display_label, collision)

            dets.append({
                "tid":          tid_i,
                "label":        display_label,
                "grid_zone":    grid_zone,
                "zone":         dist_zone,
                "dist_ft":      dist_ft,
                "movement":     mv["movement"],
                "on_course":    mv["on_course"],
                "collision":    collision,
                "is_projectile":is_projectile,
                "benign":       is_benign,
                "conf":         round(float(conf), 2),
            })

    tracker.cleanup(active)

    # Sort by threat level
    dets.sort(key=lambda d: (
        0 if d["collision"] else
        1 if (not d["benign"] and d["zone"] == "near") else
        2 if (not d["benign"] and d["zone"] == "medium") else
        3 if not d["benign"] else 4
    ))

    _, buf  = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 84])
    out_b64 = base64.b64encode(buf.tobytes()).decode()

    return _json({"detections": dets, "frame": out_b64})


@app.route("/health")
def health():
    return _json({"ok": True, "model": _session_stats["model_name"],
                  "classes": _session_stats["class_count"]})


# ──────────────────────────── SERVER-SIDE TTS ────────────────────────────────
# Uses macOS `say` command — bypasses all browser audio restrictions.

_say_lock    = threading.Lock()   # only one utterance at a time
_say_proc    = None               # current say subprocess
_say_voice   = "Samantha"        # best macOS voice for clarity
_has_say     = shutil.which("say") is not None

def _do_say(text: str):
    """Speak text via macOS say, interrupting any current speech."""
    global _say_proc
    if not _has_say:
        return
    with _say_lock:
        # Kill current utterance so new priority alert isn't delayed
        if _say_proc and _say_proc.poll() is None:
            _say_proc.terminate()
        _say_proc = subprocess.Popen(
            ["say", "-v", _say_voice, text],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

@app.route("/speak", methods=["POST"])
def speak_route():
    """Browser posts {text, priority} here; server speaks via macOS say."""
    data = request.get_json(force=True)
    text = str(data.get("text", ""))[:300]   # cap length for safety
    if text:
        threading.Thread(target=_do_say, args=(text,), daemon=True).start()
    return _json({"ok": True})


def _json(obj):
    return app.response_class(json.dumps(obj), mimetype="application/json")


# ─────────────────────────── SESSION STATS ──────────────────────────────────

_stats_lock   = threading.Lock()
_session_stats = {
    "total_detections": 0,
    "total_collisions": 0,
    "class_counts":     {},   # label → count
    "start_time":       time.time(),
    "model_name":       "Loading…",
    "class_count":      0,
}

def record_detection(label: str, collision: bool):
    with _stats_lock:
        _session_stats["total_detections"] += 1
        if collision:
            _session_stats["total_collisions"] += 1
        _session_stats["class_counts"][label] = \
            _session_stats["class_counts"].get(label, 0) + 1

@app.route("/stats")
def stats():
    with _stats_lock:
        uptime = int(time.time() - _session_stats["start_time"])
        top5   = sorted(_session_stats["class_counts"].items(),
                        key=lambda x: -x[1])[:5]
        return _json({
            "total_detections": _session_stats["total_detections"],
            "total_collisions": _session_stats["total_collisions"],
            "uptime_s":         uptime,
            "top_objects":      top5,
            "model":            _session_stats["model_name"],
            "class_count":      _session_stats["class_count"],
        })

# ─────────────────────────── ENTRY POINT ────────────────────────────────────

def _load_model():
    global model
    # 1. Try YOLO-World v2 small (open-vocabulary, 80+ custom classes)
    if _HAS_WORLD:
        try:
            print("[INFO] Trying YOLO-World v2s (open-vocabulary)…")
            m = YOLOWorld("yolov8s-worldv2.pt")
            m.set_classes(WORLD_CLASSES)
            model = m
            _session_stats["model_name"] = "YOLO-World v2s"
            _session_stats["class_count"] = len(WORLD_CLASSES)
            print(f"[INFO] YOLO-World loaded — {len(WORLD_CLASSES)} custom classes active")
            return
        except Exception as e:
            print(f"[WARN] YOLO-World unavailable ({e})")

    # 2. Fall back to YOLOv8s (COCO-80)
    try:
        print("[INFO] Loading YOLOv8s…")
        model = YOLO("yolov8s.pt")
        _session_stats["model_name"] = "YOLOv8s"
        _session_stats["class_count"] = 80
        print("[INFO] YOLOv8s loaded — COCO-80 classes active")
        return
    except Exception as e:
        print(f"[WARN] YOLOv8s failed ({e})")

    # 3. Last resort: YOLOv8n
    print("[INFO] Loading YOLOv8n (fallback)…")
    model = YOLO("yolov8n.pt")
    _session_stats["model_name"] = "YOLOv8n"
    _session_stats["class_count"] = 80


if __name__ == "__main__":
    _load_model()
    print(f"[INFO] Guardian Eye v2.1 → http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
