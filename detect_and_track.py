"""
detect_and_track.py
====================
Real-time drone detection + tracking using a custom-trained YOLO model (best.pt).

Features
--------
- Loads a custom YOLOv8 / YOLOv11 model (best.pt) via the ultralytics API.
- Real-time inference from a webcam OR a video file.
- Built-in Ultralytics multi-object tracking (ByteTrack by default, BoT-SORT optional)
  -> persistent tracking IDs across frames.
- Draws bounding boxes with class name, confidence score, and tracking ID.
- On-screen real-time FPS counter (smoothed).
- Optional output video saving (MP4).
- Clean CLI flags, sensible defaults, robust error handling.
- Optimized for real-time performance (half precision on GPU, frame skipping option,
  efficient drawing, no unnecessary copies).

Usage
-----
    # Webcam (default camera index 0)
    python detect_and_track.py --weights best.pt

    # Video file
    python detect_and_track.py --weights best.pt --source path/to/video.mp4

    # Save output video
    python detect_and_track.py --weights best.pt --save --output output.mp4

    # Use BoT-SORT instead of ByteTrack
    python detect_and_track.py --weights best.pt --tracker botsort.yaml

    # Lower confidence threshold / different image size
    python detect_and_track.py --weights best.pt --conf 0.35 --imgsz 640

Press 'q' or ESC in the display window to quit.

Author: Generated for drone-detection competition project.
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

try:
    import serial as pyserial
except ImportError:
    pyserial = None  # pyserial not installed; ESP32 sending will be disabled

try:
    from ultralytics import YOLO
except ImportError:
    print(
        "[ERROR] The 'ultralytics' package is not installed.\n"
        "Install it with: pip install -r requirements.txt"
    )
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Configuration / constants
# --------------------------------------------------------------------------- #

BOX_COLOR = (0, 255, 0)        # BGR - bounding box color
TEXT_COLOR = (255, 255, 255)   # BGR - label text color
FPS_COLOR = (0, 255, 255)      # BGR - FPS overlay color
LINE_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICKNESS = 2
FPS_SMOOTHING_WINDOW = 30       # number of frames used to smooth FPS display


# --------------------------------------------------------------------------- #
# Track confirmation / false-positive filtering
# --------------------------------------------------------------------------- #

class TrackValidator:
    """
    Persistent drone locking:
    - Each track ID must be seen for min_hits consecutive frames before it's
      "confirmed" (locked). Once locked, it stays locked forever — the box is
      always drawn even if the drone briefly disappears (occlusion, motion blur).
    - Every new drone that appears gets its own independent lock, so N drones
      produce N persistent boxes simultaneously.
    - A locked track is only released if it goes missing for max_missing_frames
      consecutive frames (configurable, default=90 ≈ 3 seconds at 30 fps).
      This means short disappearances don't drop the lock.
    - Shape/area sanity check still runs during the confirmation phase to
      prevent locking onto a one-frame false positive.
    """

    def __init__(self, min_hits: int = 3, max_age: int = 10,
                 min_area_ratio: float = 0.00005, max_area_ratio: float = 0.6,
                 min_aspect: float = 0.15, max_aspect: float = 6.0,
                 max_missing_frames: int = 90,
                 debug: bool = False):
        self.min_hits = min_hits
        self.max_age = max_age                        # kept for CLI compat (unused below)
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.max_missing_frames = max_missing_frames  # frames before a locked track is dropped
        self.debug = debug

        # Per-track state
        self._hits: dict[int, int] = {}          # track_id -> confirmation hit count
        self._locked: set[int] = set()           # confirmed/locked track IDs
        self._missing: dict[int, int] = {}       # track_id -> consecutive missing-frame count
        self._last_box: dict[int, tuple] = {}    # track_id -> last known (x1,y1,x2,y2,conf,cls)

    # ---------------------------------------------------------------------- #
    def shape_is_plausible(self, x1, y1, x2, y2, frame_w, frame_h) -> bool:
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        area_ratio = (w * h) / float(frame_w * frame_h)
        aspect = w / float(h)
        if not (self.min_area_ratio <= area_ratio <= self.max_area_ratio):
            return False
        if not (self.min_aspect <= aspect <= self.max_aspect):
            return False
        return True

    # ---------------------------------------------------------------------- #
    def update_and_filter(self, xyxy, confs, clss, ids, frame_w, frame_h):
        """
        Returns keep_mask (bool list) aligned with the input detections PLUS
        a list of ghost entries for locked tracks that didn't appear this frame.

        Return value: (keep_mask, ghost_boxes)
          keep_mask  – bool list, same length as xyxy/confs/clss/ids
          ghost_boxes – list of (x1,y1,x2,y2,conf,cls_id,track_id) for locked
                        tracks that are currently missing but still shown
        """
        seen_this_frame: set[int] = set()
        keep_mask: list[bool] = []

        # ── Process detections from the tracker ──────────────────────────── #
        for (x1, y1, x2, y2), conf, cls_id, track_id in zip(xyxy, confs, clss, ids):
            if track_id is None:
                keep_mask.append(False)
                if self.debug:
                    print(f"[DEBUG] conf={conf:.2f} REJECTED (no track id)")
                continue

            tid = int(track_id)
            seen_this_frame.add(tid)
            self._missing[tid] = 0   # reset missing counter — it's here this frame

            plausible = self.shape_is_plausible(x1, y1, x2, y2, frame_w, frame_h)

            if self.debug:
                w, h_ = max(1, x2 - x1), max(1, y2 - y1)
                ar = (w * h_) / float(frame_w * frame_h)
                asp = w / float(h_)
                print(f"[DEBUG] ID={tid} conf={conf:.2f} area={ar:.5f} aspect={asp:.2f} "
                      f"plausible={plausible} locked={tid in self._locked}")

            if plausible:
                self._hits[tid] = self._hits.get(tid, 0) + 1
                if self._hits[tid] >= self.min_hits:
                    self._locked.add(tid)          # ← lock it permanently
            else:
                # Bad shape this frame — penalise but don't remove a lock
                if tid not in self._locked:
                    self._hits[tid] = max(0, self._hits.get(tid, 0) - 1)

            if tid in self._locked:
                # Always keep locked tracks and update their last-known box
                self._last_box[tid] = (int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_id))
                keep_mask.append(True)
            else:
                keep_mask.append(False)

        # ── Handle locked tracks that didn't appear this frame (ghosts) ───── #
        ghost_boxes: list[tuple] = []
        for tid in list(self._locked):
            if tid not in seen_this_frame:
                self._missing[tid] = self._missing.get(tid, 0) + 1
                if self._missing[tid] > self.max_missing_frames:
                    # Gone too long — release the lock
                    self._locked.discard(tid)
                    self._hits.pop(tid, None)
                    self._missing.pop(tid, None)
                    self._last_box.pop(tid, None)
                    if self.debug:
                        print(f"[DEBUG] ID={tid} LOCK RELEASED (missing {self.max_missing_frames}+ frames)")
                else:
                    # Still within grace period — draw last known position
                    if tid in self._last_box:
                        ghost_boxes.append((*self._last_box[tid], tid))

        # ── Age out unconfirmed tracks ────────────────────────────────────── #
        for tid in list(self._hits.keys()):
            if tid not in seen_this_frame and tid not in self._locked:
                self._missing[tid] = self._missing.get(tid, 0) + 1
                if self._missing[tid] > self.max_age:
                    self._hits.pop(tid, None)
                    self._missing.pop(tid, None)

        return keep_mask, ghost_boxes


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time drone detection and tracking using a custom YOLO model."
    )
    parser.add_argument(
        "--weights", type=str, default="best.pt",
        help="Path to the trained YOLO model weights (default: best.pt)"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source. Use '0' (or any integer) for webcam, or a path to a video file. (default: 0)"
    )
    parser.add_argument(
        "--tracker", type=str, default="bytetrack.yaml",
        help="Ultralytics tracker config: 'bytetrack.yaml' or 'botsort.yaml' (default: bytetrack.yaml)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.4,
        help="Confidence threshold for detections (default: 0.4)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.5,
        help="IoU threshold for NMS (default: 0.5)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference image size (default: 640)"
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="Device to run inference on: 'cpu', '0' (GPU index), etc. Empty = auto-select (default: auto)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save the annotated output video to disk."
    )
    parser.add_argument(
        "--output", type=str, default="output.mp4",
        help="Output video file path when --save is used (default: output.mp4)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable the live display window (useful for headless servers)."
    )
    parser.add_argument(
        "--max-width", type=int, default=1280,
        help="Resize display/output frame width for performance (0 = no resize) (default: 1280)"
    )
    parser.add_argument(
        "--min-hits", type=int, default=3,
        help="Consecutive frames a track must be confirmed before it's drawn (default: 3)"
    )
    parser.add_argument(
        "--max-age", type=int, default=10,
        help="Frames a track can be missing before it's forgotten (default: 10)"
    )
    parser.add_argument(
        "--min-area-ratio", type=float, default=0.00005,
        help="Min (box area / frame area) accepted as a plausible drone (default: 0.00005)"
    )
    parser.add_argument(
        "--max-area-ratio", type=float, default=0.6,
        help="Max (box area / frame area) accepted as a plausible drone (default: 0.6)"
    )
    parser.add_argument(
        "--min-aspect", type=float, default=0.15,
        help="Min box width/height aspect ratio accepted (default: 0.15)"
    )
    parser.add_argument(
        "--max-aspect", type=float, default=6.0,
        help="Max box width/height aspect ratio accepted (default: 6.0)"
    )
    parser.add_argument(
        "--max-missing-frames", type=int, default=90,
        help="Frames a LOCKED drone can be missing before its lock is released "
             "(default: 90 ≈ 3 s at 30 fps). Increase to keep the box longer."
    )
    parser.add_argument(
        "--no-filter", action="store_true",
        help="Disable track confirmation / shape filtering (raw model output, for debugging only)."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print raw detection info (conf, track id, area_ratio, aspect, kept/rejected) to the console every frame."
    )
    parser.add_argument(
        "--serial-port", type=str, default="COM5",
        help="Serial port the ESP32 is connected to (default: COM5). "
             "Examples: COM4 (Windows), /dev/ttyUSB0 or /dev/ttyACM0 (Linux)."
    )
    parser.add_argument(
        "--serial-baud", type=int, default=115200,
        help="Baud rate for the ESP32 serial link (default: 115200, must match the .ino sketch)."
    )
    parser.add_argument(
        "--no-serial", action="store_true",
        help="Disable sending coordinates to the ESP32 over serial (useful for testing on a PC with no board attached)."
    )
    parser.add_argument(
        "--serial-rate", type=float, default=20.0,
        help="Max rate (Hz) at which coordinates are sent to the ESP32 (default: 20). "
             "Prevents flooding the serial link faster than the ESP32 control loop needs."
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def resolve_source(source: str):
    """
    Allow the user to pass '0', '1', etc. for webcam indices, or a file path
    for video files. Returns the correct type for cv2.VideoCapture.
    """
    if source.isdigit():
        return int(source)
    return source


def draw_label(frame, text, x1, y1, color=BOX_COLOR):
    """
    Draws a filled label background plus text above the bounding box,
    clamped inside the frame so it never gets cut off at the top.
    """
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    label_y1 = max(0, y1 - text_h - baseline - 6)
    label_y2 = label_y1 + text_h + baseline + 6

    cv2.rectangle(frame, (x1, label_y1), (x1 + text_w + 6, label_y2), color, thickness=-1)
    cv2.putText(
        frame, text, (x1 + 3, label_y2 - baseline - 2),
        FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA
    )


GHOST_BOX_COLOR = (0, 200, 255)   # BGR - orange-ish for "ghost" (last-known position)


# --------------------------------------------------------------------------- #
# ESP32 serial communication
# --------------------------------------------------------------------------- #

class ESP32SerialLink:
    """
    Sends the tracked drone's center coordinates to the ESP32 over serial
    in the "x,y,detected_flag\n" format expected by the Arduino sketch
    (ESP32_DualServo_DroneTracking.ino):
        "300,220,1"  -> drone detected at (300,220)
        "0,0,0"      -> no drone detected this frame

    - Opens the port once; if opening fails (board not plugged in / wrong
      port), tracking still runs normally, just without serial output.
    - Coordinates are rate-limited (default 20 Hz) so the ESP32's control
      loop isn't flooded faster than it can read.
    - The "no detection" line (0,0,0) is sent just like a coordinate
      line — the ESP32 reacts to detected_flag immediately rather than
      relying on a serial timeout, so it stops the moment detection is
      lost instead of waiting out a grace period.
    """

    def __init__(self, port: str, baud: int, send_hz: float = 20.0, enabled: bool = True):
        self.enabled = enabled and (pyserial is not None) and bool(port)
        self.port = port
        self.baud = baud
        self.min_interval = 1.0 / send_hz if send_hz > 0 else 0.0
        self._last_send_time = 0.0
        self.conn = None

        if not enabled:
            return

        if pyserial is None:
            print("[WARNING] 'pyserial' is not installed — ESP32 serial output disabled.\n"
                  "          Install it with: pip install pyserial")
            self.enabled = False
            return

        try:
            self.conn = pyserial.Serial(port, baud, timeout=0)
            # Give the ESP32 a moment to reset after the port opens (typical
            # behavior on most boards when DTR toggles).
            time.sleep(2.0)
            print(f"[INFO] Connected to ESP32 on {port} @ {baud} baud.")
        except Exception as e:
            print(f"[WARNING] Could not open serial port '{port}': {e}\n"
                  f"          Continuing WITHOUT ESP32 serial output.")
            self.enabled = False
            self.conn = None

    def send_coords(self, x: int, y: int, detected: int = 1):
        """Send a single 'x,y,detected_flag\\n' line, rate-limited.

        detected=1 means a drone was found this frame (x,y is its center).
        detected=0 means no drone was found (x,y are sent as 0,0 by
        convention but are ignored by the ESP32 sketch in that case).
        """
        if not self.enabled or self.conn is None:
            return

        now = time.time()
        if (now - self._last_send_time) < self.min_interval:
            return  # too soon, skip this one to respect the rate limit

        try:
            line = f"{int(x)},{int(y)},{int(bool(detected))}\n".encode("ascii")
            self.conn.write(line)
            self._last_send_time = now
        except Exception as e:
            print(f"[WARNING] Serial write failed: {e} — disabling ESP32 output.")
            self.enabled = False

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass


def pick_primary_target(xyxy, ids, keep_mask, ghost_boxes):
    """
    Chooses which tracked drone's center to send to the ESP32 this frame.

    Priority:
      1. Among currently-kept (locked & confirmed) live detections, pick the
         one with the largest box area (closest / most prominent target).
      2. If none are live this frame, fall back to a ghost box (last-known
         position of a locked track) so the turret keeps pointing roughly
         the right way during a brief occlusion.
      3. If nothing is tracked at all, return None (no target).

    Returns (cx, cy) tuple or None.
    """
    best_area = -1
    best_center = None

    for (x1, y1, x2, y2), track_id, keep in zip(xyxy, ids, keep_mask):
        if not keep:
            continue
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > best_area:
            best_area = area
            best_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

    if best_center is not None:
        return best_center

    if ghost_boxes:
        # Pick the largest ghost box as a fallback
        gx1, gy1, gx2, gy2, _, _, _ = max(
            ghost_boxes, key=lambda b: max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        )
        return (int((gx1 + gx2) / 2), int((gy1 + gy2) / 2))

    return None


def draw_detections(frame, result, class_names, validator: "TrackValidator" = None):
    """
    Draws bounding boxes + labels for confirmed/locked drones.

    Locking behaviour (when validator is active):
      - A drone must appear for min_hits frames before its box is drawn.
      - Once locked, the box is drawn EVERY frame, even if the tracker
        momentarily loses it (shown in orange as "ghost" / last-known position).
      - Multiple drones are all locked and drawn independently.
      - A lock is only released after max_missing_frames consecutive frames
        of absence (~3 seconds at 30 fps by default).
    """
    frame_h, frame_w = frame.shape[:2]
    ghost_boxes: list[tuple] = []
    target_center = None

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes
        xyxy  = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss  = boxes.cls.cpu().numpy().astype(int)
        has_ids = boxes.id is not None
        ids = boxes.id.cpu().numpy().astype(int) if has_ids else [None] * len(xyxy)

        if validator is not None:
            keep_mask, ghost_boxes = validator.update_and_filter(
                xyxy, confs, clss, ids, frame_w, frame_h
            )
        else:
            keep_mask = [True] * len(xyxy)

        # Draw live detections
        for (x1, y1, x2, y2), conf, cls_id, track_id, keep in zip(
                xyxy, confs, clss, ids, keep_mask):
            if not keep:
                continue
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, LINE_THICKNESS)
            class_name = class_names.get(cls_id, str(cls_id))
            draw_label(frame, f"{class_name} {conf:.2f}", x1, y1, BOX_COLOR)

        target_center = pick_primary_target(xyxy, ids, keep_mask, ghost_boxes)

    elif validator is not None:
        # No detections this frame — still need to update ghost state
        _, ghost_boxes = validator.update_and_filter([], [], [], [], frame_w, frame_h)
        target_center = pick_primary_target([], [], [], ghost_boxes)

    # Draw ghost boxes (locked tracks missing this frame)
    # Ghost boxes (stale last-known position) are intentionally NOT drawn —
    # the box should only appear while the drone is actually detected this frame.

    return frame, target_center


def draw_fps(frame, fps: float):
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (10, 30), FONT, 0.8, FPS_COLOR, 2, cv2.LINE_AA)


def resize_for_display(frame, max_width: int):
    if max_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    new_size = (max_width, int(h * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[ERROR] Model weights not found at: {weights_path.resolve()}")
        sys.exit(1)

    print(f"[INFO] Loading model: {weights_path}")
    model = YOLO(str(weights_path))
    class_names = model.names  # dict: {0: 'drone', ...}
    print(f"[INFO] Model classes: {class_names}")

    source = resolve_source(args.source)
    is_webcam = isinstance(source, int)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {args.source}")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1 or src_fps > 240:
        src_fps = 30.0  # sane fallback (webcams often report 0)

    writer = None
    if args.save:
        # Grab one frame to determine output size after optional resizing.
        ret, probe_frame = cap.read()
        if not ret:
            print("[ERROR] Could not read from source to initialize video writer.")
            sys.exit(1)
        probe_frame = resize_for_display(probe_frame, args.max_width)
        h, w = probe_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, src_fps, (w, h))
        if not writer.isOpened():
            print(f"[ERROR] Could not open video writer for: {args.output}")
            sys.exit(1)
        # Put the probe frame back by re-opening capture position isn't possible
        # for webcams, so we just process this frame instead of discarding it.
        cap_first_frame = probe_frame
        print(f"[INFO] Saving annotated output to: {args.output}")
    else:
        cap_first_frame = None

    print("[INFO] Starting inference. Press 'q' or ESC to quit.")

    fps_buffer = deque(maxlen=FPS_SMOOTHING_WINDOW)
    prev_time = time.time()
    frame_idx = 0

    validator = None
    if not args.no_filter:
        validator = TrackValidator(
            min_hits=args.min_hits,
            max_age=args.max_age,
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio,
            min_aspect=args.min_aspect,
            max_aspect=args.max_aspect,
            max_missing_frames=args.max_missing_frames,
            debug=args.debug,
        )

    track_kwargs = dict(
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        tracker=args.tracker,
        persist=True,
        verbose=False,
    )
    if args.device:
        track_kwargs["device"] = args.device

    # --- ESP32 serial link setup ---------------------------------------- #
    esp32 = ESP32SerialLink(
        port=args.serial_port,
        baud=args.serial_baud,
        send_hz=args.serial_rate,
        enabled=not args.no_serial,
    )
    # Reference frame size the ESP32 sketch expects (640x480, center 320,240).
    # If the actual processed frame is a different size (e.g. due to
    # --max-width resizing), coordinates are rescaled to this reference
    # before being sent, so the ESP32-side dead zone / PID math stays valid.
    ESP32_REF_WIDTH, ESP32_REF_HEIGHT = 640, 480

    try:
        while True:
            if cap_first_frame is not None:
                frame = cap_first_frame
                cap_first_frame = None
                ret = True
            else:
                ret, frame = cap.read()

            if not ret or frame is None:
                if is_webcam:
                    print("[WARNING] Frame grab failed, retrying...")
                    continue
                print("[INFO] End of video stream reached.")
                break

            frame = resize_for_display(frame, args.max_width)

            # Run tracking-enabled inference (Ultralytics built-in ByteTrack/BoT-SORT)
            results = model.track(frame, **track_kwargs)
            result = results[0]

            annotated, target_center = draw_detections(frame, result, class_names, validator)

            # Send the tracked drone's state to the ESP32 every frame
            # (rescaled to the 640x480 reference frame the Arduino sketch
            # expects). When nothing is tracked, explicitly send
            # detected=0 so the ESP32 stops immediately instead of
            # waiting out a timeout.
            if target_center is not None:
                fh, fw = frame.shape[:2]
                cx, cy = target_center
                scaled_x = int(cx * (ESP32_REF_WIDTH / fw))
                scaled_y = int(cy * (ESP32_REF_HEIGHT / fh))
                esp32.send_coords(scaled_x, scaled_y, detected=1)
            else:
                esp32.send_coords(0, 0, detected=0)

            # FPS calculation (smoothed over a rolling window)
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps_buffer.append(1.0 / dt)
            current_fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0.0
            draw_fps(annotated, current_fps)

            if writer is not None:
                writer.write(annotated)

            if not args.no_display:
                cv2.imshow("Drone Detection & Tracking", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or ESC
                    print("[INFO] Quit signal received.")
                    break

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        cap.release()
        esp32.close()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        print(f"[INFO] Processed {frame_idx} frames. Shutting down cleanly.")


if __name__ == "__main__":
    main()