# 🎯 SkyLock — Real-Time Drone Detection & Tracking Turret

**SkyLock** detects and tracks drones in real time using a custom-trained **YOLO** model with **ByteTrack**, then physically follows the target with a **pan/tilt camera turret** driven by an **ESP32** and two servos.

The PC runs the vision pipeline and streams the target's pixel coordinates over serial; the ESP32 turns them into smooth servo motion that keeps the drone centered in the frame.

```
 Camera / Video ──▶ YOLO (best.pt) ──▶ ByteTrack ──▶ Track Validator ──▶ Primary Target
                                                                              │
                                                       "x,y,flag\n" @ 20 Hz   ▼
                                    Pan servo (360°) + Tilt servo (180°) ◀── ESP32
```

---

## ✨ Features

### 🧠 Vision (Python)
- Custom YOLOv8 / YOLOv11 model (`best.pt`) via the Ultralytics API
- Persistent multi-object tracking — **ByteTrack** (default) or **BoT-SORT**
- **Track validation** to suppress false positives:
  - must be seen for N consecutive frames before it is "locked"
  - box area and aspect-ratio sanity checks
  - locked tracks stay on screen through short occlusions / motion blur (ghost boxes) and are released only after a configurable number of missed frames
- Each drone gets its own independent lock → multiple drones supported
- Annotated output: class, confidence, track ID, smoothed FPS counter
- Webcam or video-file input, optional MP4 saving, headless mode
- GPU half-precision inference when available

### 🔧 Turret control (ESP32)
- **Base (pan) servo** — 360° continuous rotation, speed-controlled with a PD controller
- **Tilt servo** — 180° positional, with capped per-loop steps for smooth, cinematic motion
- EMA smoothing of incoming coordinates and a dead zone around frame center to prevent jitter
- **Safety**: pan servo hard-stops when nothing is detected, and a 1 s serial-link timeout stops it if the USB cable drops
- Boot self-test and manual test commands over serial (`T`, `B`, `S`)

---

## 🔩 Hardware

| Part | Details |
|---|---|
| Microcontroller | ESP32 (Arduino framework) |
| Pan servo | 360° continuous-rotation servo → **GPIO 15** |
| Tilt servo | 180° positional servo → **GPIO 2** |
| Camera | USB webcam mounted on the tilt servo, tilt servo mounted on the pan servo |
| Host | PC / Jetson / Raspberry Pi running the Python script |

> ⚡ Power the servos from a suitable external supply (shared GND with the ESP32), not from the ESP32's 3.3 V pin.

---

## 📡 Serial Protocol

PC → ESP32, one ASCII line per update, `115200` baud:

```
x,y,detected_flag\n
```

| Example | Meaning |
|---|---|
| `300,220,1` | Drone detected, center at pixel (300, 220) |
| `0,0,0` | No drone this frame → pan stops, tilt holds |

Frame size is assumed to be **640×480** (center = 320, 240).

---

## 🚀 Getting Started

### 1. Python side

```bash
git clone https://github.com/<your-username>/skylock.git
cd skylock
pip install -r requirements.txt
```

> `pyserial` is needed for ESP32 output: `pip install pyserial` (add it to `requirements.txt`).

Run:

```bash
# Webcam + ESP32 on COM5 (Windows)
python detect_and_track.py --weights best.pt --serial-port COM5

# Linux
python detect_and_track.py --weights best.pt --serial-port /dev/ttyUSB0

# Video file, no hardware attached
python detect_and_track.py --weights best.pt --source path/to/video.mp4 --no-serial

# Save annotated video
python detect_and_track.py --weights best.pt --save --output output.mp4
```

Press **`q`** or **`Esc`** to quit.

### 2. ESP32 side

1. Install the **ESP32Servo** library (Arduino Library Manager).
2. Open `ESP32_DualServo_DroneTracking.ino`, select your ESP32 board and port, and upload.
3. Open the Serial Monitor at `115200` baud and send a single character to test:
   - `T` — tilt sweep
   - `B` — pan spin test
   - `S` — stop everything
4. **Calibrate** `BASE_STOP_TRIM` if the pan servo creeps at rest, and flip `TILT_DIRECTION` if the tilt axis is reversed.

---

## ⚙️ Command-Line Options

| Flag | Default | Description |
|---|---|---|
| `--weights` | `best.pt` | Model weights |
| `--source` | `0` | Webcam index or video path |
| `--tracker` | `bytetrack.yaml` | `bytetrack.yaml` or `botsort.yaml` |
| `--conf` / `--iou` | `0.4` / `0.5` | Detection / NMS thresholds |
| `--imgsz` | `640` | Inference size |
| `--device` | auto | `cpu`, `0`, ... |
| `--min-hits` | `3` | Frames needed to confirm a track |
| `--max-missing-frames` | `90` | Frames a locked drone may be missing (~3 s @ 30 fps) |
| `--min/max-area-ratio` | `0.00005` / `0.6` | Plausible box size relative to frame |
| `--min/max-aspect` | `0.15` / `6.0` | Plausible box aspect ratio |
| `--no-filter` | off | Raw model output (debug) |
| `--serial-port` | `COM5` | ESP32 serial port |
| `--serial-baud` | `115200` | Must match the `.ino` |
| `--serial-rate` | `20` | Max coordinate updates per second |
| `--no-serial` | off | Disable ESP32 output |
| `--save` / `--output` | off / `output.mp4` | Save annotated video |
| `--no-display` | off | Headless mode |
| `--max-width` | `1280` | Display/output width |
| `--debug` | off | Per-frame detection logging |

---

## 🗂️ Project Structure

```
skylock/
├── detect_and_track.py                  # Detection + tracking + serial output
├── best.pt                              # Trained YOLO weights (see note below)
├── requirements.txt
└── esp32/
    └── ESP32_DualServo_DroneTracking.ino   # Pan/tilt controller firmware
```

> ⚠️ **`best.pt` is ~114 MB — larger than GitHub's 100 MB file limit.** Use [Git LFS](https://git-lfs.com) (`git lfs track "*.pt"`) or attach it to a GitHub Release and link to it here.

---

## 🛠️ Troubleshooting

- **"Could not open serial port"** — check the port name and that no other program (Arduino Serial Monitor) is using it. Tracking still runs without it.
- **Pan servo keeps rotating at rest** — tune `BASE_STOP_TRIM` in the sketch.
- **Servo barely moves** — raise `BASE_MIN_SPEED` / `TILT_GAIN`.
- **Turret moves the wrong way** — set `TILT_DIRECTION = -1`, or swap the pan servo's direction.
- **Low FPS** — use a GPU (`--device 0`), reduce `--imgsz`, or lower `--max-width`.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

## 📄 License

Add a license of your choice (e.g. [MIT](https://choosealicense.com/licenses/mit/)) and update this section.
