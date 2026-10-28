/*
  ====================================================================
  ESP32 Dual-Servo Drone Tracking System  (explicit detection-flag protocol)
  ====================================================================
  Hardware:
    - Base Servo  (360° continuous rotation) -> GPIO 15  (horizontal/X, pan)
    - Tilt Servo  (180° positional)          -> GPIO 2   (vertical/Y, tilt)
    - Camera mounted on tilt servo, tilt servo mounted on base servo

  Serial Input Format (from PC/Jetson/RPi running the detection model):
      "x,y,detected_flag\n"
        300,220,1   -> drone detected at pixel (300,220)
        0,0,0       -> no drone detected this frame

  Frame size: 640x480  -> Center: (320, 240)

  CORE BEHAVIOR RULES
  --------------------
    1. detected_flag = 0  -> Base servo: HARD STOP (no PWM change other
       than re-asserting the stop pulse). Tilt servo: HOLD last angle,
       no new writes at all.
    2. detected_flag = 1  -> Full tracking enabled: base servo follows
       X error, tilt servo follows Y error, both with smoothing.
    3. A serial-link safety timeout (SERIAL_LINK_TIMEOUT_MS) forces the
       system into the "no detection" state if NO line arrives at all
       for that long — protects against a dead/unplugged USB cable
       silently leaving the base servo spinning forever.
    4. A dead zone around frame center on both axes prevents jitter
       when the drone is already "close enough" to centered.
    5. An EMA filter smooths the raw incoming coordinates before they
       reach either controller, and the tilt servo additionally moves
       in capped per-loop steps for stable, non-jittery (cinematic)
       motion instead of snapping to a new angle instantly.

  Code organization (each concern is its own function):
    - Serial parsing:            readAndParseSerial(), parseDetectionLine()
    - Detection state handling:  updateDetectionState(), onTrackingActivated(),
                                  onTrackingDeactivated()
    - Smoothing/filtering:       applyEMA(), resetSmoothing()
    - Base servo control (360): controlBaseServo(), stopBaseServo()
    - Tilt servo control (180): controlTiltServo(), holdTiltServo()
    - Manual hardware self-test: testTiltSweep(), testBaseSpin(), testStopAll()
      (send a single 'T' / 'B' / 'S' character over Serial any time to
      verify the servos move correctly, independent of the detection PC)

  Library required: ESP32Servo (install via Arduino Library Manager)
  ====================================================================
*/

#include <ESP32Servo.h>

// =====================================================================
// Pin & Frame Configuration
// =====================================================================
#define BASE_SERVO_PIN   15     // 360° continuous rotation servo (pan)
#define TILT_SERVO_PIN   2      // 180° positional servo (tilt)

const int FRAME_WIDTH    = 640;
const int FRAME_HEIGHT   = 480;
const int FRAME_CENTER_X = FRAME_WIDTH  / 2;   // 320
const int FRAME_CENTER_Y = FRAME_HEIGHT / 2;   // 240

// =====================================================================
// Dead Zone (pixels) — errors smaller than this are treated as "centered"
// =====================================================================
const int DEAD_ZONE_X = 15;
const int DEAD_ZONE_Y = 12;

// =====================================================================
// Base (pan) speed-control gains — output maps to rotation SPEED
// =====================================================================
const float Kp_base = 0.14;
const float Kd_base = 0.020;

// =====================================================================
// Base servo motion limits — 90 is BOTH the center AND the "stop" pulse
// for a continuous-rotation servo. This is the single most important
// calibration value in this sketch.
// =====================================================================
const int BASE_STOP_PULSE = 90;   // "stop"/center angle for the 360° servo
const int BASE_STOP_TRIM  = 0;    // <-- CALIBRATE: if it still creeps at
                                   //     90, nudge by +/-1..5 until still.
const int BASE_MAX_SPEED  = 55;   // max +/- offset from 90 (limits speed)
const int BASE_MIN_SPEED  = 18;   // min offset needed to overcome the
                                   // servo's mechanical dead-band (tune!)

// =====================================================================
// Tilt servo motion limits & gains
// Tune TILT_GAIN / TILT_MAX_STEP_PER_LOOP together:
//   - Higher TILT_GAIN / higher step cap  -> snappier, more visible, but
//     can feel jittery/aggressive on some servos.
//   - Lower values -> smoother, more "cinematic", but slower to react.
// The defaults below aim for a stable middle ground; adjust to taste.
// =====================================================================
const int   TILT_MIN_ANGLE   = 0;
const int   TILT_MAX_ANGLE   = 180;
const int   TILT_START_ANGLE = 90;    // level/center position at boot
const int   TILT_DIRECTION   = 1;     // 1 = normal, -1 = reversed mounting
const float TILT_GAIN        = 0.35;  // degrees of correction per pixel of error
const float TILT_MIN_STEP    = 3.0;   // minimum visible step per loop when moving
const float TILT_MAX_STEP_PER_LOOP = 15.0; // max degrees tilt can move per loop (smoothing cap)

// =====================================================================
// EMA Filter smoothing factor (0 < alpha <= 1)
// Lower alpha = smoother but slower response, higher = snappier but noisier
// =====================================================================
const float EMA_ALPHA = 0.30;

// =====================================================================
// Serial-link safety timeout — if NO line at all arrives for this long,
// force "no detection" regardless of the last flag received (protects
// against a dropped/unplugged serial connection).
// =====================================================================
const unsigned long SERIAL_LINK_TIMEOUT_MS = 1000;

// =====================================================================
// Control loop timing
// =====================================================================
const unsigned long LOOP_INTERVAL_MS = 10; // ~100Hz control loop

// =====================================================================
// Manual hardware self-test
//   - RUN_BOOT_SELFTEST: sweeps both servos once at power-up so you can
//     visually confirm wiring/power WITHOUT the PC connected at all.
//   - At any time, open Serial Monitor (115200 baud) and send a single
//     character + Enter:
//       T -> tilt sweep (90 -> 60 -> 120 -> 90)
//       B -> base spin test (one way -> stop -> other way -> stop)
//       S -> force-stop/hold both servos immediately
// =====================================================================
const bool RUN_BOOT_SELFTEST = true;

// =====================================================================
// Global Objects & State
// =====================================================================
Servo baseServo;
Servo tiltServo;

// Raw (last received) and EMA-filtered target coordinates
int   rawX = FRAME_CENTER_X, rawY = FRAME_CENTER_Y;
float emaX = FRAME_CENTER_X, emaY = FRAME_CENTER_Y;
bool  emaInitialized = false;

// Detection / tracking state
bool detectedFlag   = false;  // last explicit flag value received over serial
bool trackingActive = false;  // computed each loop: detectedFlag AND link not timed out
unsigned long lastSerialTime = 0; // last time ANY valid line was received

// Controller state
float baseErrorPrev = 0;
float tiltErrorPrev = 0;
float currentTiltAngle = TILT_START_ANGLE; // tilt is positional, so we track its angle

// Loop timing
unsigned long lastLoopTime = 0;

// Serial input buffer
String serialBuffer = "";

// =====================================================================
// SETUP
// =====================================================================
void setup() {
  Serial.begin(115200);
  Serial.println(F("[INIT] ESP32 Dual-Servo Drone Tracking — starting..."));

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  baseServo.setPeriodHertz(50);
  baseServo.attach(BASE_SERVO_PIN, 500, 2400);

  tiltServo.setPeriodHertz(50);
  tiltServo.attach(TILT_SERVO_PIN, 500, 2400);

  // Safe neutral state at boot: base STOPPED, tilt LEVEL.
  baseServo.write(BASE_STOP_PULSE + BASE_STOP_TRIM);
  tiltServo.write((int)currentTiltAngle);

  lastLoopTime   = millis();
  lastSerialTime = millis(); // start the timeout clock; system boots into "no detection"

  if (RUN_BOOT_SELFTEST) {
    runBootSelfTest();
  }

  Serial.println(F("[INIT] Ready. Waiting for \"x,y,detected_flag\" over serial."));
  Serial.println(F("[INIT] Manual test commands any time: 'T' tilt sweep, 'B' base spin, 'S' stop."));
}

// =====================================================================
// MAIN LOOP
// =====================================================================
void loop() {
  // 1) Always drain incoming serial bytes (non-blocking)
  readAndParseSerial();

  // 2) Run the control loop at a fixed interval
  unsigned long now = millis();
  if (now - lastLoopTime < LOOP_INTERVAL_MS) return;
  float dt = (now - lastLoopTime) / 1000.0f;
  lastLoopTime = now;

  updateDetectionState(now);

  if (trackingActive) {
    applyEMA();

    float errorX = emaX - FRAME_CENTER_X;
    float errorY = emaY - FRAME_CENTER_Y;

    controlBaseServo(errorX, dt);
    controlTiltServo(errorY, dt);

    printDebugInfo(errorX, errorY);
  } else {
    // No detection (or link down): base hard-stops every cycle (cheap
    // continuous-rotation servos can creep with a single stale write),
    // tilt simply holds — no command is sent to it at all.
    stopBaseServo();
    holdTiltServo();
  }
}

// =====================================================================
// SERIAL PARSING
// =====================================================================

// Drains the UART buffer one character at a time, accumulating a line
// until '\n'/'\r', then hands the complete line off for processing.
void readAndParseSerial() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      if (serialBuffer.length() > 0) {
        handleIncomingLine(serialBuffer);
        serialBuffer = "";
      }
    } else {
      serialBuffer += c;
      if (serialBuffer.length() > 40) {  // guard against garbage/runaway input
        serialBuffer = "";
      }
    }
  }
}

// Routes a complete line to either a manual single-letter test command
// (T/B/S) or to the normal "x,y,detected_flag" parser.
void handleIncomingLine(const String &rawLine) {
  String line = rawLine;
  line.trim();
  if (line.length() == 0) return;

  if (line.length() == 1) {
    char cmd = toupper(line.charAt(0));
    if (cmd == 'T') { testTiltSweep(); return; }
    if (cmd == 'B') { testBaseSpin();  return; }
    if (cmd == 'S') { testStopAll();   return; }
  }

  parseDetectionLine(line);
}

// Parses "x,y,detected_flag" — e.g. "300,220,1" or "0,0,0".
// Only commits the values if the whole line is well-formed; a malformed
// line is silently ignored (previous state is kept until the next valid
// line, or until SERIAL_LINK_TIMEOUT_MS forces a safe stop).
void parseDetectionLine(const String &line) {
  int comma1 = line.indexOf(',');
  if (comma1 <= 0) return;

  int comma2 = line.indexOf(',', comma1 + 1);
  if (comma2 <= comma1 + 1) return;

  String xStr = line.substring(0, comma1);
  String yStr = line.substring(comma1 + 1, comma2);
  String fStr = line.substring(comma2 + 1);
  xStr.trim(); yStr.trim(); fStr.trim();

  if (xStr.length() == 0 || yStr.length() == 0 || fStr.length() == 0) return;

  int x = xStr.toInt();
  int y = yStr.toInt();
  int f = fStr.toInt();

  // Reject obviously corrupt coordinates (out of the agreed frame range)
  if (x < 0 || x > FRAME_WIDTH || y < 0 || y > FRAME_HEIGHT) return;

  rawX = x;
  rawY = y;
  detectedFlag   = (f != 0);
  lastSerialTime = millis();
}

// =====================================================================
// DETECTION STATE HANDLING
// =====================================================================

// Combines the explicit detected_flag with a link-timeout safety net to
// decide whether tracking should be active this loop, and fires the
// activate/deactivate transition hooks exactly once on change.
void updateDetectionState(unsigned long now) {
  bool linkTimedOut = (now - lastSerialTime) > SERIAL_LINK_TIMEOUT_MS;
  bool wasActive = trackingActive;

  trackingActive = detectedFlag && !linkTimedOut;

  if (wasActive && !trackingActive) {
    onTrackingDeactivated();
  } else if (!wasActive && trackingActive) {
    onTrackingActivated();
  }
}

// Called once, the instant tracking turns ON (0 -> 1 transition).
void onTrackingActivated() {
  resetSmoothing();      // re-seed EMA to the current raw value, avoids a jump
  baseErrorPrev  = 0;
  tiltErrorPrev  = 0;
  Serial.println(F("[STATE] Drone detected -> tracking ENABLED."));
}

// Called once, the instant tracking turns OFF (1 -> 0 transition, or
// link timeout). Forces both servos into their safe "no detection" state.
void onTrackingDeactivated() {
  stopBaseServo();
  holdTiltServo();
  Serial.println(F("[STATE] No detection -> tracking DISABLED. Base stopped, tilt holding."));
}

// =====================================================================
// SMOOTHING / FILTERING LOGIC
// =====================================================================

// Exponential Moving Average filter on the raw incoming coordinates.
void applyEMA() {
  if (!emaInitialized) {
    emaX = rawX;
    emaY = rawY;
    emaInitialized = true;
  } else {
    emaX = EMA_ALPHA * rawX + (1.0f - EMA_ALPHA) * emaX;
    emaY = EMA_ALPHA * rawY + (1.0f - EMA_ALPHA) * emaY;
  }
}

// Forces the EMA filter to re-seed on the next applyEMA() call instead
// of smoothing from a stale value — prevents a visible "snap" artifact
// when tracking re-activates after a gap.
void resetSmoothing() {
  emaInitialized = false;
}

// =====================================================================
// BASE SERVO CONTROL (360° continuous rotation — SPEED actuator)
// 90 = stop, <90 = rotate one way, >90 = rotate the other way.
// Simple P + D speed control: no integral term, so it can't "wind up"
// and refuse to stop or refuse to reverse. Inside the dead zone it is
// forced to an exact, unconditional stop.
// =====================================================================
void controlBaseServo(float errorX, float dt) {
  int stopPulse = BASE_STOP_PULSE + BASE_STOP_TRIM;

  if (fabs(errorX) <= DEAD_ZONE_X) {
    baseServo.write(stopPulse);
    baseErrorPrev = errorX; // keep so D-term doesn't spike on next move
    return;
  }

  float derivative = (errorX - baseErrorPrev) / (dt > 0 ? dt : 0.001f);
  baseErrorPrev = errorX;

  float speed = (Kp_base * errorX) + (Kd_base * derivative);
  speed = constrain(speed, -BASE_MAX_SPEED, BASE_MAX_SPEED);

  // Deadband compensation so small-but-real errors still actually move
  // the servo instead of sitting in its mechanical dead-band.
  if (speed > 0 && speed < BASE_MIN_SPEED) speed = BASE_MIN_SPEED;
  if (speed < 0 && speed > -BASE_MIN_SPEED) speed = -BASE_MIN_SPEED;

  int pulse = stopPulse + (int)speed;
  pulse = constrain(pulse, stopPulse - BASE_MAX_SPEED, stopPulse + BASE_MAX_SPEED);
  baseServo.write(pulse);
}

// Unconditional hard stop — called every loop while tracking is inactive.
// Re-writing the same stop pulse every cycle (rather than once) matters
// for cheap continuous-rotation servos that can creep on a single stale
// PWM write.
void stopBaseServo() {
  baseServo.write(BASE_STOP_PULSE + BASE_STOP_TRIM);
  baseErrorPrev = 0;
}

// =====================================================================
// TILT SERVO CONTROL (180° positional — ANGLE actuator)
// Proportional response from the pixel error, moved in capped per-loop
// steps for smooth, stable ("cinematic") motion rather than snapping
// directly to the target angle.
//   - Object ABOVE center (errorY < 0) -> tilt one way.
//   - Object BELOW center (errorY > 0) -> tilt the other way.
//   - Flip TILT_DIRECTION to -1 if the motion is backwards on your rig.
// =====================================================================
void controlTiltServo(float errorY, float dt) {
  if (fabs(errorY) <= DEAD_ZONE_Y) {
    tiltErrorPrev = errorY;
    return; // centered vertically -> hold current angle, no movement
  }

  tiltErrorPrev = errorY;

  float correction = TILT_DIRECTION * TILT_GAIN * errorY;
  float desiredAngle = constrain(currentTiltAngle + correction, TILT_MIN_ANGLE, TILT_MAX_ANGLE);
  float angleDelta = desiredAngle - currentTiltAngle;

  // Guarantee a minimum visible step in the correct direction...
  if (angleDelta > 0 && angleDelta < TILT_MIN_STEP) angleDelta = TILT_MIN_STEP;
  if (angleDelta < 0 && angleDelta > -TILT_MIN_STEP) angleDelta = -TILT_MIN_STEP;
  // ...but never more than the per-loop cap, for smoothness.
  angleDelta = constrain(angleDelta, -TILT_MAX_STEP_PER_LOOP, TILT_MAX_STEP_PER_LOOP);

  currentTiltAngle += angleDelta;
  currentTiltAngle = constrain(currentTiltAngle, TILT_MIN_ANGLE, TILT_MAX_ANGLE);

  tiltServo.write((int)round(currentTiltAngle));
}

// Intentionally a no-op: a positional servo holds its last commanded
// angle on its own. We deliberately do NOT re-write it every cycle —
// repeated identical writes are what make some servos hum/jitter while
// "holding". currentTiltAngle is left untouched so tracking resumes
// smoothly from exactly where it left off.
void holdTiltServo() {
  // no-op by design
}

// =====================================================================
// DEBUG OUTPUT
// =====================================================================
void printDebugInfo(float errorX, float errorY) {
  static unsigned long lastPrint = 0;
  unsigned long now = millis();
  if (now - lastPrint < 200) return; // throttle to ~5x/sec
  lastPrint = now;

  Serial.print(F("[TRACK] raw=("));
  Serial.print(rawX);
  Serial.print(F(","));
  Serial.print(rawY);
  Serial.print(F(")  ema=("));
  Serial.print(emaX, 1);
  Serial.print(F(","));
  Serial.print(emaY, 1);
  Serial.print(F(")  errX="));
  Serial.print(errorX, 1);
  Serial.print(F("  errY="));
  Serial.print(errorY, 1);
  Serial.print(F("  tiltAngle="));
  Serial.print(currentTiltAngle, 1);
  Serial.print(F("  tracking="));
  Serial.println(trackingActive ? "ON" : "OFF");
}

// =====================================================================
// MANUAL HARDWARE SELF-TEST
// Never touches tracking state; only ever runs once at boot or when you
// manually type T / B / S into the Serial Monitor. Pure hardware sanity
// check — if these don't move the servos, the problem is wiring/power,
// not this code or the detection script.
// =====================================================================
void testTiltSweep() {
  Serial.println(F("[TEST] Tilt sweep: 90 -> 60 -> 120 -> 90"));
  tiltServo.write(90);  delay(500);
  tiltServo.write(60);  delay(600);
  tiltServo.write(120); delay(600);
  tiltServo.write(90);  delay(500);
  currentTiltAngle = 90;
  tiltErrorPrev = 0;
  Serial.println(F("[TEST] Tilt sweep done."));
}

void testBaseSpin() {
  int stopPulse = BASE_STOP_PULSE + BASE_STOP_TRIM;
  Serial.println(F("[TEST] Base spin: one way -> stop -> other way -> stop"));
  baseServo.write(stopPulse + 30); delay(700);
  baseServo.write(stopPulse);      delay(400);
  baseServo.write(stopPulse - 30); delay(700);
  baseServo.write(stopPulse);
  Serial.println(F("[TEST] Base spin done."));
}

void testStopAll() {
  baseServo.write(BASE_STOP_PULSE + BASE_STOP_TRIM);
  tiltServo.write((int)round(currentTiltAngle));
  Serial.println(F("[TEST] Both servos force-stopped/held."));
}

void runBootSelfTest() {
  Serial.println(F("[TEST] Running boot self-test (set RUN_BOOT_SELFTEST=false to skip)..."));
  testTiltSweep();
  testBaseSpin();
  Serial.println(F("[TEST] Boot self-test complete."));
}
