"""
Driver Drowsiness Detection System
===================================
Uses Eye Aspect Ratio (EAR) to detect eye closure and trigger alerts.

Requirements:
    pip install opencv-python dlib imutils scipy numpy pygame
    Also needs dlib's shape predictor: shape_predictor_68_face_landmarks.dat
"""

import cv2
import dlib
import numpy as np
import time
import os
import sys
import threading
from scipy.spatial import distance as dist
from imutils import face_utils
from imutils.video import VideoStream
import imutils

# ─────────────────────────────────────────────
#  TRY to import pygame for alarm; fallback to
#  system beep if unavailable
# ─────────────────────────────────────────────
try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[WARN] pygame not found. Falling back to system beep.")


# ═══════════════════════════════════════════════════════
#  CONFIGURATION  — tweak these to suit your face / cam
# ═══════════════════════════════════════════════════════

EAR_THRESHOLD   = 0.25   # EAR below this → eye is "closed"
EAR_CONSEC_FRAMES = 20   # frames eye must stay closed → DROWSY alert
BLINK_FRAMES    = 4      # blinks shorter than this are ignored

# Path to dlib's pre-trained landmark predictor (68-point model)
PREDICTOR_PATH  = "shape_predictor_68_face_landmarks.dat"

# Landmark indices for left / right eye (dlib 68-point model)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# ═══════════════════════════════════════════════
#  ALARM  — generates a beep / plays alarm.wav
# ═══════════════════════════════════════════════

def generate_beep_wav(filename="alarm.wav", freq=880, duration=1.0, volume=0.9):
    """
    Generates a simple sine-wave WAV file so we don't need an external audio file.
    """
    import wave, struct, math
    sample_rate = 44100
    num_samples = int(sample_rate * duration)
    with wave.open(filename, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(num_samples):
            # Sine wave with a short fade-in/out to avoid clicks
            t = i / sample_rate
            fade = min(1.0, min(t, duration - t) / 0.05)
            sample = int(32767 * volume * fade * math.sin(2 * math.pi * freq * t))
            wf.writeframes(struct.pack("<h", sample))
    return filename


class Alarm:
    def __init__(self):
        self._playing  = False
        self._thread   = None
        self.wav_path  = None

        if PYGAME_AVAILABLE:
            self.wav_path = generate_beep_wav("alarm.wav")
            pygame.mixer.music.load(self.wav_path)

    def _play_loop(self):
        """Plays the alarm in a background thread."""
        while self._playing:
            if PYGAME_AVAILABLE:
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()
            else:
                # System bell fallback
                print("\a", end="", flush=True)
            time.sleep(0.8)

    def start(self):
        if not self._playing:
            self._playing = True
            self._thread  = threading.Thread(target=self._play_loop, daemon=True)
            self._thread.start()

    def stop(self):
        if self._playing:
            self._playing = False
            if PYGAME_AVAILABLE:
                pygame.mixer.music.stop()


# ═══════════════════════════════════════════
#  EYE ASPECT RATIO  (Soukupová & Čech 2016)
# ═══════════════════════════════════════════

def eye_aspect_ratio(eye):
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Six landmarks per eye:
        p1 ... p6  going clockwise from left corner
    Returns a float; ~0.3 open, <0.25 closed.
    """
    A = dist.euclidean(eye[1], eye[5])   # vertical #1
    B = dist.euclidean(eye[2], eye[4])   # vertical #2
    C = dist.euclidean(eye[0], eye[3])   # horizontal

    ear = (A + B) / (2.0 * C)
    return ear


# ═══════════════════
#  OVERLAY DRAWING
# ═══════════════════

def draw_eye_contour(frame, eye_pts, color):
    hull = cv2.convexHull(eye_pts)
    cv2.drawContours(frame, [hull], -1, color, 1)


def draw_hud(frame, ear, counter, total_blinks, status, alarm_on):
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # EAR value
    cv2.putText(frame, f"EAR: {ear:.3f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 230, 255), 2)

    # Threshold line indicator (small bar)
    bar_x, bar_y, bar_w, bar_h = 140, 8, 120, 14
    filled = int(bar_w * min(ear / 0.45, 1.0))
    color  = (0, 200, 80) if ear >= EAR_THRESHOLD else (0, 60, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), color, -1)
    # threshold marker
    thresh_x = bar_x + int(bar_w * (EAR_THRESHOLD / 0.45))
    cv2.line(frame, (thresh_x, bar_y - 2), (thresh_x, bar_y + bar_h + 2), (255, 255, 100), 2)

    # Blink count
    cv2.putText(frame, f"Blinks: {total_blinks}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Frame counter
    cv2.putText(frame, f"Closed: {counter}f", (140, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # ── STATUS BANNER ──
    if status == "DROWSY":
        # Flashing red banner
        alpha = 0.45 + 0.25 * abs((time.time() % 1.0) - 0.5) * 2
        banner = frame.copy()
        cv2.rectangle(banner, (0, h - 90), (w, h), (0, 0, 180), -1)
        cv2.addWeighted(banner, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(frame, "⚠  WAKE UP!", (w // 2 - 130, h - 28),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 3)
        cv2.putText(frame, "DROWSINESS DETECTED", (w // 2 - 145, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 80), 2)

    elif status == "ALERT":
        cv2.putText(frame, "ALERT", (w - 110, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 80), 2)
    elif status == "NO FACE":
        cv2.putText(frame, "NO FACE DETECTED", (w // 2 - 140, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (100, 160, 255), 2)


# ════════════════
#  MAIN LOOP
# ════════════════

def main():
    # ── Verify predictor file ──
    if not os.path.exists(PREDICTOR_PATH):
        print(f"""
[ERROR] Landmark predictor not found: '{PREDICTOR_PATH}'

Download it with:
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    bunzip2 shape_predictor_68_face_landmarks.dat.bz2

Then place it in the same folder as this script.
""")
        sys.exit(1)

    print("[INFO] Loading face detector and landmark predictor …")
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    alarm = Alarm()

    print("[INFO] Starting video stream …")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)   # camera warm-up

    counter      = 0   # consecutive frames with closed eyes
    total_blinks = 0
    status       = "ALERT"
    alarm_on     = False

    print("[INFO] Press  Q  to quit.")

    while True:
        frame = vs.read()
        if frame is None:
            print("[ERROR] Cannot read from camera.")
            break

        frame = imutils.resize(frame, width=720)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray, 0)

        if len(faces) == 0:
            status = "NO FACE"
            alarm.stop()
            alarm_on = False
            draw_hud(frame, 0.0, counter, total_blinks, status, alarm_on)
        else:
            status = "ALERT"
            for face in faces:
                shape = predictor(gray, face)
                shape = face_utils.shape_to_np(shape)

                left_eye  = shape[lStart:lEnd]
                right_eye = shape[rStart:rEnd]

                left_EAR  = eye_aspect_ratio(left_eye)
                right_EAR = eye_aspect_ratio(right_eye)
                ear        = (left_EAR + right_EAR) / 2.0

                # Draw eye contours
                eye_open_color   = (0, 220, 80)
                eye_closed_color = (0, 60, 255)
                color = eye_open_color if ear >= EAR_THRESHOLD else eye_closed_color
                draw_eye_contour(frame, left_eye,  color)
                draw_eye_contour(frame, right_eye, color)

                # ── EAR logic ──
                if ear < EAR_THRESHOLD:
                    counter += 1

                    if counter >= EAR_CONSEC_FRAMES:
                        status = "DROWSY"
                        if not alarm_on:
                            alarm.start()
                            alarm_on = True
                else:
                    # Eyes just reopened
                    if BLINK_FRAMES <= counter < EAR_CONSEC_FRAMES:
                        total_blinks += 1      # counts genuine blinks

                    if alarm_on:
                        alarm.stop()
                        alarm_on = False

                    counter = 0

                draw_hud(frame, ear, counter, total_blinks, status, alarm_on)

        cv2.imshow("Driver Drowsiness Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    print("[INFO] Shutting down …")
    alarm.stop()
    vs.stop()
    cv2.destroyAllWindows()
    if PYGAME_AVAILABLE and os.path.exists("alarm.wav"):
        os.remove("alarm.wav")


if __name__ == "__main__":
    main()
