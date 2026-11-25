import argparse
import os
import subprocess
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from effects import CloudEffect
from gestures import GestureBook, normalize
from masking import LightMatcher, PersonMasker

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODELS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "selfie_segmenter.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
    "selfie_multiclass_256x256.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
}
EFFECTS = ("cloud", "dissolve", "fade")
BG_COUNTDOWN = 3.0          # seconds to leave the frame after pressing B
BG_CAPTURE_FRAMES = 45
RECORD_COUNTDOWN = 3.0      # seconds before a custom gesture starts recording
RECORD_SAMPLES = 40         # frames captured per custom gesture


def parse_args():
    ap = argparse.ArgumentParser(description="Vanish from your webcam with a hand gesture.")
    ap.add_argument("--camera", type=int, default=None,
                    help="physical camera index (default: first real, non-virtual webcam)")
    ap.add_argument("--list-cameras", action="store_true", help="print camera numbers and names, then exit")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--effect", choices=EFFECTS, default="cloud")
    ap.add_argument("--fade-speed", type=float, default=0.045, help="invisibility change per frame")
    ap.add_argument("--hold", type=int, default=10, help="frames a gesture must be held")
    ap.add_argument("--threshold", type=float, default=0.16, help="custom-gesture match strictness (lower = stricter)")
    ap.add_argument("--mask-threshold", type=float, default=0.2,
                    help="person-detection sensitivity: lower hides more of you (default 0.2)")
    ap.add_argument("--mask-grow", type=int, default=18, help="extra margin around you in px at half res (default 18)")
    ap.add_argument("--fast", action="store_true", help="skip the accurate hair/clothes model (lower CPU)")
    ap.add_argument("--no-light-match", action="store_true",
                    help="don't correct the background for camera exposure / lighting changes")
    ap.add_argument("--no-vcam", action="store_true", help="preview only, no virtual camera")
    ap.add_argument("--mirror-vcam", action="store_true",
                    help="send a mirrored image to the meeting (default: un-mirrored, like a normal webcam)")
    return ap.parse_args()


def ensure_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    paths = {}
    for name, url in MODELS.items():
        path = os.path.join(MODEL_DIR, name)
        if not os.path.exists(path):
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, path)
        paths[name] = path
    return paths


VIRTUAL_CAM_NAMES = ("obs virtual camera", "obs-camera", "unity video capture", "virtual")


def list_cameras():
    """DirectShow camera names in OpenCV index order (Windows); empty list if unknown."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        return FilterGraph().get_input_devices()
    except Exception:  # noqa: BLE001
        return []


def obs_running():
    if os.name != "nt":
        return False
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq obs64.exe"],
                             capture_output=True, text=True, timeout=5).stdout
        return "obs64.exe" in out.lower()
    except Exception:  # noqa: BLE001
        return False


def pick_camera(requested):
    names = list_cameras()
    if requested is not None:
        return requested, names[requested] if requested < len(names) else f"camera {requested}"
    for i, n in enumerate(names):
        if not any(v in n.lower() for v in VIRTUAL_CAM_NAMES):
            return i, n                       # first REAL webcam (never read our own virtual cam)
    return 0, names[0] if names else "camera 0"


def open_camera(args):
    """Open the physical webcam; if it's busy (e.g. OBS holds it) explain and keep retrying."""
    index, name = pick_camera(args.camera)
    warned = False
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if ok and frame is not None and frame.any():
            print(f"Using camera {index}: {name}")
            return cap, frame
        cap.release()
        if not warned:
            print(f"\n[!] Can't get video from camera {index} ({name}) - another app is using it.")
            if obs_running():
                print("    OBS Studio is running and is probably holding your webcam.")
                print("    -> CLOSE OBS completely (also from the system tray). It does NOT need to be open:")
                print("       this program feeds 'OBS Virtual Camera' by itself.")
            else:
                print("    Close apps that use the webcam (Meet/Zoom/Teams tabs set to the real camera,")
                print("    the Windows Camera app, etc.), or pick another camera with --camera N.")
            print("    Waiting for the camera... (Ctrl+C to quit)\n")
            warned = True
        time.sleep(2)


def open_virtual_cam(w, h, fps):
    try:
        import pyvirtualcam
        cam = pyvirtualcam.Camera(width=w, height=h, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR)
        print(f"Virtual camera started: {cam.device}  -> select it in Meet / Zoom / Teams")
        return cam
    except Exception as e:  # noqa: BLE001
        print("\n[!] Virtual camera not available:", e)
        if obs_running():
            print("    OBS Studio is open - if you clicked 'Start Virtual Camera' there, stop it or close OBS,")
            print("    then restart this program. OBS itself doesn't need to be open.")
        else:
            print("    Install OBS Studio (https://obsproject.com) once - it provides 'OBS Virtual Camera'.")
            print("    You don't need to keep OBS open.")
        print("    Continuing with preview only.\n")
        return None


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def draw_hand(img, lm, color):
    h, w = img.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(img, p, 3, color, -1)


def text(img, msg, y, color=(255, 255, 255), scale=0.7):
    cv2.putText(img, msg, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, msg, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main():
    args = parse_args()
    if args.list_cameras:
        for i, n in enumerate(list_cameras()):
            print(f"  {i}: {n}")
        return
    paths = ensure_models()
    book = GestureBook(threshold=args.threshold)

    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=paths["hand_landmarker.task"]),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.6, min_tracking_confidence=0.5))
    cap, frame = open_camera(args)
    H, W = frame.shape[:2]

    masker = PersonMasker(paths["selfie_segmenter.tflite"],
                          None if args.fast else paths["selfie_multiclass_256x256.tflite"],
                          W, H, threshold=args.mask_threshold, grow=args.mask_grow)
    light = LightMatcher(W, H)
    vcam = None if args.no_vcam else open_virtual_cam(W, H, args.fps)
    fx = CloudEffect(W, H)
    effect = args.effect

    background = None
    bg_start = None                 # time B was pressed (capture runs after a countdown)
    bg_sum = bg_wsum = bg_plain = None
    bg_count = 0
    invisible, level = False, 0.0
    streak, armed, release = 0, True, 0
    show_debug = False
    last_ts = 0
    rec_slot, rec_start, rec_samples = None, 0.0, []
    flash_msg, flash_until = "", 0.0

    win = "Disappear (control window)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, int(960 * H / W))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.monotonic() * 1000), last_ts + 1)
        last_ts = ts
        now = time.time()

        #person mask
        soft, guard = masker.compute(mp_img, rgb, ts)

        #background capture: after a countdown, average only pixels where nobody is seen
        capturing = bg_start is not None and now - bg_start >= BG_COUNTDOWN
        if capturing:
            if bg_sum is None:
                bg_sum = np.zeros((H, W, 3), np.float32)
                bg_wsum = np.zeros((H, W, 1), np.float32)
                bg_plain = np.zeros((H, W, 3), np.float32)
                bg_count = 0
            w = (cv2.resize(guard, (W, H), interpolation=cv2.INTER_NEAREST) == 0).astype(np.float32)[..., None]
            bg_sum += frame * w
            bg_wsum += w
            bg_plain += frame
            bg_count += 1
            if bg_count >= BG_CAPTURE_FRAMES:
                seen = bg_wsum > 0
                background = np.where(seen, bg_sum / np.maximum(bg_wsum, 1), bg_plain / bg_count).astype(np.float32)
                covered = 1 - float(seen.mean())
                if covered > 0.01:
                    flash_msg = f"Background saved, but you were in {covered:.0%} of it - press B and step fully out"
                else:
                    flash_msg = "Background captured"
                flash_until = now + 4
                bg_start = bg_sum = bg_wsum = bg_plain = None

        #hands
        hres = hands.detect_for_video(mp_img, ts)
        hand_pts = []
        for i, lm in enumerate(hres.hand_landmarks):
            side = hres.handedness[i][0].category_name if hres.handedness else None
            hand_pts.append((lm, normalize(lm, side)))

        #custom gesture recording
        if rec_slot:
            if now - rec_start >= RECORD_COUNTDOWN and hand_pts:
                rec_samples.append(hand_pts[0][1])
            if len(rec_samples) >= RECORD_SAMPLES:
                book.set(rec_slot, rec_samples[::2])
                flash_msg, flash_until = f"Saved your {rec_slot.upper()} gesture!", now + 3
                rec_slot, armed, streak = None, False, 0
            elif now - rec_start > RECORD_COUNTDOWN + 8:
                flash_msg, flash_until = "No hand seen - recording cancelled", now + 3
                rec_slot = None

        #gesture -> state (only look for the gesture that changes the current state)
        slot = "appear" if invisible else "vanish"
        matched, score, hand_hits = False, 0.0, []
        if not rec_slot:
            for lm, pts in hand_pts:
                m, s = book.matches(slot, pts)
                score = max(score, s)
                matched |= m
                hand_hits.append((lm, m))
        if matched:
            streak, release = streak + 1, 0
        else:
            streak, release = 0, release + 1
            if release >= 4:
                armed = True        # gesture must be released before it can fire again
        if background is not None and bg_start is None and armed and streak >= args.hold:
            invisible, armed, streak = not invisible, False, 0
            if effect == "cloud":
                fx.burst(soft, 90)

        #animate
        target = 1.0 if invisible else 0.0
        level += float(np.clip(target - level, -args.fade_speed, args.fade_speed))
        if effect == "cloud" and 0 < level < 1:
            fx.burst(soft, 4)

        #compose
        out = frame
        if background is not None and bg_start is None:
            # match the saved background to the camera's current exposure / white balance
            if level > 0:
                bg_now = light.correct(frame, background, soft) if not args.no_light_match \
                    else background.astype(np.uint8)
                amt = level if effect == "fade" else fx.dissolve(level)
                a = (soft * amt).astype(np.float32)
                out = cv2.blendLinear(frame, bg_now, 1.0 - a, a)
        out = fx.update_and_render(out)

        #send clean frame to meeting
        if vcam is not None:
            vcam.send(out if args.mirror_vcam else cv2.flip(out, 1))

        #control-window HUD (not sent to the meeting)
        view = out.copy()
        if show_debug:
            # top-right: what is hidden (white) | the saved background (should NOT contain you)
            tw, th = W // 5, H // 5
            m = cv2.resize(cv2.cvtColor((soft * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), (tw, th))
            view[10:10 + th, W - 20 - 2 * tw:W - 20 - tw] = m
            if background is not None:
                view[10:10 + th, W - 10 - tw:W - 10] = cv2.resize(background.astype(np.uint8), (tw, th))
        if show_debug or background is None or rec_slot:
            for lm, _ in hand_pts:
                hit = any(m for l2, m in hand_hits if l2 is lm)
                draw_hand(view, lm, (0, 255, 0) if hit else (0, 255, 255))

        if rec_slot:
            left = RECORD_COUNTDOWN - (now - rec_start)
            if left > 0:
                text(view, f"Show your {rec_slot.upper()} gesture in {left:.1f}s", 40, (0, 200, 255), 1.0)
            else:
                text(view, f"Recording {rec_slot.upper()}... hold it, move slightly "
                           f"({len(rec_samples)}/{RECORD_SAMPLES})", 40, (0, 0, 255), 0.9)
        elif bg_start is not None:
            left = BG_COUNTDOWN - (now - bg_start)
            if left > 0:
                text(view, f"Step OUT of the camera view... capturing in {left:.1f}s", 40, (0, 200, 255), 1.0)
            else:
                text(view, f"Capturing background - stay out of view ({bg_count}/{BG_CAPTURE_FRAMES})",
                     40, (0, 0, 255), 0.9)
        elif background is None:
            text(view, "Press B, then step out of view to capture the background", 40, (0, 200, 255), 0.9)
        else:
            state = "INVISIBLE" if invisible else "VISIBLE"
            text(view, f"{state}  |  show '{book.name(slot)}' to {slot}", 40,
                 (0, 255, 0) if invisible else (255, 255, 255), 0.9)
            if hand_pts:
                cv2.rectangle(view, (12, 55), (12 + int(200 * score), 63), (0, 200, 255), -1)
                cv2.rectangle(view, (12, 55), (212, 63), (255, 255, 255), 1)
            if streak:
                cv2.rectangle(view, (12, 68), (12 + int(200 * min(1, streak / args.hold)), 76), (0, 255, 0), -1)
        if now < flash_until:
            text(view, flash_msg, 110, (0, 255, 255), 0.9)
        footer = (f"B bg | 1/2 record vanish/appear | 0 reset | E effect: {effect} | M debug | Q quit"
                  f"   {'VCAM ON' if vcam else 'NO VCAM'}")
        text(view, footer, H - 15, (220, 220, 220), 0.55)

        cv2.imshow(win, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("b"):
            bg_start, bg_sum, bg_count = now, None, 0
            light.reset()
            invisible, level = False, 0.0
        elif key in (ord("1"), ord("2")):
            rec_slot = "vanish" if key == ord("1") else "appear"
            rec_start, rec_samples = now, []
        elif key == ord("0"):
            book.reset()
            flash_msg, flash_until = "Gestures reset: open palm / fist", now + 3
        elif key == ord("e"):
            effect = EFFECTS[(EFFECTS.index(effect) + 1) % len(EFFECTS)]
        elif key == ord("m"):
            show_debug = not show_debug
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    if vcam is not None:
        vcam.close()
    cv2.destroyAllWindows()
    hands.close()
    masker.close()


if __name__ == "__main__":
    main()
