# Disappear

Show a hand gesture and you vanish in a puff of cloud. The background stays where you were.
It works live in **Google Meet, Zoom, Microsoft Teams, Discord** and any other app that uses a webcam.

## One-time setup

1. `pip install -r requirements.txt`
2. Install **[OBS Studio](https://obsproject.com)** (free). It provides the **"OBS Virtual Camera"** device that meeting apps can select.
   You don't need to open or configure OBS after installing it.

The first run downloads two small MediaPipe models into `models/`.

## Use it in a meeting

1. Run `python disappear.py`. A control window opens and the terminal prints `Virtual camera started`.
2. In Meet, Zoom or Teams, open the video settings and select the camera **"OBS Virtual Camera"**.
3. Press **B**, then step completely out of the frame within 3 seconds. It captures the empty background.
   If you were still in view, it shows a warning and you should press B again.
4. Come back in and show your **vanish** gesture. You dissolve into a cloud.
5. Show your **appear** gesture. You come back, also in a cloud.

The meeting gets a clean image. Text, hand outlines and debug views only appear in the control window.

## Set your own gestures

The defaults are **open palm = vanish** and **fist = appear**. To use any hand pose you like (peace sign, rock sign, thumbs up and so on):

| Key | Action |
|-----|--------|
| **1** | Record your **vanish** gesture. Recording starts after a 3 s countdown. Hold the pose and move your hand a little. |
| **2** | Record your **appear** gesture. |
| **0** | Reset to the defaults (palm and fist). |

You can record the **same gesture for both**. It then works as a toggle.
Gestures are saved to `gestures.json` and work with either hand.
The yellow bar shows how closely your hand matches the gesture. The green bar fills while you hold the gesture.

## Other keys

| Key | Action |
|-----|--------|
| B | Re-capture the background |
| E | Switch effect: `cloud`, `dissolve` (patchy fade) or `fade` (smooth) |
| M | Debug view: what gets hidden (white), the saved background and the hand skeleton |
| Q / Esc | Quit |

## Command-line options

```
python disappear.py --camera 1          # use another webcam
python disappear.py --effect dissolve   # start with another effect
python disappear.py --fade-speed 0.03   # slower vanish (default 0.045)
python disappear.py --hold 15           # hold the gesture longer before it triggers (frames)
python disappear.py --threshold 0.12    # stricter custom-gesture matching (default 0.16)
python disappear.py --mask-threshold 0.1  # hide more aggressively if parts of you still show
python disappear.py --mask-grow 30      # bigger margin around you (loose hair, fast motion)
python disappear.py --fast              # skip the accurate hair/clothes model (less CPU, less accurate)
python disappear.py --no-light-match    # turn off the exposure / lighting correction
python disappear.py --no-vcam           # preview window only
python disappear.py --mirror-vcam       # send a mirrored image to the meeting
```
