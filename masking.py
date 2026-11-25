import threading
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


def _segmenter(path, mode):
    return vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=path),
        running_mode=mode, output_confidence_masks=True, output_category_mask=False))


class QualityWorker(threading.Thread):
    """Runs the multiclass model (background/hair/body/face/clothes/other) on the newest frame."""

    def __init__(self, path):
        super().__init__(daemon=True)
        self.seg = _segmenter(path, vision.RunningMode.IMAGE)
        self.frame = None
        self.result, self.result_time = None, 0.0
        self.cv = threading.Condition()
        self.running = True

    def submit(self, rgb_small):
        with self.cv:
            self.frame = rgb_small
            self.cv.notify()

    def run(self):
        while self.running:
            with self.cv:
                while self.frame is None and self.running:
                    self.cv.wait(0.5)
                frame, self.frame = self.frame, None
            if frame is None:
                continue
            masks = self.seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)).confidence_masks
            hw = frame.shape[:2]
            # hair + body skin + face skin + clothes  ("others" and "background" excluded)
            person = sum(masks[i].numpy_view().reshape(hw) for i in (1, 2, 3, 4))
            person = np.clip((person - 0.3) / 0.4, 0, 1)      # suppress low-confidence noise
            self.result, self.result_time = person.astype(np.float32), time.time()

    def stop(self):
        self.running = False
        with self.cv:
            self.cv.notify()
        self.seg.close()


class PersonMasker:
    def __init__(self, fast_path, quality_path, w, h, threshold=0.2, grow=18):
        self.w, self.h = w, h
        self.pw, self.ph = w // 2, h // 2                    # morphology done at half res
        self.fast = _segmenter(fast_path, vision.RunningMode.VIDEO)
        self.worker = None
        if quality_path:
            self.worker = QualityWorker(quality_path)
            self.worker.start()
        self.threshold = threshold
        self.prev = None
        k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        self.k_close, self.k_grow, self.k_guard = k(8), k(grow), k(grow * 3)
        self.min_blob = 0.002 * self.pw * self.ph

    def compute(self, mp_img, rgb, ts):
        """Returns (soft, guard): soft = 0..1 full-res cover mask; guard = uint8 half-res area
        that must NOT be used to update the background (person + wide safety margin)."""
        fast = self.fast.segment_for_video(mp_img, ts).confidence_masks[0].numpy_view()
        prob = cv2.resize(fast.reshape(self.h, self.w), (self.pw, self.ph), interpolation=cv2.INTER_AREA)

        if self.worker:
            self.worker.submit(cv2.resize(rgb, (self.pw, self.ph), interpolation=cv2.INTER_AREA))
            q = self.worker.result
            if q is not None and time.time() - self.worker.result_time < 0.6:
                prob = np.maximum(prob, q)

        # sticky over time: a flicker of low confidence doesn't reveal you
        if self.prev is not None:
            prob = np.maximum(prob, self.prev * 0.85)
        self.prev = prob

        hard = (prob > self.threshold).astype(np.uint8)
        hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, self.k_close)
        # fill holes inside the silhouette, drop tiny noise blobs
        contours, _ = cv2.findContours(hard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(hard)
        big = [c for c in contours if cv2.contourArea(c) > self.min_blob]
        cv2.drawContours(filled, big, -1, 1, cv2.FILLED)
        grown = cv2.dilate(filled, self.k_grow)              # extra margin for hair strands & motion
        soft = cv2.GaussianBlur(grown.astype(np.float32), (0, 0), 6)
        soft = np.clip(soft * 1.6, 0, 1)                     # keep the interior fully covered
        guard = cv2.dilate(grown, self.k_guard)
        soft = cv2.resize(soft, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return soft, guard

    def close(self):
        if self.worker:
            self.worker.stop()
        self.fast.close()


class LightMatcher:
    """Webcams auto-adjust exposure / white balance (e.g. when you walk in wearing dark clothes),
    so the saved background no longer matches the live image -> a dark/bright 'shadow' of you.
    This measures the live/background brightness ratio per colour channel in the visible area
    around you, spreads it smoothly into the hidden area, and corrects the background with it."""

    def __init__(self, w, h, grid_w=160):
        self.w, self.h = w, h
        self.gw, self.gh = grid_w, max(2, round(grid_w * h / w))
        self.ratio = None

    @staticmethod
    def _pull_push(val, wt):
        """Fill cells with weight 0 by smooth interpolation from cells with weight 1
        (pyramid 'pull-push': exact at known cells, membrane-smooth inside holes)."""
        vals, wts = [val * wt[..., None]], [wt]
        while min(wts[-1].shape) > 4:
            vals.append(cv2.pyrDown(vals[-1]))
            wts.append(cv2.pyrDown(wts[-1]))
        w = wts[-1][..., None]
        est = np.where(w > 1e-4, vals[-1] / np.maximum(w, 1e-4),
                       vals[-1].sum((0, 1)) / max(float(w.sum()), 1e-4))
        for v, wl in zip(reversed(vals[:-1]), reversed(wts[:-1])):
            up = cv2.pyrUp(est, dstsize=(v.shape[1], v.shape[0]))
            wl = wl[..., None]
            a = np.clip(wl * 2, 0, 1)                           # trust data where coverage is good
            est = a * (v / np.maximum(wl, 1e-4)) + (1 - a) * up
        return est

    def correct(self, frame, background, soft):
        """frame uint8, background float32, soft = full-res hide mask (0..1).
        Returns the lighting-matched background as uint8."""
        size = (self.gw, self.gh)
        f = cv2.resize(frame, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        b = cv2.resize(background, size, interpolation=cv2.INTER_AREA)
        hidden = cv2.resize(soft, size, interpolation=cv2.INTER_AREA)
        # measure right up to the edge of the hidden area (cells with no person pixels at all)
        known = (cv2.dilate(hidden, np.ones((3, 3), np.uint8)) < 0.01).astype(np.float32)

        if known.sum() < 20:                                   # almost no free area: can't measure
            ratio = np.ones_like(f) if self.ratio is None else self.ratio
        else:
            raw = np.clip((f + 2.0) / (b + 2.0), 0.3, 3.0)     # per-channel live/saved ratio
            # light denoise of the measurements (normalised blur over known cells only)
            k = known[..., None]
            num = cv2.GaussianBlur(raw * k, (0, 0), 1.0)
            den = cv2.GaussianBlur(known, (0, 0), 1.0)[..., None]
            raw = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 1.0)
            ratio = self._pull_push(raw.astype(np.float32), known)
            ratio = np.clip(ratio, 0.4, 2.5).astype(np.float32)
            if self.ratio is not None:
                ratio = 0.6 * self.ratio + 0.4 * ratio          # smooth over time
        self.ratio = ratio
        full = cv2.resize(ratio, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return cv2.convertScaleAbs(cv2.multiply(background, full))

    def reset(self):
        self.ratio = None
