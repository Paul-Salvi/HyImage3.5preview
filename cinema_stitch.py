#!/usr/bin/env python3
"""
cinema_stitch.py - Non-slideshow presentation engines for Hy-Image-3.5-preview frames.

Modes:
  1. --mode vortex : Continuous zero-stop 4-layer deep logarithmic tunnel with
                     inner cutout drop-shadows and subtle 3D camera banking.
  2. --mode bento  : Dynamic multi-panel Swiss Bento Grid where scenes split,
                     slide, and tile simultaneously on screen (zero zoom-slideshow feel).

USAGE:
  python cinema_stitch.py --frames ./frames --mode vortex --out vortex_cascade.mp4
  python cinema_stitch.py --frames ./frames --mode bento  --out bento_cascade.mp4
"""

import argparse
import glob
import math
import os
import shutil
import subprocess
import sys
import cv2
import numpy as np

from stitch import Scene, placement, Sink

YELLOW_BGR = (0, 212, 255)   # #FFD400 signal yellow
NAVY_BGR = (32, 18, 11)      # #0B1220 ink-navy
PAPER_BGR = (214, 230, 237)  # #EDE6D6 newsprint


def smoother(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)


# =============================================================================
# MODE 1: 4-LAYER DEEP NON-STOP VORTEX (Zero Holds, Constant Velocity, Shadows)
# =============================================================================
def build_deep_nested_scenes(scenes, depth=4):
    """
    Pre-composites `depth` future scenes recursively inside each scene's portal
    with a realistic inner drop-shadow so you see 4 layers deep at all times.
    """
    n = len(scenes)
    raw = [s.levels[0].copy() for s in scenes]

    # Work backwards from deepest level so nesting cascades cleanly
    composed = [img.copy() for img in raw]
    for _ in range(depth):
        next_composed = []
        for i, s in enumerate(scenes):
            base = raw[i].copy()
            child = composed[(i + 1) % n]
            px, py, pw, ph = (int(round(v)) for v in s.portal)
            bx, by, bw, bh = placement(s)

            sx = bw / child.shape[1]
            m = np.array([[sx, 0, bx - px], [0, sx, by - py]], np.float32)
            patch = cv2.warpAffine(child, m, (pw, ph), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)

            # Add an inner drop-shadow around the portal opening for physical depth
            shadow = np.ones((ph, pw, 3), np.float32)
            b_thick = max(1, int(min(pw, ph) * 0.035))
            b_thick = min(b_thick, pw // 2, ph // 2) # Prevents crash on tiny artifacts
            for d in range(b_thick):
                factor = 0.35 + 0.65 * (d / b_thick)
                shadow[d, :, :] *= factor
                shadow[ph - 1 - d, :, :] *= factor
                shadow[:, d, :] *= factor
                shadow[:, pw - 1 - d, :] *= factor
            patch = np.clip(patch.astype(np.float32) * shadow, 0, 255).astype(np.uint8)

            y0, y1 = max(0, py), min(s.h, py + ph)
            x0, x1 = max(0, px), min(s.w, px + pw)
            base[y0:y1, x0:x1] = patch[y0 - py:y1 - py, x0 - px:x1 - px]

            # Crisp dark gunmetal chamfer + 1px metallic inner highlight
            cv2.rectangle(base, (x0, y0), (x1 - 1, y1 - 1), (28, 24, 20), 4)
            cv2.rectangle(base, (x0 + 2, y0 + 2), (x1 - 3, y1 - 3), (180, 190, 200), 1)
            next_composed.append(base)
        composed = next_composed

    for i, s in enumerate(scenes):
        s.levels = [composed[i]]
        while min(s.levels[-1].shape[:2]) > 256:
            p = s.levels[-1]
            s.levels.append(cv2.resize(p, (p.shape[1] // 2, p.shape[0] // 2), interpolation=cv2.INTER_AREA))


def render_vortex(scenes, sink, ow, oh, fps, sec_per_scene=1.15, roll_deg=3.5):
    """Continuous constant-speed logarithmic camera flight with 3D Z-roll."""
    build_deep_nested_scenes(scenes, depth=4)
    n = len(scenes)
    frames_per_scene = int(round(sec_per_scene * fps))
    total_frames = n * frames_per_scene
    rng = np.random.default_rng(42)

    for g_frame in range(total_frames):
        i = g_frame // frames_per_scene
        f = g_frame % frames_per_scene
        # Linear t in logarithmic zoom space = ZERO stop-and-go slideshow feel!
        t = f / float(frames_per_scene)

        a = scenes[i]
        b = scenes[(i + 1) % n]
        px, py, pw, ph = a.portal
        bx, by, bw, bh = placement(a)

        # Overshoot canvas slightly (1.08x) so 3D camera roll never exposes black corners
        pad_w, pad_h = int(ow * 1.08), int(oh * 1.08)

        w = a.w * (bw / a.w) ** t
        alpha = t if abs(bw - a.w) < 1e-3 else (1 / w - 1 / a.w) / (1 / bw - 1 / a.w)
        cx = a.w / 2 + ((bx + bw / 2) - a.w / 2) * alpha
        cy = a.h / 2 + ((by + bh / 2) - a.h / 2) * alpha
        h = w * a.h / a.w
        vx, vy = cx - w / 2, cy - h / 2
        k = pad_w / w

        base = a.warp(k, -vx * k, -vy * k, pad_w, pad_h, cv2.BORDER_REPLICATE)
        nxt = b.warp(bw / b.w * k, (bx - vx) * k, (by - vy) * k, pad_w, pad_h, cv2.BORDER_CONSTANT)

        # Seamless portal-to-full-frame blend over the second half of the flight
        ramp = float(smoother((t - 0.35) / 0.55))
        p0 = np.array([(px - vx) * k, (py - vy) * k, (px + pw - vx) * k, (py + ph - vy) * k])
        p1 = np.array([(bx - vx) * k, (by - vy) * k, (bx + bw - vx) * k, (by + bh - vy) * k])
        x0, y0, x1, y1 = p0 * (1 - ramp) + p1 * ramp

        m = np.zeros((pad_h, pad_w), np.float32)
        xa, ya = int(max(0, round(x0))), int(max(0, round(y0)))
        xb, yb = int(min(pad_w, round(x1))), int(min(pad_h, round(y1)))
        if xb > xa and yb > ya:
            m[ya:yb, xa:xb] = 1.0
        m = cv2.GaussianBlur(m, (9, 9), 2.0)[..., None]
        frame = base.astype(np.float32) * (1 - m) + nxt.astype(np.float32) * m

        # Continuous sinusoidal 3D camera roll across the entire loop
        angle = roll_deg * math.sin(2.0 * math.pi * (g_frame / float(total_frames)))
        rot_m = cv2.getRotationMatrix2D((pad_w / 2, pad_h / 2), angle, 1.0)
        rotated = cv2.warpAffine(frame, rot_m, (pad_w, pad_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Center crop to output resolution + subtle grain
        ox, oy = (pad_w - ow) // 2, (pad_h - oh) // 2
        out = rotated[oy:oy + oh, ox:ox + ow]
        out += rng.normal(0, 2.5, (oh, ow, 1)).astype(np.float32)
        sink.write(np.clip(out, 0, 255).astype(np.uint8))


# =============================================================================
# MODE 2: KINETIC SWISS BENTO GRID (Multi-Panel Split-Screen Choreography)
# =============================================================================
def fill_portal_with_graphic(scene, next_scene):
    """Replaces the magenta box with a high-contrast dithered preview of the next scene."""
    img = scene.levels[0].copy()
    px, py, pw, ph = (int(round(v)) for v in scene.portal)
    patch = cv2.resize(next_scene.levels[0], (pw, ph), interpolation=cv2.INTER_AREA)
    y0, y1 = max(0, py), min(scene.h, py + ph)
    x0, x1 = max(0, px), min(scene.w, px + pw)
    img[y0:y1, x0:x1] = patch[y0 - py:y1 - py, x0 - px:x1 - px]
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), YELLOW_BGR, 4)
    return img


def place_in_rect(canvas, img, rect, zoom_pan=0.0):
    """Draws `img` cropped/fitted into `rect` (x, y, w, h) with subtle internal kinetic pan."""
    rx, ry, rw, rh = [int(round(v)) for v in rect]
    if rw <= 4 or rh <= 4:
        return
    ih, iw = img.shape[:2]
    scale = max(rw / iw, rh / ih) * (1.04 + 0.06 * zoom_pan)
    sw, sh = int(round(iw * scale)), int(round(ih * scale))
    resized = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    ox = max(0, (sw - rw) // 2)
    oy = max(0, (sh - rh) // 2)
    crop = resized[oy:oy + rh, ox:ox + rw]

    y0, y1 = max(0, ry), min(canvas.shape[0], ry + rh)
    x0, x1 = max(0, rx), min(canvas.shape[1], rx + rw)
    if y1 > y0 and x1 > x0:
        canvas[y0:y1, x0:x1] = crop[:y1 - y0, :x1 - x0]
        # Crisp Swiss grid border + signal-yellow corner ticks
        cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), PAPER_BGR, 2)
        tick = min(20, (x1 - x0) // 6, (y1 - y0) // 6)
        cv2.line(canvas, (x0, y0), (x0 + tick, y0), YELLOW_BGR, 4)
        cv2.line(canvas, (x0, y0), (x0, y0 + tick), YELLOW_BGR, 4)


def render_bento(scenes, sink, ow, oh, fps, sec_per_beat=1.25):
    """
    Choreographs all 8 scenes as an evolving multi-window Bloomberg/Swiss Bento wall.
    Every beat splits or slides the active grid so 2-4 scenes live on screen together.
    """
    n = len(scenes)
    clean_imgs = [fill_portal_with_graphic(scenes[i], scenes[(i + 1) % n]) for i in range(n)]
    frames_per_beat = int(round(sec_per_beat * fps))
    gap = 12

    # Define 8 evolving multi-panel Bento layouts (normalized [x, y, w, h] slots)
    # Each beat transitions smoothly from layout[b] -> layout[b+1]
    for i in range(n):
        img_a = clean_imgs[i]
        img_b = clean_imgs[(i + 1) % n]
        img_c = clean_imgs[(i + 2) % n]
        img_d = clean_imgs[(i + 3) % n]

        for f in range(frames_per_beat):
            t = f / float(frames_per_beat)
            e = float(smoother(t))
            canvas = np.full((oh, ow, 3), NAVY_BGR, dtype=np.uint8)

            if i % 3 == 0:
                # Pattern A: Full screen splits left (55%), while B and C stack on the right (45%)
                w_left = int(ow * (1.0 - 0.44 * e)) - gap
                place_in_rect(canvas, img_a, (gap, gap, w_left - gap, oh - 2 * gap), t)
                rx = w_left + gap
                rw = ow - rx - gap
                if rw > 20:
                    h_top = int((oh - 3 * gap) * 0.55)
                    place_in_rect(canvas, img_b, (rx, gap, rw, h_top), t)
                    place_in_rect(canvas, img_c, (rx, 2 * gap + h_top, rw, oh - 3 * gap - h_top), t)
            elif i % 3 == 1:
                # Pattern B: 3-panel Bento (A on left, B top-right, C bottom-right) expands B to full top, D enters bottom
                split_x = int(ow * 0.56 * (1.0 - e))
                if split_x > 20:
                    place_in_rect(canvas, img_a, (gap, gap, split_x - gap, oh - 2 * gap), t)
                rx = max(gap, split_x + gap)
                rw = ow - rx - gap
                h_top = int((oh - 3 * gap) * (0.55 + 0.10 * e))
                place_in_rect(canvas, img_b, (rx, gap, rw, h_top), t)
                w_bot_left = int(rw * (1.0 - 0.5 * e))
                place_in_rect(canvas, img_c, (rx, 2 * gap + h_top, w_bot_left, oh - 3 * gap - h_top), t)
                if rw - w_bot_left - gap > 20:
                    place_in_rect(canvas, img_d, (rx + w_bot_left + gap, 2 * gap + h_top,
                                                  rw - w_bot_left - gap, oh - 3 * gap - h_top), t)
            else:
                # Pattern C: Quad telemetry wall collapses cleanly into the next full-bleed anchor frame
                exp = e
                bx = int(gap + (ow * 0.28) * (1 - exp))
                by = int(gap + (oh * 0.22) * (1 - exp))
                bw = int(ow - 2 * bx)
                bh = int(oh - 2 * by)
                # Background flanking panels
                place_in_rect(canvas, img_a, (gap, gap, ow // 2 - gap, oh // 2 - gap), t)
                place_in_rect(canvas, img_c, (ow // 2 + gap, gap, ow // 2 - 2 * gap, oh // 2 - gap), t)
                place_in_rect(canvas, img_d, (gap, oh // 2 + gap, ow - 2 * gap, oh // 2 - 2 * gap), t)
                # Expanding hero panel in center with heavy navy drop-border
                cv2.rectangle(canvas, (bx - 8, by - 8), (bx + bw + 8, by + bh + 8), NAVY_BGR, -1)
                place_in_rect(canvas, img_b, (bx, by, bw, bh), t)

            # Live top telemetry status bar
            cv2.rectangle(canvas, (0, 0), (ow, 28), NAVY_BGR, -1)
            status = f"PACIFIC COAST ROUTE // NODE 0{i+1} -> 0{((i+1)%n)+1} // TELEMETRY ACTIVE"
            cv2.putText(canvas, status, (16, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, YELLOW_BGR, 1, cv2.LINE_AA)
            sink.write(canvas)


def main():
    ap = argparse.ArgumentParser(description="Non-slideshow cinema stitcher")
    ap.add_argument("--frames", default="./frames", help="folder with frames")
    ap.add_argument("--config", help="optional JSON config to override portal coordinates")
    ap.add_argument("--mode", choices=["vortex", "bento"], default="bento")
    ap.add_argument("--out", default="cascade_cinema.mp4")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--sec", type=float, default=1.2, help="seconds per scene beat")
    args = ap.parse_args()

    import json
    from stitch import load_scenes
    
    cfg = json.load(open(args.config, encoding="utf-8")) if args.config else {}
    scenes = load_scenes(args, cfg)

    ow = (args.width // 2) * 2
    oh = (int(round(ow * 9 / 16)) // 2) * 2

    sink = Sink(args.out, ow, oh, args.fps, crf=16)
    print(f"Rendering {len(scenes)} scenes in '{args.mode}' mode -> {args.out} ({ow}x{oh} @ {args.fps}fps)...")

    if args.mode == "vortex":
        render_vortex(scenes, sink, ow, oh, args.fps, sec_per_scene=args.sec)
    else:
        render_bento(scenes, sink, ow, oh, args.fps, sec_per_beat=args.sec)

    sink.close()
    print(f"Done! Saved {args.out}")

if __name__ == "__main__":
    main()