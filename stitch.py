#!/usr/bin/env python3
"""
stitch.py v2 - ant-scale "run through the motherboard" zoom-through video.

* Each plate has a flat magenta gateway. The camera dollies into the gateway of plate i while
  plate i+1 is composited into it (cover-fit 16:9 window that "opens" as the camera arrives).
  The last plate feeds plate 1, so the video loops.
* A 10-frame run cycle (keyed out of a green-screen sprite sheet) runs along the bottom of
  the frame the whole time, with a contact shadow and a small camera shake.
* Grain + haze ramp DOWN across the run, so the world visibly "denoises" as he runs.

REQUIREMENTS   pip install opencv-python numpy      (+ ffmpeg on PATH recommended)

USAGE
    python stitch.py --frames ./plates --runner runner_sheet.png --detect-only
    python stitch.py --frames ./plates --runner runner_sheet.png --preview
    python stitch.py --frames ./plates --runner runner_sheet.png --out run.mp4

    Plates are read in filename order (s01_*.png, s02_*.png, ...).
    --runner-debug writes the keyed/aligned sprites to ./debug/runner_sheet.png
"""
import argparse, glob, json, os, shutil, subprocess, sys, time
import cv2
import numpy as np

LIGHT = np.array([214, 230, 237], np.float32)
DARK = np.array([32, 18, 11], np.float32)


def smoother(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)


def bayer_matrix(n=8):
    m = np.array([[0, 2], [3, 1]], np.float32)
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / (n * n)


# ----------------------------------------------------------------------------- portal detection
def _longest_run(flags):
    best = (0, 0, 0)
    start = None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        if not f and start is not None:
            if i - start > best[0]:
                best = (i - start, start, i)
            start = None
    return best[1], best[2]


def detect_portal(img, expand=1):
    """Flat magenta/pink/purple gateway -> ((x,y,w,h), fill). Trims thin glow/reflection tails."""
    b, g, r = (img[..., i].astype(np.int16) for i in range(3))
    mask = ((r - g) > 80) & ((b - g) > 40) & (r > 100) & (b > 60)
    mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        raise RuntimeError("no magenta gateway found")
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (labels == k)
    cols, rows = comp.sum(0), comp.sum(1)
    x0, x1 = _longest_run(cols >= 0.5 * cols.max())
    y0, y1 = _longest_run(rows >= 0.5 * rows.max())
    core = comp[y0:y1, x0:x1]
    fill = float(core.mean()) if core.size else 0.0
    return (x0 - expand, y0 - expand, (x1 - x0) + 2 * expand, (y1 - y0) + 2 * expand), fill


def build_pyramid(img):
    lv = [img]
    while min(lv[-1].shape[:2]) > 256:
        p = lv[-1]
        lv.append(cv2.resize(p, (p.shape[1] // 2, p.shape[0] // 2), interpolation=cv2.INTER_AREA))
    return lv


def warp_levels(levels, w, sx, tx, ty, ow, oh, border):
    """Render with screen = source_px * sx + (tx, ty), picking a pyramid level to avoid aliasing."""
    if abs(tx) < 0.5 and abs(ty) < 0.5 and abs(sx * w - ow) < 0.5:
        return cv2.resize(levels[0], (ow, oh), interpolation=cv2.INTER_AREA)
    lv = 0
    while lv + 1 < len(levels) and sx * 2 ** (lv + 1) <= 1.0:
        lv += 1
    m = np.array([[sx * 2 ** lv, 0, tx], [0, sx * 2 ** lv, ty]], np.float32)
    return cv2.warpAffine(levels[lv], m, (ow, oh), flags=cv2.INTER_CUBIC, borderMode=border)


def despill(img, portal):
    """Neutralize magenta glow / reflections around the gateway (outside the gateway itself)."""
    x, y, w, h = (int(round(v)) for v in portal)
    pad = int(0.9 * max(w, h))
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    roi = img[y0:y1, x0:x1].astype(np.float32)
    b, g, r = roi[..., 0], roi[..., 1], roi[..., 2]
    spill = np.clip(np.minimum(r - g, b - g) / 55.0, 0, 1)
    spill[max(0, y - y0):max(0, y + h - y0), max(0, x - x0):max(0, x + w - x0)] = 0
    avg = (r + b) / 2
    g2 = g + spill * np.maximum(avg * 0.9 - g, 0)
    r2 = r - spill * 0.25 * np.maximum(r - g2, 0)
    b2 = b - spill * 0.25 * np.maximum(b - g2, 0)
    out = img.copy()
    out[y0:y1, x0:x1] = np.clip(np.dstack([b2, g2, r2]), 0, 255).astype(np.uint8)
    return out


class Scene:
    def __init__(self, path, portal=None, zoom=None, hold=None):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cannot read {path}")
        self.path, self.name = path, os.path.basename(path)
        self.h, self.w = img.shape[:2]
        self.fill = 1.0
        if portal is None:
            portal, self.fill = detect_portal(img)
        self.portal = tuple(float(v) for v in portal)
        img = despill(img, self.portal)
        self.levels = build_pyramid(img)          # raw plate (magenta gateway visible)
        self.layer_levels = self.levels           # plate with next plate nested in its gateway (depth 2)
        self.hold_levels = self.levels            # same, one level deeper (used for held frames)
        self.zoom, self.hold = zoom, hold

    def warp(self, sx, tx, ty, ow, oh, border, which="raw"):
        lv = {"raw": self.levels, "layer": self.layer_levels, "hold": self.hold_levels}[which]
        return warp_levels(lv, self.w, sx, tx, ty, ow, oh, border)


def placement(scene):
    px, py, pw, ph = scene.portal
    bw = max(pw, ph * scene.w / scene.h)
    bh = bw * scene.h / scene.w
    cx, cy = px + pw / 2, py + ph / 2
    return cx - bw / 2, cy - bh / 2, bw, bh


def fill_gateway(a, nxt, nxt_levels):
    """Full-res copy of plate `a` whose magenta gateway shows plate `nxt` (cover-fit, same geometry as the zoom)."""
    bx, by, bw, bh = placement(a)
    px, py, pw, ph = a.portal
    layer = warp_levels(nxt_levels, nxt.w, bw / nxt.w, bx, by, a.w, a.h, cv2.BORDER_CONSTANT)
    m = np.zeros((a.h, a.w), np.float32)
    x0, y0, x1, y1 = int(round(px)), int(round(py)), int(round(px + pw)), int(round(py + ph))
    m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 1.0
    m = cv2.GaussianBlur(m, (0, 0), 0.8)[..., None]
    out = a.levels[0].astype(np.float32) * (1 - m) + layer.astype(np.float32) * m
    return np.clip(out, 0, 255).astype(np.uint8)


def build_layers(scenes, loop=True):
    """Nest each plate inside the previous plate's gateway (3 levels deep) so no flat magenta is ever visible."""
    n = len(scenes)
    prev = [s.levels for s in scenes]
    for depth in range(3):
        cur = []
        for i, s in enumerate(scenes):
            j = i + 1
            if j >= n and not loop:
                cur.append(s.levels)
                continue
            nxt = scenes[j % n]
            cur.append(build_pyramid(fill_gateway(s, nxt, prev[j % n])))
        if depth == 1:
            for s, lv in zip(scenes, cur):
                s.layer_levels = lv
        if depth == 2:
            for s, lv in zip(scenes, cur):
                s.hold_levels = lv
        prev = cur


# ----------------------------------------------------------------------------- runner sprites
def load_runner(sheet_path, n=10, debug_dir=None):
    """Key the green-screen sprite sheet -> list of aligned RGBA uint8 frames (same canvas)."""
    img = cv2.imread(sheet_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cannot read runner sheet {sheet_path}")
    f = img.astype(np.float32)
    H, W = f.shape[:2]
    border = np.concatenate([f[:24].reshape(-1, 3), f[-24:].reshape(-1, 3),
                             f[:, :24].reshape(-1, 3), f[:, -24:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    d = np.linalg.norm(f - bg, axis=2)

    fg = (d > 40).astype(np.uint8)
    merged = cv2.dilate(cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)), np.ones((31, 31), np.uint8))
    num, _, stats, cent = cv2.connectedComponentsWithStats(merged, 8)
    comps = [i for i in range(1, num) if stats[i, 4] > 6000]
    boxes = []
    if len(comps) == n:
        for i in comps:
            x, y, w, h = stats[i, :4]
            boxes.append((int(x), int(y), int(w), int(h), cent[i][1]))
        boxes.sort(key=lambda b: b[4])
        half = n // 2
        boxes = sorted(boxes[:half], key=lambda b: b[0]) + sorted(boxes[half:], key=lambda b: b[0])
    else:  # fallback: equal 2x5 grid
        print(f"  runner: found {len(comps)} blobs (expected {n}); using an equal 2x{n // 2} grid")
        cw, ch = W // (n // 2), H // 2
        for r in range(2):
            for c in range(n // 2):
                boxes.append((c * cw, r * ch, cw, ch, 0))

    sprites = []
    for (x, y, w, h, _) in boxes:
        m = 14
        x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m)
        crop = f[y0:y1, x0:x1]
        dc = np.linalg.norm(crop - bg, axis=2)
        a = np.clip((dc - 18.0) / (55.0 - 18.0), 0, 1)
        a[:, :1] = a[:, -1:] = 0
        a[:1] = a[-1:] = 0
        a = cv2.GaussianBlur(a, (3, 3), 0.6)
        fgc = np.clip((crop - (1 - a[..., None]) * bg) / np.maximum(a[..., None], 1e-3), 0, 255)
        edge = a < 0.98                                    # despill only on soft edges
        bch, gch, rch = fgc[..., 0], fgc[..., 1], fgc[..., 2]
        cap = np.maximum(rch, bch)
        fgc[..., 1] = np.where(edge & (gch > cap), cap, gch)
        rgba = np.dstack([fgc, a * 255]).astype(np.uint8)
        sprites.append(rgba)

    # anchor on the orange jacket (stable across poses)
    anchors = []
    for s in sprites:
        r_, g_, b_, a_ = s[..., 2].astype(int), s[..., 1].astype(int), s[..., 0].astype(int), s[..., 3]
        om = (r_ > 200) & (g_ > 90) & (g_ < 175) & (b_ < 150) & (a_ > 200)
        ys, xs = np.nonzero(om)
        if len(xs) < 50:
            ys, xs = np.nonzero(a_ > 200)
        anchors.append((float(np.median(xs)), float(ys.min())))
    left = max(ax for ax, _ in anchors)
    top = max(ay for _, ay in anchors)
    right = max(s.shape[1] - ax for s, (ax, _) in zip(sprites, anchors))
    bottom = max(s.shape[0] - ay for s, (_, ay) in zip(sprites, anchors))
    cw, chh = int(np.ceil(left + right)), int(np.ceil(top + bottom))
    out = []
    for s, (ax, ay) in zip(sprites, anchors):
        canvas = np.zeros((chh, cw, 4), np.uint8)
        ox, oy = int(round(left - ax)), int(round(top - ay))
        canvas[oy:oy + s.shape[0], ox:ox + s.shape[1]] = s
        out.append(canvas)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        tile = []
        for s in out:
            bgc = np.full((s.shape[0], s.shape[1], 3), 60, np.uint8)
            al = s[..., 3:4].astype(np.float32) / 255
            tile.append((s[..., :3] * al + bgc * (1 - al)).astype(np.uint8))
        rows = [np.hstack(tile[:n // 2]), np.hstack(tile[n // 2:])]
        sheet = np.vstack(rows)
        sc = 1600 / sheet.shape[1]
        cv2.imwrite(os.path.join(debug_dir, "runner_sheet.png"), cv2.resize(sheet, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA))
    print(f"  runner: {len(out)} frames, canvas {cw}x{chh}px")
    return out


# ----------------------------------------------------------------------------- renderer
class Renderer:
    def __init__(self, ow, oh, runner=None, runner_h=0.34, feet_y=0.965, grade=True, dither=True,
                 grain_hi=10.0, grain_lo=1.5, seed=7):
        self.ow, self.oh, self.grade_on, self.dither_on = ow, oh, grade, dither
        self.grain_hi, self.grain_lo = grain_hi, grain_lo
        self.rng = np.random.default_rng(seed)
        self.cell = 2 if ow >= 1400 else 1
        sw, sh = ow // self.cell, oh // self.cell
        self.thr = np.tile(bayer_matrix(8), (sh // 8 + 1, sw // 8 + 1))[:sh, :sw]
        yy, xx = np.mgrid[0:oh, 0:ow].astype(np.float32)
        rr = ((xx / ow - 0.5) ** 2 + (yy / oh - 0.5) ** 2) / 0.5
        self.vig = (1.0 - 0.16 * rr).astype(np.float32)[..., None]
        self.sprites, self.feet_y = None, feet_y
        if runner:
            ch0 = runner[0].shape[0]
            sc = runner_h * oh / ch0
            self.sprites = []
            for s in runner:
                rs = cv2.resize(s, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA).astype(np.float32)
                a = rs[..., 3:4] / 255.0
                rs = np.dstack([rs[..., :3] * a, a])          # premultiplied
                rs[..., :3] = cv2.GaussianBlur(rs[..., :3], (3, 3), 0.7)
                self.sprites.append(rs)
            sh_, sw_ = self.sprites[0].shape[:2]
            pw, ph = int(sw_ * 0.9), max(6, int(oh * 0.03))
            yy2, xx2 = np.mgrid[0:ph, 0:pw].astype(np.float32)
            e = ((xx2 - pw / 2) / (pw / 2)) ** 2 + ((yy2 - ph / 2) / (ph / 2)) ** 2
            self.shadow = np.clip(1 - e, 0, 1) ** 0.7
            self.shadow = cv2.GaussianBlur(self.shadow, (0, 0), max(1.0, oh * 0.004))

    # --- world rendering
    def static(self, scene):
        return scene.warp(self.ow / scene.w, 0, 0, self.ow, self.oh, cv2.BORDER_REPLICATE, 'hold')

    def transition(self, a, b, e):
        ow, oh = self.ow, self.oh
        px, py, pw, ph = a.portal
        bx, by, bw, bh = placement(a)
        w = a.w * (bw / a.w) ** e
        alpha = e if abs(bw - a.w) < 1e-3 else (1 / w - 1 / a.w) / (1 / bw - 1 / a.w)
        cx = a.w / 2 + ((bx + bw / 2) - a.w / 2) * alpha
        cy = a.h / 2 + ((by + bh / 2) - a.h / 2) * alpha
        h = w * a.h / a.w
        vx, vy = cx - w / 2, cy - h / 2
        k = ow / w
        base = a.warp(k, -vx * k, -vy * k, ow, oh, cv2.BORDER_REPLICATE)
        nxt = b.warp(bw / b.w * k, (bx - vx) * k, (by - vy) * k, ow, oh, cv2.BORDER_CONSTANT, 'layer')
        ramp = float(smoother((e - 0.40) / 0.50))
        p0 = np.array([(px - vx) * k, (py - vy) * k, (px + pw - vx) * k, (py + ph - vy) * k])
        p1 = np.array([(bx - vx) * k, (by - vy) * k, (bx + bw - vx) * k, (by + bh - vy) * k])
        x0, y0, x1, y1 = p0 * (1 - ramp) + p1 * ramp
        m = np.zeros((oh, ow), np.float32)
        xa, ya = int(max(0, round(x0))), int(max(0, round(y0)))
        xb, yb = int(min(ow, round(x1))), int(min(oh, round(y1)))
        if xb > xa and yb > ya:
            m[ya:yb, xa:xb] = 1.0
        m = cv2.GaussianBlur(m, (5, 5), 1.2)[..., None]
        out = base.astype(np.float32) * (1 - m) + nxt.astype(np.float32) * m
        if self.dither_on:
            p = 0.9 * float(np.sin(np.pi * np.clip((e - 0.10) / 0.85, 0, 1)) ** 2)
            out = self.dither(out, p)
        return out

    def dither(self, img, p):
        if p < 0.02:
            return img
        gray = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        sw, sh = self.ow // self.cell, self.oh // self.cell
        small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        tone = np.where((small > self.thr)[..., None], LIGHT, DARK).astype(np.float32)
        if self.cell > 1:
            tone = cv2.resize(tone, (self.ow, self.oh), interpolation=cv2.INTER_NEAREST)
        return img * (1 - p) + tone * p

    # --- per-output-frame passes
    def shake(self, img, t):
        amp = 0.0035 * self.oh
        dy = amp * np.sin(2 * np.pi * 2.0 * t)
        dx = 0.4 * amp * np.sin(2 * np.pi * 1.0 * t + 1.1)
        m = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
        return cv2.warpAffine(img, m, (self.ow, self.oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def add_runner(self, img, idx, fps, runner_fps):
        if not self.sprites:
            return img
        k = int(idx / fps * runner_fps) % len(self.sprites)
        s = self.sprites[k]
        sh_, sw_ = s.shape[:2]
        phase = k / len(self.sprites)
        bob = 0.004 * self.oh * np.cos(2 * np.pi * 2 * phase)
        x = int(round(self.ow / 2 - sw_ / 2))
        y = int(round(self.feet_y * self.oh - sh_ + bob))
        # contact shadow
        shp, swp = self.shadow.shape
        sx0, sy0 = int(round(self.ow / 2 - swp / 2)), int(round(self.feet_y * self.oh - shp / 2 - 0.004 * self.oh))
        x0, y0, x1, y1 = max(0, sx0), max(0, sy0), min(self.ow, sx0 + swp), min(self.oh, sy0 + shp)
        if x1 > x0 and y1 > y0:
            sub = self.shadow[y0 - sy0:y1 - sy0, x0 - sx0:x1 - sx0]
            img[y0:y1, x0:x1] *= (1 - 0.55 * sub)[..., None]
        # sprite (premultiplied)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(self.ow, x + sw_), min(self.oh, y + sh_)
        if x1 > x0 and y1 > y0:
            sp = s[y0 - y:y1 - y, x0 - x:x1 - x]
            img[y0:y1, x0:x1] = sp[..., :3] + img[y0:y1, x0:x1] * (1 - sp[..., 3:4])
        return img

    def finish(self, img, level):
        """level: 0 = noisiest (start of run) ... 1 = clean (end of run)."""
        f = img.astype(np.float32)
        if self.grade_on:
            nz = (1 - level)
            sigma = self.grain_lo + (self.grain_hi - self.grain_lo) * nz ** 1.3
            haze = 0.10 * nz ** 1.5                               # lifted blacks while "noisy"
            f = f * (1 - haze) + haze * 70.0
            f = f * self.vig
            noise = self.rng.standard_normal((self.oh, self.ow, 1), dtype=np.float32) * sigma
            chroma = self.rng.standard_normal((self.oh, self.ow, 3), dtype=np.float32) * (0.35 * sigma)
            f = f + noise + chroma
        return np.clip(f, 0, 255).astype(np.uint8)


class Sink:
    def __init__(self, path, ow, oh, fps, crf):
        self.proc = self.vw = None
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{ow}x{oh}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                   "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            print("ffmpeg not found - using OpenCV mp4v fallback (lower quality)")
            self.vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))

    def write(self, frame):
        (self.proc.stdin.write(frame.tobytes()) if self.proc else self.vw.write(frame))

    def close(self):
        if self.proc:
            self.proc.stdin.close(); self.proc.wait()
        else:
            self.vw.release()


# ----------------------------------------------------------------------------- main
def load_scenes(args, cfg):
    entries = cfg.get("frames")
    if entries:
        base = args.frames or "."
        entries = [dict(e) for e in entries]
        for e in entries:
            e["file"] = e["file"] if os.path.isabs(e["file"]) else os.path.join(base, e["file"])
    else:
        files = sorted(glob.glob(os.path.join(args.frames, "s[0-9][0-9]_*.png")))
        entries = [{"file": f} for f in files]
    if len(entries) < 2:
        sys.exit("need at least 2 plates named s01_*.png, s02_*.png ...")
    return [Scene(e["file"], e.get("portal"), e.get("zoom"), e.get("hold")) for e in entries]


def report(scenes, debug_dir):
    print(f"\n{'plate':40s} {'gateway x,y,w,h':20s} {'w%':>5s} {'aspect':>6s} {'fill':>5s}  notes")
    for i, s in enumerate(scenes, 1):
        x, y, w, h = s.portal
        asp, wp = w / h, 100 * w / s.w
        notes = []
        if abs(asp / (16 / 9) - 1) > 0.35:
            notes.append("aspect off 16:9 (next plate is cover-cropped while it opens)")
        if wp < 6:
            notes.append(f"tiny gateway (~{s.w / max(w, h * 16 / 9):.0f}x zoom)")
        if s.fill < 0.9:
            notes.append("gateway not solid")
        print(f"{s.name[:40]:40s} {int(x)},{int(y)},{int(w)},{int(h)}".ljust(61) +
              f" {wp:5.1f} {asp:6.2f} {s.fill:5.2f}  {'; '.join(notes) or 'ok'}")
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            img = s.levels[0].copy()
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 5)
            bx, by, bw, bh = placement(s)
            cv2.rectangle(img, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (255, 160, 0), 4)
            cv2.putText(img, str(i), (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 8)
            cv2.imwrite(os.path.join(debug_dir, f"detect_{i:02d}.jpg"), cv2.resize(img, (960, 540)))
    print()


def main():
    ap = argparse.ArgumentParser(description="Motherboard-run zoom-through stitcher v2")
    ap.add_argument("--frames", default=".")
    ap.add_argument("--runner", help="green-screen sprite sheet (2x5) of the run cycle")
    ap.add_argument("--config")
    ap.add_argument("--out", default="run.mp4")
    ap.add_argument("--width", type=int)
    ap.add_argument("--fps", type=int)
    ap.add_argument("--hold", type=float, help="seconds each plate is held (default 0.35)")
    ap.add_argument("--zoom", type=float, help="seconds per run-through (default 1.25)")
    ap.add_argument("--start-hold", type=float)
    ap.add_argument("--end-hold", type=float)
    ap.add_argument("--runner-height", type=float, default=0.34, help="runner height as fraction of frame height")
    ap.add_argument("--runner-fps", type=float, default=15.0, help="run-cycle frames per second")
    ap.add_argument("--feet-y", type=float, default=0.965, help="runner foot line as fraction of frame height")
    ap.add_argument("--grain-hi", type=float, default=10.0, help="grain sigma at the start of the run")
    ap.add_argument("--grain-lo", type=float, default=1.5, help="grain sigma at the end of the run")
    ap.add_argument("--no-loop", action="store_true")
    ap.add_argument("--no-grade", action="store_true")
    ap.add_argument("--no-dither", action="store_true")
    ap.add_argument("--no-runner", action="store_true")
    ap.add_argument("--no-shake", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--detect-only", action="store_true")
    ap.add_argument("--runner-debug", action="store_true")
    ap.add_argument("--debug-dir", default="debug")
    ap.add_argument("--crf", type=int)
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8")) if args.config else {}
    pick = lambda cli, key, d: cli if cli is not None else cfg.get(key, d)
    width = pick(args.width, "width", 960 if args.preview else 1920)
    fps = pick(args.fps, "fps", 30)
    hold = pick(args.hold, "hold", 0.35)
    zoom = pick(args.zoom, "zoom", 1.25)
    start_hold = pick(args.start_hold, "start_hold", 0.8)
    end_hold = pick(args.end_hold, "end_hold", 0.4)
    crf = pick(args.crf, "crf", 24 if args.preview else 16)
    ow = int(width) // 2 * 2
    oh = int(round(ow * 9 / 16)) // 2 * 2

    scenes = load_scenes(args, cfg)
    report(scenes, args.debug_dir)
    runner = None
    if args.runner and not args.no_runner:
        runner = load_runner(args.runner, debug_dir=args.debug_dir if (args.runner_debug or args.detect_only) else None)
    if args.detect_only:
        print(f"overlays in ./{args.debug_dir}/  (green = detected gateway, orange = 16:9 window the next plate fills)")
        return

    build_layers(scenes, loop=not args.no_loop)
    rend = Renderer(ow, oh, runner, args.runner_height, args.feet_y, not args.no_grade, not args.no_dither,
                    args.grain_hi, args.grain_lo)
    sink = Sink(args.out, ow, oh, fps, crf)
    n = len(scenes)
    order = list(range(n)) + ([] if args.no_loop else [0])
    state = {"idx": 0}
    t0 = time.time()

    def emit(world, level, count=1):
        for _ in range(count):
            i = state["idx"]
            f = world.astype(np.float32) if world.dtype != np.float32 else world.copy()
            if not args.no_shake:
                f = rend.shake(f, i / fps)
            if runner:
                f = rend.add_runner(f, i, fps, args.runner_fps)
            sink.write(rend.finish(f, level))
            state["idx"] += 1

    lv = lambda i: i / max(1, n - 1)
    cur = rend.static(scenes[0])
    emit(cur, lv(0), int(round(start_hold * fps)))
    for a_idx, b_idx in zip(order[:-1], order[1:]):
        a, b = scenes[a_idx], scenes[b_idx]
        nz = max(2, int(round((a.zoom or zoom) * fps)))
        for k in range(1, nz):
            u = k / nz
            e = float(0.5 * u + 0.5 * smoother(u))              # mostly-constant speed: he never stops
            if b_idx == 0 and a_idx == n - 1:
                level = 1.0 - float(smoother((u - 0.25) / 0.6))  # noise resets across the loop seam
            else:
                level = lv(a_idx) + (lv(b_idx) - lv(a_idx)) * u
            emit(rend.transition(a, b, e), level)
        wrap = (b_idx == 0)
        cur = rend.static(b)
        last = b_idx == order[-1] and not args.no_loop
        emit(cur, lv(b_idx), int(round((end_hold if last else (b.hold or hold)) * fps)))
        print(f"  plate {a_idx + 1} -> {b_idx + 1} done ({state['idx']} frames, {time.time() - t0:.0f}s)", flush=True)
    sink.close()
    print(f"\nwrote {args.out}: {ow}x{oh} @ {fps}fps, {state['idx']} frames, {state['idx'] / fps:.1f}s")


if __name__ == "__main__":
    main()
