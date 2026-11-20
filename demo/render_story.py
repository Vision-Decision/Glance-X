"""Render the demo story from demo/predict_story.py output: 256 images, one request, a scan picks the image, zoom
in, then new requests find their targets in it. Every box, score, ranking and timing is a real model output.

python demo/render_story.py --pred demo/story_predictions.json --image_root IMG --out assets/demo.mp4 --gif assets/demo.gif
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import (BG, BLUE, CJK_CANDIDATES, DIM, FG, FONT_CANDIDATES, FPS, H, LINE, MONO_CANDIDATES,  # noqa: E402
                    MUTED, ORANGE, PALETTE, PANEL, RED, S, W, Canvas, Fitted, Fonts, clamp01, draw_box, ease, first)

COLS, ROWS, GAP = 16, 16, 2


def smooth(t):
    t = clamp01(t)
    return t * t * (3 - 2 * t)


def cell_rect(i):
    cw, ch = (W - GAP * (COLS + 1)) / COLS, (H - GAP * (ROWS + 1)) / ROWS
    c, r = i % COLS, i // COLS
    x, y = GAP + c * (cw + GAP), GAP + r * (ch + GAP)
    return x, y, cw, ch


def thumb(path, w, h):
    img = Image.open(path).convert("RGB")
    W0, H0 = img.size
    s = max(w / W0, h / H0)
    img = img.resize((max(1, int(W0 * s)), max(1, int(H0 * s))), Image.LANCZOS)
    x, y = (img.size[0] - w) // 2, (img.size[1] - h) // 2
    return img.crop((x, y, x + w, y + h))


def dim(img, a):
    return Image.blend(img, Image.new("RGB", img.size, BG), clamp01(a)) if a > 0 else img


def search_bar(cv, text, typed_frac, alpha, status=None):
    q = text[: max(1, int(len(text) * clamp01(typed_frac)))]
    tw = max(cv.tw(text, 24, True), 360)
    x0, x1, y0, y1 = W / 2 - tw / 2 - 56, W / 2 + tw / 2 + 32, 300, 352
    a = int(235 * alpha)
    cv.rrect([x0, y0, x1, y1], 26, fill=PANEL + (a,), outline=BLUE + (int(255 * alpha),), width=2)
    cx, cy = x0 + 26, (y0 + y1) / 2
    cv.d.ellipse([(cx - 9) * S, (cy - 9) * S, (cx + 6) * S, (cy + 6) * S], outline=BLUE + (int(255 * alpha),), width=3 * S)
    cv.d.line([(cx + 5) * S, (cy + 5) * S, (cx + 12) * S, (cy + 12) * S], fill=BLUE + (int(255 * alpha),), width=3 * S)
    cv.text((x0 + 48, cy), q, 24, FG + (int(255 * alpha),), True, anchor="lm")
    if status:
        sw = cv.tw(status, 16, True)
        cv.rrect([W / 2 - sw / 2 - 14, y1 + 12, W / 2 + sw / 2 + 14, y1 + 42], 15, fill=(13, 17, 23, int(220 * alpha)))
        cv.text((W / 2, y1 + 27), status, 16, MUTED + (int(255 * alpha),), True, anchor="mm")


def render(pred, root, out_mp4, gif=None):
    fonts = Fonts(first(FONT_CANDIDATES), first(CJK_CANDIDATES), first(MONO_CANDIDATES))
    files = pred["files"]
    n = len(files)
    conf = {r["image_index"]: r["confidence"] for r in pred["ranking"]}
    pick = pred["pick"]
    _, _, cw, ch = cell_rect(0)
    thumbs = [thumb(os.path.join(root, f), int(cw * S), int(ch * S)) for f in files]
    rng = random.Random(0)
    appear = {i: rng.random() for i in range(n)}
    big = Fitted(os.path.join(root, pred["zoom_file"]), 32, 76, 780, 612)
    full = Image.open(os.path.join(root, pred["zoom_file"])).convert("RGB")
    zoom = pred["zoom"]
    T_GRID, T_BAR, T_SCAN, T_PICK, T_ZOOM, PER, HOLD = 1.2, 1.4, 2.6, 1.0, 1.4, 1.7, 3.0
    t_bar, t_scan = T_GRID, T_GRID + T_BAR
    t_pick, t_zoom = t_scan + T_SCAN, t_scan + T_SCAN + T_PICK
    t_ask = t_zoom + T_ZOOM
    total = t_ask + 0.4 + len(zoom) * PER + HOLD
    status = f"{n} cached images  ·  {pred['search_ms']:.0f} ms on one GPU"
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                          "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_mp4], stdin=subprocess.PIPE)
    nframes = int(total * FPS)
    for fi in range(nframes):
        t = fi / FPS
        cv = Canvas(fonts)
        if t < t_ask:
            zt = smooth((t - t_zoom) / T_ZOOM) if t >= t_zoom else 0.0
            scan_x = -1 if t < t_scan else (W + 80) * smooth((t - t_scan) / (T_SCAN * 0.85)) - 40
            for i in range(n):
                x, y, w, h = cell_rect(i)
                a_in = smooth((t - appear[i] * 0.8) / 0.4)
                if a_in <= 0 or (zt > 0 and i == pick):
                    continue
                d = 1 - a_in
                if t >= t_scan and x + w < scan_x:                     # evaluated: fade by confidence
                    d = max(d, 0.78 * (1 - conf[i]) ** 0.7)
                if t >= t_pick and i != pick:
                    d = max(d, 0.55 + 0.35 * smooth((t - t_pick) / 0.6))
                if zt > 0:
                    d = max(d, 0.9 + 0.1 * zt)
                cv.paste(dim(thumbs[i], d), (x, y))
                if t >= t_scan and x + w < scan_x and conf[i] > 0.5 and t < t_pick + 0.6:
                    cv.rrect([x, y, x + w, y + h], 3, outline=BLUE + (200,), width=2)
            if t_scan <= t < t_pick + 0.2 and -40 < scan_x < W + 40:      # scan line + glow trail
                for k in range(26):
                    xx = scan_x - k * 3
                    cv.d.rectangle([xx * S, 0, (xx + 3) * S, H * S], fill=BLUE + (int(90 * (1 - k / 26) ** 2),))
                cv.d.rectangle([(scan_x - 1) * S, 0, (scan_x + 2) * S, H * S], fill=(200, 225, 255, 230))
            if t >= t_pick:                                               # the pick
                x, y, w, h = cell_rect(pick)
                if zt <= 0:
                    pulse = 0.5 + 0.5 * math.sin((t - t_pick) * 10)
                    cv.paste(thumbs[pick], (x, y))
                    cv.rrect([x - 2, y - 2, x + w + 2, y + h + 2], 4, outline=ORANGE, width=3 + 2 * pulse)
                    lab = f"#1  {conf[pick]:.2f}"
                    lw = cv.tw(lab, 14, True)
                    lx = min(max(x + w / 2 - lw / 2 - 8, 4), W - lw - 20)
                    cv.rrect([lx, y - 28, lx + lw + 16, y - 6], 6, fill=ORANGE)
                    cv.text((lx + 8, y - 17), lab, 14, (13, 17, 23), True, anchor="lm")
                else:                                                     # zoom: the cell grows into the frame
                    tx, ty, tw_, th_ = big.x, big.y, big.w, big.h
                    zx, zy = x + (tx - x) * zt, y + (ty - y) * zt
                    zw, zh = w + (tw_ - w) * zt, h + (th_ - h) * zt
                    img = full.resize((max(1, int(zw * S)), max(1, int(zh * S))), Image.BILINEAR)
                    cv.paste(img, (zx, zy))
                    cv.rrect([zx, zy, zx + zw, zy + zh], 6, outline=ORANGE + (int(255 * (1 - zt)),), width=3)
            if t_bar <= t < t_zoom + 0.5:
                fade = smooth((t - t_bar) / 0.3) * (1 - smooth((t - t_zoom) / 0.5))
                st = status if t >= t_pick else None
                search_bar(cv, pred["search"], (t - t_bar) / 0.9, fade, st)
        else:                                                             # inside the picked image: new requests
            big.draw(cv)
            cv.text((32, 40), "inside the picked image: new requests, same cached image", 16, MUTED, True, anchor="lm")
            y0 = 76
            for k, r in enumerate(zoom):
                t0 = t_ask + 0.4 + k * PER
                if t < t0:
                    break
                lt = t - t0
                col = PALETTE[k % len(PALETTE)]
                active = t < t0 + PER or k == len(zoom) - 1
                lines = cv.wrap(r["request"], 17, 340)[:2]
                hgt = 26 + 23 * len(lines) + 26
                al = ease(lt / 0.25)
                cv.rrect([836, y0, 1248, y0 + hgt], 12, fill=PANEL + (int(255 * al),),
                         outline=(col if active else LINE) + (int(255 * al),), width=2 if active else 1)
                cv.rrect([836, y0, 842, y0 + hgt], 3, fill=col + (int(255 * al),))
                typed = r["request"][: max(1, int(len(r["request"]) * clamp01(lt / 0.45)))]
                cv.text((858, y0 + 18), typed if len(lines) == 1 else typed[: len(lines[0])], 17, FG)
                if len(lines) > 1 and len(typed) > len(lines[0]):
                    cv.text((858, y0 + 41), typed[len(lines[0]):].lstrip(), 17, FG)
                yb = y0 + 18 + 23 * len(lines) + 8
                present = r["present_prob"] >= 0.5
                boxes = (r["boxes"] or [r["box"]]) if present else []
                if lt > 0.5:
                    msg = (f"found {len(boxes)}  ·  P(present) {r['present_prob']:.2f}" if present
                           else f"not in the image  ·  P(present) {r['present_prob']:.2f}")
                    cv.text((858, yb), msg, 14, col if present else RED, True)
                if lt > 0.55 and present:
                    for b in boxes:
                        draw_box(cv, big.map(b), col, clamp01((lt - 0.55) / 0.45), label=str(k + 1),
                                 thick=4 if active else 3, fill_a=0.14 if active else 0.0)
                y0 += hgt + 10
            cv.text((1248, 676), f"{len(zoom)} requests  ·  {pred['zoom_ms']:.0f} ms", 14, DIM, True, anchor="rm")
        cv.text((W - 14, H - 12), "Glance-X-Lite · real outputs", 12, (120, 128, 145), anchor="rs")
        p.stdin.write(cv.frame().tobytes())
    p.stdin.close()
    p.wait()
    print(f"{nframes} frames ({nframes / FPS:.1f}s) -> {out_mp4}")
    if gif:
        vf = ("fps=10,scale=880:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=192:stats_mode=diff[p];"
              "[s1][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out_mp4, "-vf", vf, gif], check=True)
        print(f"-> {gif} ({os.path.getsize(gif) / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="demo/story_predictions.json")
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--out", default="assets/demo.mp4")
    ap.add_argument("--gif", default="")
    a = ap.parse_args()
    render(json.load(open(a.pred)), a.image_root, a.out, a.gif or None)
