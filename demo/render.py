"""Render demo/predictions.json (from demo/predict.py) into the demo video: mp4 (1280x720) + a smaller gif.

python demo/render.py --pred demo/predictions.json --image_root data/images --logo assets/logo-dark.svg \
    --out assets/demo.mp4 --gif assets/demo.gif

Every box and number in the video is a real model output; nothing is edited by hand.
Needs ffmpeg, and rsvg-convert for the logo. Fonts: pass --font / --font_cjk, or it tries common macOS / Linux paths.
"""
import argparse
import json
import math
import os
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

W, H, FPS, S = 1280, 720, 30, 2          # S = supersampling factor
BG, PANEL, LINE = (13, 17, 23), (22, 27, 35), (42, 50, 66)
FG, MUTED, DIM = (244, 246, 250), (154, 163, 181), (96, 104, 120)
ORANGE, BLUE, RED = (240, 138, 75), (91, 155, 255), (239, 98, 98)
PALETTE = [(240, 138, 75), (91, 155, 255), (88, 204, 160), (236, 112, 190), (250, 204, 82), (160, 130, 255),
           (90, 210, 230)]
MODE_TEXT = {"image_first": ("image_first", "one image · many requests"),
             "text_first": ("text_first", "one request · many images"),
             "joint": ("joint", "image and request read each other")}

FONT_CANDIDATES = ["/System/Library/Fonts/SFNS.ttf", "/usr/share/fonts/truetype/inter/Inter-Regular.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
CJK_CANDIDATES = ["/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/PingFang.ttc",
                  "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                  os.path.expanduser("~/.local/share/fonts/noto-cjk/NotoSansSC-Regular.otf")]
MONO_CANDIDATES = ["/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]


def first(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(f"no font found among {paths}")


class Fonts:
    def __init__(self, latin, cjk, mono):
        self.latin, self.cjk, self.mono, self.cache = latin, cjk, mono, {}

    def get(self, text, size, bold=False, mono=False):
        cjk = any("　" <= ch <= "鿿" or "＀" <= ch <= "￯" for ch in text)
        path = self.mono if mono else (self.cjk if cjk else self.latin)
        key = (path, size, bold)
        if key not in self.cache:
            idx = 1 if (bold and path.endswith(".ttc") and cjk) else 0
            f = ImageFont.truetype(path, size * S, index=idx)
            if not cjk and not mono:
                try:
                    f.set_variation_by_name("Bold" if bold else "Regular")
                except Exception:
                    pass
            self.cache[key] = f
        return self.cache[key]


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def clamp01(t):
    return min(max(t, 0.0), 1.0)


def mix(c, a, bg=BG):
    return tuple(int(bg[i] + (c[i] - bg[i]) * a) for i in range(3))


class Canvas:
    def __init__(self, fonts):
        self.im = Image.new("RGB", (W * S, H * S), BG)
        self.d = ImageDraw.Draw(self.im, "RGBA")
        self.f = fonts

    def rrect(self, box, r, fill=None, outline=None, width=1):
        self.d.rounded_rectangle([v * S for v in box], radius=r * S, fill=fill, outline=outline, width=int(width * S))

    def text(self, xy, s, size, color=FG, bold=False, mono=False, anchor="la"):
        self.d.text((xy[0] * S, xy[1] * S), s, font=self.f.get(s, size, bold, mono), fill=color, anchor=anchor)

    def tw(self, s, size, bold=False, mono=False):
        return self.d.textlength(s, font=self.f.get(s, size, bold, mono)) / S

    def wrap(self, s, size, width, bold=False):
        """Greedy wrap; splits on spaces, or per character for CJK."""
        units = list(s) if " " not in s else s.split(" ")
        sep = "" if " " not in s else " "
        lines, cur = [], ""
        for u in units:
            nxt = (cur + sep + u) if cur else u
            if self.tw(nxt, size, bold) <= width:
                cur = nxt
            else:
                lines.append(cur)
                cur = u
        if cur:
            lines.append(cur)
        return lines

    def paste(self, img, xy):
        self.im.paste(img, (int(xy[0] * S), int(xy[1] * S)))

    def frame(self):
        return self.im.resize((W, H), Image.LANCZOS)


def rounded(img, r):
    m = Image.new("L", img.size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, img.size[0] - 1, img.size[1] - 1], radius=r, fill=255)
    out = Image.new("RGB", img.size, BG)
    out.paste(img, (0, 0), m)
    return out


class Fitted:
    """An image scaled into a panel; maps original-pixel boxes to canvas coordinates."""

    def __init__(self, path, x, y, w, h, radius=14):
        img = Image.open(path).convert("RGB")
        self.W0, self.H0 = img.size
        s = min(w / self.W0, h / self.H0)
        self.s = s
        self.w, self.h = int(self.W0 * s), int(self.H0 * s)
        self.x, self.y = x + (w - self.w) / 2, y + (h - self.h) / 2
        self.img = rounded(img.resize((self.w * S, self.h * S), Image.LANCZOS), radius * S)

    def map(self, b):
        return [self.x + b[0] * self.s, self.y + b[1] * self.s, self.x + b[2] * self.s, self.y + b[3] * self.s]

    def draw(self, cv, dim=0.0):
        cv.paste(self.img, (self.x, self.y))
        if dim:
            cv.d.rectangle([self.x * S, self.y * S, (self.x + self.w) * S, (self.y + self.h) * S],
                           fill=BG + (int(255 * dim),))


def draw_box(cv, b, color, t, label=None, thick=4, fill_a=0.12):
    """Box grows from its centre (t: 0..1)."""
    if t <= 0:
        return
    e = ease(t)
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    bx = [cx + (b[0] - cx) * e, cy + (b[1] - cy) * e, cx + (b[2] - cx) * e, cy + (b[3] - cy) * e]
    cv.d.rounded_rectangle([v * S for v in bx], radius=6 * S, fill=color + (int(255 * fill_a * e),),
                           outline=color + (int(255 * min(1, 2 * t)),), width=int(thick * S))
    if label and t > 0.6:
        a = clamp01((t - 0.6) / 0.4)
        sz = 15
        tw = cv.tw(label, sz, True)
        lx, ly = bx[0], bx[1] - 26 if bx[1] - 26 > 60 else bx[1] + 4
        cv.rrect([lx, ly, lx + tw + 16, ly + 24], 6, fill=color + (int(235 * a),))
        cv.text((lx + 8, ly + 12), label, sz, (13, 17, 23), True, anchor="lm")


def header(cv, logo, mode, title_alpha=1.0):
    if logo is not None:
        cv.paste(logo, (28, 14))
    cv.text((76, 33), "Glance-X", 24, FG, True, anchor="lm")
    if mode:
        name, desc = MODE_TEXT[mode]
        w1, w2 = cv.tw(name, 16, mono=True), cv.tw(desc, 16)
        x1 = W - 32 - (w1 + w2 + 54)
        cv.rrect([x1, 16, W - 32, 50], 17, fill=PANEL, outline=LINE)
        cv.d.ellipse([(x1 + 14) * S, 29 * S, (x1 + 22) * S, 37 * S], fill=ORANGE if mode != "text_first" else BLUE)
        cv.text((x1 + 30, 33), name, 16, FG, mono=True, anchor="lm")
        cv.text((x1 + 40 + w1, 33), desc, 16, MUTED, anchor="lm")


# ------------------------------------------------------------------ scenes
def scene_ask(sc, fonts, logo, root, per=1.5, hold=2.2, title=None):
    """Image left, requests right; each request types in, then its answer draws on the image."""
    img = Fitted(os.path.join(root, sc["image"]), 32, 76, 780, 612)
    res = sc["results"]
    multi = sc.get("multi", False)
    n = len(res)
    total = 0.6 + n * per + hold
    frames = []
    for fi in range(int(total * FPS)):
        t = fi / FPS
        cv = Canvas(fonts)
        header(cv, logo, sc["mode"])
        img.draw(cv)
        y = 76
        if title:
            cv.text((836, y + 8), title, 15, MUTED, True)
            y += 34
        for i, r in enumerate(res):
            t0 = 0.6 + i * per
            if t < t0:
                break
            col = PALETTE[i % len(PALETTE)]
            lt = t - t0
            typed = r["request"][: max(1, int(len(r["request"]) * clamp01(lt / 0.45)))]
            lines = cv.wrap(r["request"], 17, 340)[:3]
            ch = 26 + 23 * len(lines) + 26
            a = ease(lt / 0.25)
            active = t < t0 + per or i == n - 1
            cv.rrect([836, y, 1248, y + ch], 12, fill=PANEL + (int(255 * a),),
                     outline=(col if active else LINE) + (int(255 * a),), width=2 if active else 1)
            cv.rrect([836, y, 842, y + ch], 3, fill=col + (int(255 * a),))
            # typed text, wrapped like the full request
            shown, k = [], len(typed)
            for ln in lines:
                shown.append(ln[:k])
                k = max(0, k - len(ln) - (1 if " " in r["request"] else 0))
            for j, ln in enumerate(shown):
                cv.text((858, y + 18 + 23 * j), ln, 17, FG)
            yb = y + 18 + 23 * len(lines) + 8
            if lt > 0.5:
                at = clamp01((lt - 0.5) / 0.3)
                if r["present_prob"] >= 0.5:
                    nb = len(r["boxes"] or [r["box"]]) if multi else 1
                    s = (f"found {nb}" if multi else "found") + f"  ·  P(present) {r['present_prob']:.2f}"
                    cv.text((858, yb), s, 14, mix(col, at, PANEL), True)
                else:
                    cv.text((858, yb), f"not in the image  ·  P(present) {r['present_prob']:.2f}", 14,
                            mix(RED, at, PANEL), True)
            if lt > 0.55 and r["present_prob"] >= 0.5:
                bt = clamp01((lt - 0.55) / 0.45)
                boxes = (r["boxes"] or [r["box"]]) if multi else [r["box"]]
                for j, b in enumerate(boxes):
                    draw_box(cv, img.map(b), col, bt, label=str(i + 1), thick=4 if active else 3,
                             fill_a=0.14 if active else 0.0)
            y += ch + 10
        frames.append(cv.frame())
    return frames


def scene_search(sc, fonts, logo, root, hold=2.4):
    """One request over a grid of images; every image answers, then they are ranked by confidence."""
    res = sc["results"]
    n = len(res)
    cols = 3 if n > 4 else 2
    rows = math.ceil(n / cols)
    gx, gy, gw, gh, gap = 32, 132, W - 64, H - 132 - 24, 14
    tw, th = (gw - gap * (cols - 1)) / cols, (gh - gap * (rows - 1)) / rows
    tiles = [Fitted(os.path.join(root, sc["images"][r["image_index"]]), gx + (i % cols) * (tw + gap),
                    gy + (i // cols) * (th + gap), tw, th, radius=10) for i, r in enumerate(res)]
    order = sorted(range(n), key=lambda i: -res[i]["confidence"])
    rank = {i: k for k, i in enumerate(order)}
    total = 1.2 + n * 0.45 + 1.2 + hold
    frames = []
    for fi in range(int(total * FPS)):
        t = fi / FPS
        cv = Canvas(fonts)
        header(cv, logo, sc["mode"])
        q = sc["request"]
        typed = q[: max(1, int(len(q) * clamp01(t / 0.7)))]
        qw = cv.tw(q, 22, True)
        cv.rrect([W / 2 - qw / 2 - 52, 72, W / 2 + qw / 2 + 28, 116], 22, fill=PANEL, outline=BLUE, width=2)
        cx, cy = W / 2 - qw / 2 - 30, 94
        cv.d.ellipse([(cx - 8) * S, (cy - 8) * S, (cx + 5) * S, (cy + 5) * S], outline=BLUE, width=3 * S)
        cv.d.line([(cx + 4) * S, (cy + 4) * S, (cx + 10) * S, (cy + 10) * S], fill=BLUE, width=3 * S)
        cv.text((W / 2 - qw / 2 - 10, 94), typed, 22, FG, True, anchor="lm")
        for i, (tile, r) in enumerate(zip(tiles, res)):
            t0 = 1.2 + i * 0.45
            ranked = t > 1.2 + n * 0.45 + 0.6
            present = r["present_prob"] >= 0.5
            dim = 0.0
            if t >= t0 + 0.4 and not present:
                dim = 0.62 * clamp01((t - t0 - 0.4) / 0.3)
            if ranked and rank[i] > 0:
                dim = max(dim, 0.25 if present else 0.62)
            tile.draw(cv, dim)
            if t < t0:
                # scanning shimmer on unanswered tiles
                ph = (t * 1.4 + i * 0.13) % 1.0
                sx = tile.x + ph * tile.w
                cv.d.rectangle([(sx - 30) * S, tile.y * S, sx * S, (tile.y + tile.h) * S], fill=BLUE + (28,))
                continue
            bt = clamp01((t - t0) / 0.45)
            if present:
                draw_box(cv, tile.map(r["box"]), BLUE if not (ranked and rank[i] == 0) else ORANGE, bt,
                         label=f"{r['confidence']:.2f}", thick=3)
            elif bt > 0.5:
                s = f"not here · {r['present_prob']:.2f}"
                sw = cv.tw(s, 15, True)
                mx, my = tile.x + tile.w / 2, tile.y + tile.h / 2
                cv.rrect([mx - sw / 2 - 12, my - 15, mx + sw / 2 + 12, my + 15], 15, fill=(30, 20, 24, 230))
                cv.text((mx, my), s, 15, RED, True, anchor="mm")
            if ranked and present:
                a = ease((t - (1.2 + n * 0.45 + 0.6)) / 0.4)
                badge = f"#{rank[i] + 1}"
                c = ORANGE if rank[i] == 0 else BLUE
                cv.rrect([tile.x + tile.w - 58, tile.y + 10, tile.x + tile.w - 10, tile.y + 40], 15,
                         fill=c + (int(240 * a),))
                cv.text((tile.x + tile.w - 34, tile.y + 25), badge, 16, (13, 17, 23), True, anchor="mm")
        frames.append(cv.frame())
    return frames


def card(fonts, banner, lines, total=2.6, fade=0.5):
    frames = []
    for fi in range(int(total * FPS)):
        t = fi / FPS
        a = min(ease(t / fade), ease((total - t) / fade))
        cv = Canvas(fonts)
        if banner is not None:
            bw, bh = banner.size
            cv.paste(banner, (W / 2 - bw / S / 2, 210))
        for j, (s, size, col) in enumerate(lines):
            cv.text((W / 2, 520 + 44 * j), s, size, col, j == 0, anchor="mm")
        f = cv.frame()
        if a < 1:
            f = Image.blend(Image.new("RGB", f.size, BG), f, a)
        frames.append(f)
    return frames


def svg_png(svg, width):
    out = tempfile.mktemp(suffix=".png")
    subprocess.run(["rsvg-convert", "-w", str(width), svg, "-o", out], check=True)
    return Image.open(out).convert("RGBA")


def on_bg(img):
    bg = Image.new("RGB", img.size, BG)
    bg.paste(img, (0, 0), img)
    return bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--image_root", default=".")
    ap.add_argument("--logo", default="assets/logo-dark.svg")
    ap.add_argument("--banner", default="assets/banner-dark.svg")
    ap.add_argument("--out", default="assets/demo.mp4")
    ap.add_argument("--gif", default="")
    ap.add_argument("--gif_width", type=int, default=800)
    ap.add_argument("--gif_fps", type=int, default=12)
    ap.add_argument("--font", default="")
    ap.add_argument("--font_cjk", default="")
    ap.add_argument("--font_mono", default="")
    a = ap.parse_args()
    fonts = Fonts(first([a.font] + FONT_CANDIDATES), first([a.font_cjk] + CJK_CANDIDATES),
                  first([a.font_mono] + MONO_CANDIDATES))
    logo = on_bg(svg_png(a.logo, 40 * S))
    banner = on_bg(svg_png(a.banner, 900 * S))
    pred = json.load(open(a.pred))
    frames = card(fonts, banner, [("Find any object from a text request — both ways.", 26, FG),
                                  ("real outputs of Glance-X-Lite (0.8B) on held-out images", 18, MUTED)], total=3.0)
    for sc in pred:
        if sc["kind"] == "search":
            frames += scene_search(sc, fonts, logo, a.image_root)
        else:
            frames += scene_ask(sc, fonts, logo, a.image_root, title=sc.get("title"))
    frames += card(fonts, banner, [("github.com/Vision-Decision/Glance-X", 26, FG),
                                   ("built on Qwen3.5-0.8B-Base · trained on RefCOCO family + Glance-Data", 18, MUTED),
                                   ("images: MS COCO (cocodataset.org), Flickr terms of use", 15, DIM)], total=3.2)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                          "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    print(f"{len(frames)} frames ({len(frames) / FPS:.1f}s) -> {a.out}")
    if a.gif:
        vf = (f"fps={a.gif_fps},scale={a.gif_width}:-1:flags=lanczos,split[s0][s1];"
              f"[s0]palettegen=max_colors=160:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.out, "-vf", vf, a.gif], check=True)
        print(f"-> {a.gif} ({os.path.getsize(a.gif) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
