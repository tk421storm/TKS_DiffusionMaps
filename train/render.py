"""ASCII and PNG renderings of category grids. Rows are stored south first; north is drawn at the top."""
import numpy as np

# RGB per category name; unknown names fall back to magenta.
COLORS = {
    "empty": (38, 46, 34), "wall": (200, 200, 200), "door": (230, 160, 40), "floor": (120, 95, 70),
    "bridge": (70, 120, 200), "fence": (150, 200, 90), "wall_mounted": (220, 70, 70),
}


def ascii_grid(cells, chars):
    return "\n".join("".join(chars.get(int(v), "?") for v in row).rstrip() for row in cells[::-1])


def ascii_sheet(windows, chars, per_row=2, gap=3):
    blocks = [ascii_grid(w, chars).split("\n") for w in windows]
    W = windows.shape[2]
    out = []
    for i in range(0, len(blocks), per_row):
        group = blocks[i:i + per_row]
        h = max(len(b) for b in group)
        for r in range(h):
            out.append((" " * gap).join((b[r] if r < len(b) else "").ljust(W) for b in group).rstrip())
        out.append("")
    return "\n".join(out)


def png_sheet(windows, names, path, cols=8, scale=4, gap=2):
    from PIL import Image
    n, H, W = windows.shape
    rows = (n + cols - 1) // cols
    img = np.full((rows * (H * scale + gap) + gap, cols * (W * scale + gap) + gap, 3), 12, np.uint8)
    lut = np.array([COLORS.get(names.get(i, ""), (255, 0, 255)) for i in range(max(names) + 1)], np.uint8)
    for i, w in enumerate(windows):
        r, c = divmod(i, cols)
        tile = lut[w[::-1]].repeat(scale, 0).repeat(scale, 1)
        y, x = gap + r * (H * scale + gap), gap + c * (W * scale + gap)
        img[y:y + H * scale, x:x + W * scale] = tile
    Image.fromarray(img).save(path)
