from typing import Any
from PIL import ImageColor, Image
import aggdraw

class RectShape:
    x1: int
    x2: int
    y1: int
    y2: int
    _fill: Any
    _outline: Any

    @property
    def fill(self):
        return self._fill

    @property
    def outline(self):
        return self._outline

    @fill.setter
    def fill(self, v):
        self._fill = get_rgba_tuple(v)

    @outline.setter
    def outline(self, v):
        self._outline = get_rgba_tuple(v)

    def _get_pen_brush(self):
        pen = aggdraw.Pen(self._outline)
        brush = aggdraw.Brush(self._fill)
        return pen, brush

    def to_tikz(self):
        # Base method: subclasses should override
        raise NotImplementedError("to_tikz() not implemented in base class")


class Box(RectShape):
    de: int = 0
    shade: int = 20

    def draw(self, draw, draw_reversed: bool = False):
        pen, brush = self._get_pen_brush()

        if hasattr(self, 'de') and self.de > 0:
            brush_s1 = aggdraw.Brush(fade_color(self.fill, self.shade))
            brush_s2 = aggdraw.Brush(fade_color(self.fill, 2 * self.shade))

            if draw_reversed:
                draw.line([self.x2 - self.de, self.y1 - self.de,
                          self.x2 - self.de, self.y2 - self.de], pen)
                draw.line([self.x2 - self.de, self.y2 -
                          self.de, self.x2, self.y2], pen)
                draw.line([self.x1 - self.de, self.y2 - self.de,
                          self.x2 - self.de, self.y2 - self.de], pen)

                draw.polygon([self.x1, self.y1,
                              self.x1 - self.de, self.y1 - self.de,
                              self.x2 - self.de, self.y1 - self.de,
                              self.x2, self.y1
                              ], pen, brush_s1)

                draw.polygon([self.x1 - self.de, self.y1 - self.de,
                              self.x1, self.y1,
                              self.x1, self.y2,
                              self.x1 - self.de, self.y2 - self.de
                              ], pen, brush_s2)
            else:
                draw.line([self.x1 + self.de, self.y1 - self.de,
                          self.x1 + self.de, self.y2 - self.de], pen)
                draw.line([self.x1 + self.de, self.y2 -
                          self.de, self.x1, self.y2], pen)
                draw.line([self.x1 + self.de, self.y2 - self.de,
                          self.x2 + self.de, self.y2 - self.de], pen)

                draw.polygon([self.x1, self.y1,
                              self.x1 + self.de, self.y1 - self.de,
                              self.x2 + self.de, self.y1 - self.de,
                              self.x2, self.y1
                              ], pen, brush_s1)

                draw.polygon([self.x2 + self.de, self.y1 - self.de,
                              self.x2, self.y1,
                              self.x2, self.y2,
                              self.x2 + self.de, self.y2 - self.de
                              ], pen, brush_s2)

        draw.rectangle([self.x1, self.y1, self.x2, self.y2], pen, brush)

    def to_tikz(self):
        fill_color = rgba_to_tikz_color(self.fill)
        draw_color = rgba_to_tikz_color(self.outline)
        # Coordinates for TikZ: note y-flip (TikZ y-axis goes up)
        x1, y1, x2, y2 = self.x1, -self.y1, self.x2, -self.y2

        # Basic rectangle
        tikz = (
            f"\\filldraw[fill={fill_color}, draw={draw_color}] "
            f"({x1},{y1}) rectangle ({x2},{y2});"
        )
        return tikz


class Circle(RectShape):
    def draw(self, draw):
        pen, brush = self._get_pen_brush()
        draw.ellipse([self.x1, self.y1, self.x2, self.y2], pen, brush)

    def to_tikz(self):
        fill_color = rgba_to_tikz_color(self.fill)
        draw_color = rgba_to_tikz_color(self.outline)
        cx = (self.x1 + self.x2) / 2
        cy = -(self.y1 + self.y2) / 2  # flip y
        rx = abs(self.x2 - self.x1) / 2
        ry = abs(self.y2 - self.y1) / 2
        if abs(rx - ry) < 1e-3:
            # Circle
            tikz = (
                f"\\filldraw[fill={fill_color}, draw={draw_color}] "
                f"({cx},{cy}) circle ({rx});"
            )
        else:
            # Ellipse
            tikz = (
                f"\\filldraw[fill={fill_color}, draw={draw_color}] "
                f"({cx},{cy}) ellipse ({rx} and {ry});"
            )
        return tikz


class Ellipses(RectShape):
    def draw(self, draw):
        pen, brush = self._get_pen_brush()
        w = self.x2 - self.x1
        d = int(w / 7)
        draw.ellipse([self.x1 + (w - d) / 2, self.y1 + 1 * d,
                     self.x1 + (w + d) / 2, self.y1 + 2 * d], pen, brush)
        draw.ellipse([self.x1 + (w - d) / 2, self.y1 + 3 * d,
                     self.x1 + (w + d) / 2, self.y1 + 4 * d], pen, brush)
        draw.ellipse([self.x1 + (w - d) / 2, self.y1 + 5 * d,
                     self.x1 + (w + d) / 2, self.y1 + 6 * d], pen, brush)

    def to_tikz(self):
        fill_color = rgba_to_tikz_color(self.fill)
        draw_color = rgba_to_tikz_color(self.outline)
        w = self.x2 - self.x1
        d = w / 7

        cx = self.x1 + w / 2
        # Positions of three small ellipses vertically spaced, flipped y
        centers = [
            (cx, -(self.y1 + 1.5 * d)),
            (cx, -(self.y1 + 3.5 * d)),
            (cx, -(self.y1 + 5.5 * d))
        ]
        radius_x = d / 2
        radius_y = d / 2

        tikz = ""
        for (x, y) in centers:
            tikz += (
                f"\\filldraw[fill={fill_color}, draw={draw_color}] "
                f"({x},{y}) ellipse ({radius_x} and {radius_y});\n"
            )
        return tikz

class ColorWheel:
    def __init__(self, colors: list = None):
        self._cache = dict()
        self.colors = colors if colors is not None else [
            "#ffd166", "#ef476f", "#118ab2", "#073b4c", "#842da1",
            "#ffbad4", "#fe9775", "#83d483", "#06d6a0", "#0cb0a9"]

    def get_color(self, class_type: type):
        if class_type not in self._cache.keys():
            index = len(self._cache.keys()) % len(self.colors)
            self._cache[class_type] = self.colors[index]
        return self._cache.get(class_type)
# Helper functions for color and saving

def rgba_to_tikz_color(rgba):
    r, g, b, a = rgba
    # Return TikZ rgb format string with normalized rgb in 0-1
    return f"{{rgb,255:red,{r};green,{g};blue,{b}}}"


def get_rgba_tuple(color: Any) -> tuple:
    """
    Convert color to (R, G, B, A) tuple.
    """
    if isinstance(color, tuple):
        rgba = color
    elif isinstance(color, int):
        rgba = (color >> 16 & 0xff, color >> 8 & 0xff,
                color & 0xff, color >> 24 & 0xff)
    else:
        rgba = ImageColor.getrgb(color)

    if len(rgba) == 3:
        rgba = (rgba[0], rgba[1], rgba[2], 255)
    return rgba


def fade_color(color: tuple, fade_amount: int) -> tuple:
    r = max(0, color[0] - fade_amount)
    g = max(0, color[1] - fade_amount)
    b = max(0, color[2] - fade_amount)
    return r, g, b, color[3]


def save_shapes_as_latex(shapes, filename="output.tex", scale=1.0):
    tikz_commands = []
    for shape in shapes:
        tikz_code = shape.to_tikz()
        # Optionally scale coordinates in tikz_code by `scale` if needed (left as future)
        tikz_commands.append(tikz_code)

    document = r"""\documentclass[tikz,border=3mm]{standalone}
\usepackage{xcolor}
\begin{document}
\begin{tikzpicture}[scale=%f, yscale=-1] %% yscale=-1 flips y-axis to match image coords
%s
\end{tikzpicture}
\end{document}""" % (scale, "\n".join(tikz_commands))

    with open(filename, "w") as f:
        f.write(document)
    print(f"LaTeX TikZ code saved to {filename}")




def get_keys_by_value(d, v):
    for key in d.keys():
        if d[key] == v:
            yield key


def self_multiply(tensor_tuple: tuple):
    tensor_list = list(tensor_tuple)
    if None in tensor_list:
        tensor_list.remove(None)
    if len(tensor_list) == 0:
        return 0
    s = tensor_list[0]
    for i in range(1, len(tensor_list)):
        s *= tensor_list[i]
    return s


def vertical_image_concat(im1: Image, im2: Image, background_fill: Any = 'white'):
    dst = Image.new('RGBA', (max(im1.width, im2.width),
                    im1.height + im2.height), background_fill)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def linear_layout(images: list, max_width: int = -1, max_height: int = -1, horizontal: bool = True,
                  padding: int = 0, spacing: int = 0, background_fill: Any = 'white'):
    coords = []
    width = 0
    height = 0
    x, y = padding, padding

    for img in images:
        if horizontal:
            if max_width != -1 and x + img.width > max_width:
                x = padding
                y = height - padding + spacing
            coords.append((x, y))
            width = max(x + img.width + padding, width)
            height = max(y + img.height + padding, height)
            x += img.width + spacing
        else:
            if max_height != -1 and y + img.height > max_height:
                x = width - padding + spacing
                y = padding
            coords.append((x, y))
            width = max(x + img.width + padding, width)
            height = max(y + img.height + padding, height)
            y += img.height + spacing

    layout = Image.new('RGBA', (width, height), background_fill)
    for img, coord in zip(images, coords):
        layout.paste(img, coord)
    return layout
