from typing import Any
from PIL import ImageColor
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
