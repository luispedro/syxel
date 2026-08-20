'''A matplotlib backend that draws figures into the terminal using SIXEL

Select it before importing `pyplot`::

    import matplotlib
    matplotlib.use('module://syxel.backend_sixel')

or from the environment::

    MPLBACKEND=module://syxel.backend_sixel python script.py

`show()` then writes the figure to standard output instead of opening a window.
The backend also registers a `sixel` output format, so `savefig` works too::

    fig.savefig('plot.sixel')

Unlike the rest of the package, this module imports matplotlib at the top level:
matplotlib looks up `FigureCanvas`, `FigureManager` and `show` as module
attributes, so they cannot be built lazily. `numpy` and `syxel.sixel` are still
imported inside the functions that use them.
'''

from matplotlib.backend_bases import FigureManagerBase, _Backend
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib._pylab_helpers import Gcf

# Used when the terminal does not report its size (these are the values that
# `imcat` uses for image files)
MAX_WIDTH = 1200
MAX_HEIGHT = 800


def _terminal_size():
    '''The size of the terminal in pixels, or None if it did not report it

    This is `syxel.terminal.terminal_size`, which is where the ioctl lives
    now that `imcat --info` reports the size too.
    '''
    from syxel.terminal import terminal_size
    return terminal_size()


def target_size():
    '''The size, in pixels, that a figure should be scaled to fit into

    The environment variables `SYXEL_MAX_WIDTH` and `SYXEL_MAX_HEIGHT` take
    precedence (they are the only way out for terminals that do not report
    their pixel size); otherwise the terminal is asked, and failing that the
    `MAX_WIDTH` x `MAX_HEIGHT` defaults are used.
    '''
    import os
    reported = None
    width = os.environ.get('SYXEL_MAX_WIDTH')
    height = os.environ.get('SYXEL_MAX_HEIGHT')
    if width is None or height is None:
        reported = _terminal_size()
    if width is None:
        width = reported[0] if reported is not None else MAX_WIDTH
    if height is None:
        height = reported[1] if reported is not None else MAX_HEIGHT
    return int(width), int(height)


def figure_to_rgb(figure, max_width=None, max_height=None):
    '''Render a matplotlib figure, scaled to fit within the given box

    The figure is scaled by adjusting its dpi rather than its size in inches,
    so that the layout is unchanged (font sizes are in points and scale with
    it) and everything is redrawn at the target resolution instead of being
    subsampled. The figure is left exactly as it was found.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
    max_width, max_height : int, optional
        The box to fit into, in pixels (default: `target_size()`)

    Returns
    -------
    rgb : ndarray
        An image with shape (M,N,3) and dtype uint8
    '''
    import numpy as np
    if max_width is None or max_height is None:
        width, height = target_size()
        if max_width is None:
            max_width = width
        if max_height is None:
            max_height = height

    canvas = figure.canvas
    if not isinstance(canvas, FigureCanvasAgg):
        canvas = FigureCanvasAgg(figure)

    natural_width, natural_height = figure.get_size_inches() * figure.dpi
    scale = min(max_width / natural_width, max_height / natural_height)
    dpi = figure.dpi
    try:
        # Below a pixel per inch the renderer has nothing left to draw into
        figure.dpi = max(dpi * scale, 1)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
    finally:
        figure.dpi = dpi
    return _flatten_onto_white(rgba)


def _flatten_onto_white(rgba):
    '''Composite an RGBA image onto a white background

    A figure is normally opaque, but `facecolor='none'` (or a transparent
    savefig) is not and SIXEL has no notion of transparency.
    '''
    import numpy as np
    rgb = rgba[:,:,:3].astype(np.uint32)
    alpha = rgba[:,:,3:].astype(np.uint32)
    if np.all(alpha == 255):
        return rgba[:,:,:3].copy()
    # Rounded division by 255, keeping everything integral
    flat = rgb * alpha + 255 * (255 - alpha) + 127
    return ((flat + flat // 255) // 256).astype(np.uint8)


def write_figure(out, figure, max_width=None, max_height=None, max_colours=None):
    '''Write a matplotlib figure to `out` as a SIXEL escape sequence

    Parameters
    ----------
    out : file-like
        Opened for writing in binary mode (anything with a bytes `write`)
    figure : matplotlib.figure.Figure
    max_width, max_height : int, optional
        The box to fit into, in pixels (default: `target_size()`)
    max_colours : int, optional
        How many colours to quantize to (default:
        `syxel.terminal.colour_registers()`)
    '''
    from syxel.sixel import rgb_to_palette, write_sixel
    from syxel.terminal import colour_registers
    rgb = figure_to_rgb(figure, max_width=max_width, max_height=max_height)
    if max_colours is None:
        max_colours = colour_registers()
    active, data = rgb_to_palette(rgb, max_colours=max_colours)
    write_sixel(out, data, active)


class FigureManagerSixel(FigureManagerBase):
    '''Shows a figure by writing it to standard output'''

    def show(self):
        import sys
        out = sys.stdout.buffer
        write_figure(out, self.canvas.figure)
        # Otherwise the shell prompt (or the next figure) lands on top of this one
        out.write(b'\n')
        out.flush()


class FigureCanvasSixel(FigureCanvasAgg):
    manager_class = FigureManagerSixel
    filetypes = {**FigureCanvasAgg.filetypes,
                 'sixel': 'SIXEL terminal graphics'}

    def print_sixel(self, filename_or_obj, max_colours=None, **kwargs):
        '''Write the figure at its own size, for `savefig`

        Unlike `show`, this does not scale to the terminal: `savefig` is
        expected to honour the figure's `figsize` and `dpi`. The palette still
        comes from the terminal, because a figure has nothing to say about how
        many colour registers there are; pass `max_colours` to pin it.
        '''
        from matplotlib import cbook
        from syxel.sixel import rgb_to_palette, write_sixel
        from syxel.terminal import colour_registers
        import numpy as np
        FigureCanvasAgg.draw(self)
        rgb = _flatten_onto_white(np.asarray(self.buffer_rgba()))
        if max_colours is None:
            max_colours = colour_registers()
        active, data = rgb_to_palette(rgb, max_colours=max_colours)
        with cbook.open_file_cm(filename_or_obj, 'wb') as fh:
            write_sixel(fh, data, active)


# `_Backend` is private, but subclassing it is the only supported way to define
# a backend and is what third-party backends do
@_Backend.export
class _BackendSixel(_Backend):
    FigureCanvas = FigureCanvasSixel
    FigureManager = FigureManagerSixel
    mainloop = None

    @classmethod
    def show(cls, *, block=None):
        super().show(block=block)
        # There is no window to keep around: without this, the next show()
        # would print every figure shown so far all over again
        Gcf.destroy_all()
