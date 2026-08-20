# Syxel

> **Early alpha release.** Interfaces and output may still change, and some
> rough edges remain. Bug reports (and any other feedback) are very welcome:
> please open an issue at
> [github.com/luispedro/syxel/issues](https://github.com/luispedro/syxel/issues).


[![Test Status](https://github.com/luispedro/syxel/actions/workflows/test.yml/badge.svg)](https://github.com/luispedro/syxel/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

SIXEL in Python: display images directly in your terminal.

`syxel` is a small, pure-Python package that converts images to the
[SIXEL](https://en.wikipedia.org/wiki/Sixel) escape-sequence protocol and writes
them to standard output. If your terminal speaks SIXEL, the image appears inline,
no image viewer required.

It ships a command line tool, `imcat`, and a matplotlib backend, so that
`plt.show()` draws your plots in the terminal.

## Installation

```bash
pip install syxel
```

Or, from a checkout, for development (the `imcat` script then tracks your
working copy):

```bash
pip install -e .
```

Requires Python 3.11 or later.

The base install only depends on numpy, which is enough to convert arrays (or
matplotlib figures) to SIXEL. Reading image files and the matplotlib backend each
come as an extra:

```bash
pip install 'syxel[imcat]'         # the imcat command line tool
pip install 'syxel[matplotlib]'    # the matplotlib backend
pip install 'syxel[imcat,matplotlib]'
```

### Dependencies

- [numpy](https://numpy.org/)
- [imread](https://pypi.org/project/imread/) — optional (`imcat` extra), image
  loading for the command line tool
- [matplotlib](https://matplotlib.org/) — optional (`matplotlib` extra), only for
  the backend

## Usage

```bash
imcat image.png
```

Several images can be given at once; each is written followed by a newline, so
they stack vertically:

```bash
imcat one.png two.jpg three.tiff
```

Images larger than 800x1200 are subsampled (halved repeatedly) until they fit.
Override the limits with:

```bash
imcat --max-height 400 --max-width 600 image.png
```

How many colours the image is quantized to is the terminal's business: SIXEL
draws from a palette of colour registers, and terminals differ in how many they
provide (16 on the original DEC VT340, 256 by default today, up to 1024 for
`xterm` configured with `numColorRegisters`). `imcat` asks the terminal with
XTSMGRAPHICS and uses whatever it says, falling back to 255 for terminals that do
not answer. Set it by hand with `--max-colours N`, or for both `imcat` and the
matplotlib backend with the `SYXEL_MAX_COLOURS` environment variable:

```bash
imcat --max-colours 1024 image.png      # or
SYXEL_MAX_COLOURS=1024 imcat image.png
```

To see what the terminal has to say for itself, ask for a report instead of an
image:

```console
$ imcat --info
syxel version:    0.2
Terminal:         foot
SIXEL support:    yes
Colour registers: 1024
Terminal size:    120x40 characters, 1440x960 pixels
Largest image:    1000x1000 pixels
Colours in use:   1024 (from the terminal)
Images fit into:  1200x800 pixels (--max-width/--max-height)
```

Everything above the last two lines is what the terminal answered (`unknown`
where it said nothing at all, which is what a terminal without SIXEL support
usually does); the last two are what `imcat` would do with an image, so
`--info` can be combined with `--max-colours` and the size limits to see their
effect.

Full option list:

| Option | Meaning |
| --- | --- |
| `--max-height N` | subsample until the image is at most N pixels high (default: 800) |
| `--max-width N` | subsample until the image is at most N pixels wide (default: 1200) |
| `--max-colours N` | quantize to at most N colours (default: ask the terminal) |
| `--info` | report what the terminal supports and exit |
| `--version` | print the version and exit |
| `--help` | print usage and exit |

### Terminal support

You need a terminal emulator with SIXEL support, such as
[foot](https://codeberg.org/dnkl/foot),
[WezTerm](https://wezfurlong.org/wezterm/),
[mlterm](https://mlterm.sourceforge.net/), or
`xterm` started with SIXEL enabled (`xterm -ti vt340`). In a terminal without
SIXEL support you will just see a wall of escape-sequence bytes; `imcat --info`
asks the terminal whether it can draw them before you try.

Since `imcat` writes to `sys.stdout.buffer`, you can also capture the raw byte
stream instead of rendering it:

```bash
imcat image.png > image.six
```

## matplotlib backend

`syxel.backend_sixel` is a matplotlib backend that draws figures into the terminal
instead of opening a window. Select it from the environment:

```bash
MPLBACKEND=module://syxel.backend_sixel python plot.py
```

or from Python, before importing `pyplot`:

```python
import matplotlib
matplotlib.use('module://syxel.backend_sixel')

import matplotlib.pyplot as plt
plt.plot([1, 4, 9])
plt.show()                  # the figure appears in the terminal
```

With matplotlib 3.9 or later the short name `matplotlib.use('sixel')` also works,
via an entry point.

The figure is scaled to fill the terminal: its dpi is raised or lowered so that it
fits the window, which redraws it at the right resolution rather than resampling
it. Terminals that do not report their size in pixels (some multiplexers) fall
back to 1200x800; set `SYXEL_MAX_WIDTH` and `SYXEL_MAX_HEIGHT` to override.

The palette comes from the terminal in the same way as for `imcat`, including for
`savefig` (a figure has nothing to say about how many colour registers there are);
`SYXEL_MAX_COLOURS` overrides it.

`savefig` gains a `sixel` format, which writes at the figure's own size:

```python
fig.savefig('plot.sixel')                       # or
fig.savefig(sys.stdout.buffer, format='sixel')
```

Each `plt.show()` prints the figures drawn so far and then drops them, so a script
with several `show()` calls does not reprint earlier figures.

## Using it as a library

The conversion is available directly, independently of the command line tool:

```python
import sys
from syxel.imcat import load_image
from syxel.sixel import rgb_to_palette, write_sixel

rgb = load_image('image.png')        # (M,N,3) uint8 array
active, data = rgb_to_palette(rgb)   # palette (P,3) and indexed image (M,N)
write_sixel(sys.stdout.buffer, data, active)
```

`write_sixel` accepts any object with a `write` method taking bytes, so an
`io.BytesIO` works for testing or for building the sequence in memory.

The same is true of figures, without going through the backend machinery:

```python
from syxel.backend_sixel import write_figure

write_figure(sys.stdout.buffer, fig)
```

## How it works

The pipeline has three stages:

1. **Load** (`load_image`) — read the file with `imread`, subsample by `[::2,::2]`
   until it fits the size limits, expand greyscale to three channels and drop any
   alpha channel. The result is an (M,N,3) uint8 array.

2. **Quantize** (`rgb_to_palette`) — the terminal has a limited number of colour
   registers (`syxel.terminal.colour_registers` asks it how many), so colours are
   counted and the most frequent ones that fit are kept. If those do not cover at
   least half of the image, the colours are rounded — one low bit at a time, from
   blue first, then red, then green, then the next bit of each — until the most
   frequent of what is left do cover half of it, so the registers end up where the
   image actually has colours. Every distinct source colour is then mapped to its
   nearest palette entry by squared Euclidean distance.

3. **Emit** (`write_sixel`) — write the escape sequences. The image is processed
   in bands of six rows, with one pass per colour present in the band; each output
   byte encodes one column's six-pixel bitmask. A final band shorter than six rows
   only sets the bits of the rows that exist.

The matplotlib backend replaces the first stage: it renders the figure through Agg
and hands the resulting pixels to the same last two stages.

## Development

Tests use [pytest](https://pytest.org/) and
[hypothesis](https://hypothesis.readthedocs.io/), and are run through
[pixi](https://pixi.sh/):

```bash
pixi run -e test test
```

CI runs the suite on Python 3.11 through 3.14.

Releases are cut by pushing a `v<version>` tag matching `syxel/syxel_version.py`
(for example `git tag v0.1 && git push origin v0.1`), which builds the sdist and
wheel and uploads them to PyPI through
[trusted publishing](https://docs.pypi.org/trusted-publishers/).

## License

MIT (see [COPYING.MIT](COPYING.MIT)).

Copyright (c) 2024–2026 Luis Pedro Coelho
