# Syxel

> **Pre-release.** This is a work in progress and has not been released yet.
> Interfaces and output may change, and some rough edges remain.

SIXEL in Python: display images directly in your terminal.

`syxel` is a small, pure-Python package that converts images to the
[SIXEL](https://en.wikipedia.org/wiki/Sixel) escape-sequence protocol and writes
them to standard output. If your terminal speaks SIXEL, the image appears inline,
no image viewer required.

It ships a single command line tool, `imcat`.

## Installation

```bash
pip install .
```

Or, for development (the `imcat` script then tracks your working copy):

```bash
pip install -e .
```

Requires Python 3.11 or later.

### Dependencies

- [imread](https://pypi.org/project/imread/) — image loading
- [numpy](https://numpy.org/)

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

Full option list:

| Option | Meaning |
| --- | --- |
| `--max-height N` | subsample until the image is at most N pixels high (default: 800) |
| `--max-width N` | subsample until the image is at most N pixels wide (default: 1200) |
| `--version` | print the version and exit |
| `--help` | print usage and exit |

### Terminal support

You need a terminal emulator with SIXEL support, such as
[foot](https://codeberg.org/dnkl/foot),
[WezTerm](https://wezfurlong.org/wezterm/),
[mlterm](https://mlterm.sourceforge.net/), or
`xterm` started with SIXEL enabled (`xterm -ti vt340`). In a terminal without
SIXEL support you will just see a wall of escape-sequence bytes.

Since `imcat` writes to `sys.stdout.buffer`, you can also capture the raw byte
stream instead of rendering it:

```bash
imcat image.png > image.six
```

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

## How it works

The pipeline has three stages:

1. **Load** (`load_image`) — read the file with `imread`, subsample by `[::2,::2]`
   until it fits the size limits, expand greyscale to three channels and drop any
   alpha channel. The result is an (M,N,3) uint8 array.

2. **Quantize** (`rgb_to_palette`) — SIXEL supports at most 256 colour registers,
   so colours are counted and the 255 most frequent are kept. If those do not
   cover at least half of the image, a fixed 5x9x5 RGB cube is used instead.
   Every distinct source colour is then mapped to its nearest palette entry by
   squared Euclidean distance.

3. **Emit** (`write_sixel`) — write the escape sequences. The image is processed
   in bands of six rows, with one pass per colour present in the band; each output
   byte encodes one column's six-pixel bitmask.

## Development

Tests use [pytest](https://pytest.org/) and
[hypothesis](https://hypothesis.readthedocs.io/), and are run through
[pixi](https://pixi.sh/):

```bash
pixi run -e test test
```

CI runs the suite on Python 3.11 through 3.14.

## License

MIT (see [COPYING.MIT](COPYING.MIT)).

Copyright (c) 2024–2026 Luis Pedro Coelho
