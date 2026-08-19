# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`syxel` is a small pure-Python package that writes images to the terminal using the
[SIXEL](https://en.wikipedia.org/wiki/Sixel) escape-sequence protocol. It installs a
console script, `imcat`, which maps to `syxel.imcat:main`, and a matplotlib backend
(`syxel.backend_sixel`) that draws figures into the terminal.

The only hard runtime dependency is `numpy`. `imread` (image loading, needed by
`load_image` and hence by the `imcat` command) and `matplotlib` (the backend) are
optional extras, `syxel[imcat]` and `syxel[matplotlib]`; `load_image` turns a
missing `imread` into an `ImportError` naming the extra.

All imports are done *inside* functions rather than at module top level;
`syxel/backend_sixel.py` is the one exception, because matplotlib looks up
`FigureCanvas`, `FigureManager` and `show` as module attributes (numpy and
`syxel.sixel` are still imported lazily there).

## Commands

```bash
pip install -e '.[imcat]'    # install for development (entry point: imcat)
imcat image.png              # render an image to a SIXEL-capable terminal
imcat --help                 # one or more files, plus --version/--max-height/--max-width/--max-colours
pixi run -e test test        # run the test suite (pytest + hypothesis, in tests/)

MPLBACKEND=module://syxel.backend_sixel python plot.py   # matplotlib in the terminal
```

There is no linter config. CI (`.github/workflows/test.yml`) runs `python -m pytest`
in the `test-py311` through `test-py314` pixi environments. Beyond the tests,
verification is manual: run `imcat` in a terminal that supports SIXEL (e.g. xterm
with sixel enabled, foot, mlterm, WezTerm) and look at the output. To inspect the
byte stream instead of rendering it, redirect stdout to a file — `main()` writes to
`sys.stdout.buffer`.

## Architecture

The code is a three-stage pipeline that `syxel/imcat.py`'s `main()` wires together.
`imcat.py` holds the image loading and the `argparse`-based command line entry point
(`parse_args` builds the parser, `main(argv=None)` renders each file and writes a
newline after it); the SIXEL conversion itself lives in `syxel/sixel.py` (imported
inside `main()`).

1. `load_image(ifname, max_height=800, max_width=1200)` (`imcat.py`) — reads with
   `imread`, then halves the image by simple `[::2,::2]` subsampling until it fits
   within the limits (overridable with `--max-height` / `--max-width`), drops an
   alpha channel if present, and rescales the dtype to uint8 (`_as_uint8`:
   integer types from the full range of their dtype, floats clipped to [0, 1]).
   Returns an (M,N,3) uint8 array.

2. `rgb_to_palette(rgb, max_colours=None)` (`sixel.py`) — the terminal has a
   limited number of colour registers, so the image must be quantized. The
   strategy is: count exact colours, take the `max_colours` most frequent; if
   those do not cover at least half the image, fall back to a fixed RGB cube
   (`_fixed_cube`, the largest n x (2n-1) x n that fits: 5x9x5 = 225 for the
   default 255, 8x15x8 = 960 for 1024). Every distinct source colour is then
   mapped to its nearest palette entry by squared Euclidean distance. Returns
   `(active, res)` — the palette as (P,3) and the indexed image as (M,N) uint8,
   or uint16 when the palette has more than 256 entries. `max_colours=None`
   means `DEFAULT_COLOURS` (255, what the code assumed before it could ask).

3. `write_sixel(out, data, active)` (`sixel.py`) — emits the escape sequences.
   Palette values are rescaled from 0-255 to the SIXEL 0-100 range. The image is
   processed in bands of six rows; within each band, one pass is emitted per colour
   present (`#<n>` selects the register), where each output byte encodes a column's
   six-pixel bitmask via a dot product with `[1,2,4,8,16,32]` plus 63. Each pass is
   then run-length encoded by `_rle`, which drops trailing empty sixels and writes
   runs as `!<n><byte>` when that is shorter (repeat counts capped at 255). `$`
   returns the cursor to the start of the band for the next colour pass, `-`
   advances to the next band, and `\x1b\\` terminates.

`syxel/backend_sixel.py` is a fourth entry point into stages 2 and 3: matplotlib's
Agg backend replaces stage 1. `figure_to_rgb` renders a figure scaled to the
terminal (queried with `TIOCGWINSZ`, overridable with `SYXEL_MAX_WIDTH` /
`SYXEL_MAX_HEIGHT`) by temporarily adjusting the figure's dpi, and `write_figure`
runs the rest of the pipeline. `FigureManagerSixel.show` writes to
`sys.stdout.buffer`, while `FigureCanvasSixel.print_sixel` registers a `sixel`
`savefig` format that renders at the figure's own size. The backend is exported with
matplotlib's private `_Backend` class, which is the only supported way to define one.

`syxel/terminal.py` asks the terminal how many colour registers it has, with the
XTSMGRAPHICS query `\x1b[?1;1;0S`. It talks to `/dev/tty` rather than stdout (so
the answer is right even when the SIXEL stream is redirected), skips the query
from a background process group (SIGTTIN/SIGTTOU would stop the process), uses
cbreak mode with `TCSADRAIN` (`setcbreak` defaults to `TCSAFLUSH`, which would
discard the answer), and gives up after `QUERY_TIMEOUT`. `colour_registers()`
returns None when the terminal cannot be asked, caches the answer for the
process, and is overridden by `SYXEL_MAX_COLOURS`/`SYXEL_MAX_COLORS`. Both entry
points call it and pass the result to `rgb_to_palette`; `imcat --max-colours`
and the backend's `max_colours=` arguments bypass it.

`syxel/syxel_version.py` holds `__version__`, read dynamically by `pyproject.toml`.

## Conventions

Commit messages use a short uppercase tag prefix: `ENH` (enhancement), `RFCT`
(refactor). Follow this style — e.g. `BUG`, `DOC`, `TST` for other kinds of change.

User-visible changes (new features, bug fixes, behaviour or interface changes) are
recorded in `ChangeLog`, newest version first, under a `Version <n>  <date>` heading
with one tab-indented line per change. Add an entry as part of the change itself;
if the top of the file is an already released version, start an unreleased section
above it. Purely internal work (refactors, tests, CI, documentation) is not listed.
