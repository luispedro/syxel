# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`syxel` is a small pure-Python package that writes images to the terminal using the
[SIXEL](https://en.wikipedia.org/wiki/Sixel) escape-sequence protocol. It installs a
single console script, `imcat`, which maps to `syxel.imcat:main`.

Runtime dependencies: `imread` (image loading) and `numpy` (used throughout, but
currently only pulled in transitively via `imread`). All imports are done *inside*
functions rather than at module top level.

## Commands

```bash
pip install -e .        # install for development (entry point: imcat)
imcat image.png         # render an image to a SIXEL-capable terminal
```

There is no test suite, linter config, or CI in this repo. Verification is manual:
run `imcat` in a terminal that supports SIXEL (e.g. xterm with sixel enabled, foot,
mlterm, WezTerm) and look at the output. To inspect the byte stream instead of
rendering it, redirect stdout to a file — `main()` writes to `sys.stdout.buffer`.

## Architecture

Everything lives in `syxel/imcat.py`, structured as a three-stage pipeline that
`main()` wires together:

1. `load_image(ifname)` — reads with `imread`, then halves the image by simple
   `[::2,::2]` subsampling until it fits within 800x1200, and drops an alpha channel
   if present. Returns an (M,N,3) uint8 array.

2. `rgb_to_palette(rgb)` — SIXEL supports at most 256 registers, so the image must be
   quantized. The strategy is: count exact colours, take the 255 most frequent; if
   those do not cover at least half the image, fall back to a fixed 5x9x5 RGB cube.
   Every distinct source colour is then mapped to its nearest palette entry by squared
   Euclidean distance. Returns `(active, res)` — the palette as (P,3) and the indexed
   image as (M,N) uint8.

3. `write_sixel(out, data, active)` — emits the escape sequences. Palette values are
   rescaled from 0-255 to the SIXEL 0-100 range. The image is processed in bands of
   six rows; within each band, one pass is emitted per colour present (`#<n>` selects
   the register), where each output byte encodes a column's six-pixel bitmask via a
   dot product with `[1,2,4,8,16,32]` plus 63. `$` returns the cursor to the start of
   the band for the next colour pass, `-` advances to the next band, and `\x1b\\`
   terminates.

`syxel/syxel_version.py` holds `__version__`, read dynamically by `pyproject.toml`.

## Conventions

Commit messages use a short uppercase tag prefix: `ENH` (enhancement), `RFCT`
(refactor). Follow this style — e.g. `BUG`, `DOC`, `TST` for other kinds of change.
