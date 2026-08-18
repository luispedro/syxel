import io
import re

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp

from syxel.sixel import rgb_to_palette, write_sixel

# The fixed fallback cube in rgb_to_palette is 5 x 9 x 5
CUBE_SIZE = 5 * 9 * 5


def reconstruct(active, res):
    '''Rebuild an RGB image from the palette and the indexed image'''
    return active[res].astype(np.uint8)


def n_distinct(rgb):
    return len(np.unique(rgb.reshape((-1, 3)), axis=0))


def test_few_colours_are_reproduced_exactly():
    '''An image with a handful of colours needs no quantization at all'''
    rgb = np.zeros((60, 60, 3), np.uint8)
    rgb[:30, :30] = [255, 0, 0]
    rgb[:30, 30:] = [0, 255, 0]
    rgb[30:, :30] = [0, 0, 255]
    rgb[30:, 30:] = [10, 20, 30]

    active, res = rgb_to_palette(rgb)

    assert len(active) == 4
    assert np.array_equal(reconstruct(active, res), rgb)


def test_255_colours_are_reproduced_exactly():
    '''255 colours is exactly the number of registers the palette path uses'''
    colours = np.arange(255, dtype=np.uint8)[:, None] * np.array([1, 1, 1], np.uint8)
    rgb = colours.reshape((15, 17, 3))

    active, res = rgb_to_palette(rgb)

    assert len(active) == 255
    assert np.array_equal(reconstruct(active, res), rgb)


def test_single_colour_image():
    rgb = np.full((7, 5, 3), 200, np.uint8)

    active, res = rgb_to_palette(rgb)

    assert len(active) == 1
    assert np.array_equal(reconstruct(active, res), rgb)


def test_many_colours_fall_back_to_the_cube():
    '''With far more than 255 colours, the top 255 cover almost nothing'''
    random = np.random.RandomState(42)
    rgb = random.randint(0, 256, size=(64, 64, 3)).astype(np.uint8)
    assert n_distinct(rgb) > 255

    active, res = rgb_to_palette(rgb)

    assert len(active) == CUBE_SIZE


def test_dominant_colours_are_kept_despite_a_noisy_minority():
    '''A quarter of the image is noise; the flat majority still gets exact colours'''
    random = np.random.RandomState(7)
    rgb = np.zeros((64, 64, 3), np.uint8)
    rgb[:, :] = [17, 34, 51]
    rgb[:16] = random.randint(0, 256, size=(16, 64, 3))
    assert n_distinct(rgb) > 255

    active, res = rgb_to_palette(rgb)

    assert len(active) == 255
    # The flat three quarters are the most frequent colour, so they survive intact
    assert np.array_equal(reconstruct(active, res)[16:], rgb[16:])


colour = st.tuples(*[st.integers(0, 255)] * 3)


@st.composite
def few_colour_images(draw):
    '''Images built from at most 255 distinct colours'''
    n_colours = draw(st.integers(1, 255))
    colours = draw(st.lists(colour, min_size=n_colours, max_size=n_colours,
                            unique=True))
    height = draw(st.integers(1, 12))
    width = draw(st.integers(1, 12))
    index = draw(hnp.arrays(np.intp, (height, width),
                            elements=st.integers(0, n_colours - 1)))
    return np.array(colours, np.uint8)[index]


@settings(max_examples=200, deadline=None)
@given(few_colour_images())
def test_at_most_255_colours_round_trips_exactly(rgb):
    active, res = rgb_to_palette(rgb)

    assert len(active) == n_distinct(rgb)
    assert np.array_equal(reconstruct(active, res), rgb)


@settings(max_examples=50, deadline=None)
@given(few_colour_images())
def test_palette_is_always_addressable_by_a_sixel_register(rgb):
    active, res = rgb_to_palette(rgb)

    assert len(active) < 256
    assert res.dtype == np.uint8
    assert res.max() < len(active)


def rgb_to_palette_slow(rgb):
    '''Reference implementation: the original per-pixel Python loop

    This is what `rgb_to_palette` looked like before it was vectorized. It is
    kept here so that the fast version can be checked against it.
    '''
    from collections import Counter
    cs = Counter([tuple(pix) for pix in rgb.reshape((-1, 3))])

    colours = list(cs.keys())
    colours.sort(key=lambda x: -cs[x])
    active = np.array(colours[:255], dtype=np.int32)
    n_pixels = rgb.shape[0] * rgb.shape[1]
    if sum(cs[tuple(c)] for c in active) < 0.5 * n_pixels:
        active = []
        for r in range(0, 257, 64):
            if r == 256:
                r = 255
            for g in range(0, 257, 32):
                if g == 256:
                    g = 255
                for b in range(0, 257, 64):
                    if b == 256:
                        b = 255
                    active.append([r, g, b])
        active = np.array(active, dtype=np.int32)

    palette = {}
    for c in colours:
        palette[c] = ((active - c)**2).sum(1).argmin()

    res = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            res[i, j] = palette[tuple(rgb[i, j])]
    return active, res


def assert_same_as_slow(rgb):
    '''The fast implementation must agree with the reference exactly'''
    active, res = rgb_to_palette(rgb)
    active_slow, res_slow = rgb_to_palette_slow(rgb)

    assert np.array_equal(active, active_slow)
    assert active.dtype == active_slow.dtype
    assert np.array_equal(res, res_slow)
    assert res.dtype == res_slow.dtype


def test_fast_matches_slow_on_few_colours():
    rgb = np.zeros((12, 18, 3), np.uint8)
    rgb[:6, :9] = [255, 0, 0]
    rgb[:6, 9:] = [0, 255, 0]
    rgb[6:, :9] = [0, 0, 255]
    rgb[6:, 9:] = [10, 20, 30]

    assert_same_as_slow(rgb)


def test_fast_matches_slow_on_the_cube_fallback():
    random = np.random.RandomState(42)
    rgb = random.randint(0, 256, size=(40, 40, 3)).astype(np.uint8)
    assert n_distinct(rgb) > 255

    assert_same_as_slow(rgb)


def test_fast_matches_slow_on_a_gradient():
    '''A greyscale ramp: exactly 200 distinct colours, all equally frequent'''
    rgb = (np.arange(200, dtype=np.uint8)[:, None] % 200) \
                * np.ones((1, 3), np.uint8)
    rgb = rgb.reshape((20, 10, 3))

    assert_same_as_slow(rgb)


def test_fast_matches_slow_when_ties_decide_the_palette():
    '''More than 255 equally frequent colours: the cut is decided by ties

    The first 255 colours to appear win, so the two implementations only agree
    if they order equal counts the same way.
    '''
    random = np.random.RandomState(3)
    colours = np.unique(random.randint(0, 256, size=(300, 3)), axis=0)
    rgb = np.repeat(colours, 4, axis=0).astype(np.uint8)
    rgb = rgb.reshape((-1, 20, 3))
    random.shuffle(rgb)

    assert_same_as_slow(rgb)


def test_fast_matches_slow_on_a_single_colour():
    assert_same_as_slow(np.full((5, 7, 3), 200, np.uint8))


@settings(max_examples=50, deadline=None)
@given(few_colour_images())
def test_fast_matches_slow_on_few_colour_images(rgb):
    assert_same_as_slow(rgb)


@settings(max_examples=25, deadline=None)
@given(hnp.arrays(np.uint8, st.tuples(st.integers(1, 10), st.integers(1, 10),
                                      st.just(3))))
def test_fast_matches_slow_on_arbitrary_images(rgb):
    assert_same_as_slow(rgb)



def decode_sixel(out):
    '''A minimal decoder for the subset of SIXEL that `write_sixel` emits

    Returns the palette image, with -1 wherever no colour was ever set.
    '''
    prefix = b'\x1bP0;0;0q"1;1;'
    assert out.startswith(prefix)
    assert out.endswith(b'\x1b\\')
    body = out[len(prefix):-2].decode('ascii')

    size = re.match(r'(\d+);(\d+)', body)
    width, height = int(size.group(1)), int(size.group(2))
    res = np.full((height, width), -1, np.int64)

    colour = None
    x = y = 0
    i = size.end()
    while i < len(body):
        ch = body[i]
        if ch == '#':
            # `#n;2;r;g;b` defines a register, a bare `#n` selects one
            token = re.match(r'#(\d+)(;2;\d+;\d+;\d+)?', body[i:])
            if token.group(2) is None:
                colour = int(token.group(1))
                x = 0
            i += token.end()
        elif ch == '$':
            x = 0
            i += 1
        elif ch == '-':
            y += 6
            x = 0
            i += 1
        else:
            bits = ord(ch) - 63
            for row in range(6):
                if bits & (1 << row):
                    res[y + row, x] = colour
            x += 1
            i += 1
    return res


def encode(data, active):
    '''Run `write_sixel` into memory and return the bytes'''
    out = io.BytesIO()
    write_sixel(out, np.asarray(data, np.uint8), np.asarray(active, np.int32))
    return out.getvalue()


def test_write_sixel_header_and_terminator():
    data = np.zeros((12, 20), np.uint8)

    out = encode(data, [[255, 0, 0]])

    # Raster attributes are `"1;1;<width>;<height>`
    assert out.startswith(b'\x1bP0;0;0q"1;1;20;12#')
    assert out.endswith(b'\x1b\\')


def test_write_sixel_rescales_the_palette():
    '''SIXEL colour components run from 0 to 100, not from 0 to 255'''
    out = encode(np.zeros((6, 1), np.uint8), [[255, 0, 128]])

    assert b'#0;2;100;0;50' in out


@pytest.mark.parametrize('height', [1, 5, 6, 7, 12, 13])
def test_write_sixel_emits_every_band(height):
    '''Bands are six rows tall; a short final band must still be written'''
    data = np.zeros((height, 4), np.uint8)

    out = encode(data, [[0, 0, 0]])

    assert out.count(b'-') == -(-height // 6)
    assert f';4;{height}'.encode('ascii') in out


@pytest.mark.parametrize('height,expected', [
                            (1, 1),
                            (4, 1 + 2 + 4 + 8),
                            (6, 1 + 2 + 4 + 8 + 16 + 32),
                            ])
def test_write_sixel_sets_only_the_rows_that_exist(height, expected):
    '''A short band sets the bits of its own rows and no others'''
    data = np.zeros((height, 1), np.uint8)

    out = encode(data, [[10, 20, 30]])
    body = out[out.index(b'#0;2;'):]
    pixels = body[body.index(b'#0', 1) + 2:body.index(b'-')]

    assert pixels == bytes([63 + expected])


def test_write_sixel_one_pass_per_colour_in_a_band():
    '''Passes over the same band are separated by `$`, bands by `-`'''
    data = np.zeros((6, 2), np.uint8)
    data[:, 1] = 1

    out = encode(data, [[0, 0, 0], [255, 255, 255]])
    bands = out[out.index(b'#1;2;'):].split(b'-')[0]

    assert bands.count(b'$') == 1


def test_write_sixel_round_trips_a_quantized_image():
    '''The whole pipeline, decoded back into an image'''
    rgb = np.zeros((10, 8, 3), np.uint8)
    rgb[:5] = [200, 100, 0]
    rgb[5:] = [0, 100, 200]

    active, data = rgb_to_palette(rgb)
    assert np.array_equal(decode_sixel(encode(data, active)), data)
