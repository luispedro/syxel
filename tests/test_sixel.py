import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp

from syxel.sixel import rgb_to_palette

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
