import io

import numpy as np
import pytest

matplotlib = pytest.importorskip('matplotlib')

from matplotlib.figure import Figure
from matplotlib._pylab_helpers import Gcf

from syxel.backend_sixel import (FigureCanvasSixel, FigureManagerSixel,
                                 MAX_HEIGHT, MAX_WIDTH,
                                 figure_to_rgb, target_size, write_figure,
                                 _terminal_size)


def a_figure(figsize=(4, 3), dpi=100, **kwargs):
    '''A small figure with something drawn in it

    The canvas is attached explicitly because these figures are built without
    pyplot; selecting the backend does it for you.
    '''
    fig = Figure(figsize=figsize, dpi=dpi, **kwargs)
    fig.subplots().plot([1, 4, 9])
    FigureCanvasSixel(fig)
    return fig


@pytest.fixture
def sixel_backend():
    '''Select the backend for the duration of a test, then put it back'''
    previous = matplotlib.get_backend()
    matplotlib.use('module://syxel.backend_sixel')
    yield
    matplotlib.use(previous)


@pytest.fixture
def small_terminal(monkeypatch):
    '''Keep the figures that go to stdout small'''
    monkeypatch.setenv('SYXEL_MAX_WIDTH', '120')
    monkeypatch.setenv('SYXEL_MAX_HEIGHT', '90')


def test_terminal_size_is_unknown_when_capturing():
    '''pytest replaces stdout with something that is not a terminal'''
    assert _terminal_size() is None


def test_target_size_falls_back_to_the_defaults(monkeypatch):
    monkeypatch.delenv('SYXEL_MAX_WIDTH', raising=False)
    monkeypatch.delenv('SYXEL_MAX_HEIGHT', raising=False)

    assert target_size() == (MAX_WIDTH, MAX_HEIGHT)


def test_target_size_honours_the_environment(monkeypatch):
    monkeypatch.setenv('SYXEL_MAX_WIDTH', '640')
    monkeypatch.setenv('SYXEL_MAX_HEIGHT', '480')

    assert target_size() == (640, 480)


def test_figure_to_rgb_shrinks_to_fit():
    rgb = figure_to_rgb(a_figure(figsize=(8, 6), dpi=100),
                        max_width=400, max_height=300)

    assert rgb.dtype == np.uint8
    assert rgb.shape[2] == 3
    assert rgb.shape[0] <= 300 and rgb.shape[1] <= 400
    # Scaling is uniform, so one of the two limits is (nearly) reached
    assert rgb.shape[0] >= 299 or rgb.shape[1] >= 399


def test_figure_to_rgb_grows_to_fill():
    '''A small figure is redrawn at a higher dpi rather than left tiny'''
    rgb = figure_to_rgb(a_figure(figsize=(2, 1.5), dpi=50),
                        max_width=800, max_height=600)

    assert rgb.shape[:2] == (600, 800)


def test_figure_to_rgb_keeps_the_aspect_ratio():
    rgb = figure_to_rgb(a_figure(figsize=(4, 2), dpi=100),
                        max_width=1000, max_height=1000)

    assert rgb.shape[1] == 2 * rgb.shape[0]


def test_figure_to_rgb_leaves_the_figure_alone():
    fig = a_figure(figsize=(4, 3), dpi=100)

    figure_to_rgb(fig, max_width=137, max_height=91)

    assert fig.dpi == 100
    assert tuple(fig.get_size_inches()) == (4, 3)


def test_figure_to_rgb_flattens_transparency():
    '''SIXEL has no transparency, so a bare figure comes out white'''
    fig = Figure(figsize=(1, 1), dpi=50, facecolor='none')

    rgb = figure_to_rgb(fig, max_width=50, max_height=50)

    assert np.all(rgb == 255)


def test_write_figure_is_a_complete_sixel_stream():
    out = io.BytesIO()

    write_figure(out, a_figure(), max_width=200, max_height=200)
    written = out.getvalue()

    assert written.startswith(b'\x1bP')
    assert written.endswith(b'\x1b\\')


def test_write_figure_declares_the_rendered_size():
    '''The raster attributes are `"1;1;<width>;<height>`'''
    fig = a_figure(figsize=(4, 3), dpi=100)
    rgb = figure_to_rgb(fig, max_width=200, max_height=200)
    out = io.BytesIO()

    write_figure(out, fig, max_width=200, max_height=200)

    height, width = rgb.shape[:2]
    assert out.getvalue().startswith(f'\x1bP0;0;0q"1;1;{width};{height}'
                                     .encode('ascii'))


def test_sixel_is_a_supported_savefig_format():
    canvas = FigureCanvasSixel(a_figure())

    assert 'sixel' in canvas.get_supported_filetypes()


def test_savefig_to_a_file_object():
    out = io.BytesIO()

    a_figure().savefig(out, format='sixel')

    assert out.getvalue().startswith(b'\x1bP')
    assert out.getvalue().endswith(b'\x1b\\')


def test_savefig_to_a_path(tmp_path):
    ofname = tmp_path / 'plot.sixel'

    a_figure().savefig(ofname)

    assert ofname.read_bytes().endswith(b'\x1b\\')


def test_savefig_uses_the_figure_size_not_the_terminal():
    '''`savefig` must honour figsize/dpi, unlike `show`'''
    out = io.BytesIO()

    a_figure(figsize=(3, 2), dpi=80).savefig(out, format='sixel')

    assert out.getvalue().startswith(b'\x1bP0;0;0q"1;1;240;160')


def test_manager_show_writes_to_stdout(small_terminal, capsysbinary):
    fig = a_figure()
    manager = FigureCanvasSixel.new_manager(fig, 1)
    assert isinstance(manager, FigureManagerSixel)

    manager.show()
    written = capsysbinary.readouterr().out

    assert written.startswith(b'\x1bP')
    # Terminated and followed by a newline so that the shell prompt does not
    # land on top of the figure
    assert written.endswith(b'\x1b\\\n')
    Gcf.destroy_all()


def test_pyplot_show_prints_each_figure_once(sixel_backend, small_terminal,
                                             capsysbinary):
    '''Figures are dropped once shown; a second show() must not repeat them'''
    import matplotlib.pyplot as plt

    plt.figure(figsize=(2, 1.5), dpi=40)
    plt.plot([1, 4, 9])
    plt.show()
    plt.figure(figsize=(2, 1.5), dpi=40)
    plt.plot([3, 2, 1])
    plt.show()
    written = capsysbinary.readouterr().out

    assert written.count(b'\x1bP') == 2
    assert written.count(b'\x1b\\\n') == 2
    assert not Gcf.get_all_fig_managers()


def test_pyplot_uses_the_sixel_canvas(sixel_backend):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    try:
        assert isinstance(fig.canvas, FigureCanvasSixel)
        assert isinstance(fig.canvas.manager, FigureManagerSixel)
    finally:
        plt.close(fig)
