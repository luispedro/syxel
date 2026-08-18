import numpy as np
import pytest

from syxel.imcat import load_image, parse_args, main


def test_defaults():
    args = parse_args(['image.png'])
    assert args.images == ['image.png']
    assert (args.max_height, args.max_width) == (800, 1200)


def test_multiple_files():
    assert parse_args(['a.png', 'b.png']).images == ['a.png', 'b.png']


def test_size_override():
    args = parse_args(['a.png', '--max-height', '100', '--max-width', '200'])
    assert (args.max_height, args.max_width) == (100, 200)


@pytest.mark.parametrize('argv', [
                            [],
                            ['a.png', '--max-height', '0'],
                            ['a.png', '--max-width', '-1'],
                            ['a.png', '--max-height', 'wide'],
                            ])
def test_bad_arguments(argv):
    # argparse exits with status 2 on a usage error rather than raising
    with pytest.raises(SystemExit) as exc:
        parse_args(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize('flag', ['--help', '--version'])
def test_informational_flags(flag):
    with pytest.raises(SystemExit) as exc:
        parse_args([flag])
    assert exc.value.code == 0


def image_file(tmp_path, ndim=3):
    imread = pytest.importorskip('imread')
    h, w = 24, 32
    y, x = np.mgrid[0:h,0:w]
    im = ((x * 8 + y) % 256).astype(np.uint8)
    if ndim == 3:
        im = np.stack([im, (im // 2), 255 - im], axis=2)
    ofname = str(tmp_path / f'im{ndim}.png')
    imread.imsave(ofname, im)
    return ofname


def test_load_image_max_size(tmp_path):
    ifname = image_file(tmp_path)
    assert load_image(ifname).shape == (24, 32, 3)
    # Subsampling halves until both limits are met
    assert load_image(ifname, max_height=12, max_width=32).shape == (12, 16, 3)
    assert load_image(ifname, max_height=24, max_width=8).shape == (6, 8, 3)


def test_load_image_greyscale(tmp_path):
    data = load_image(image_file(tmp_path, ndim=2))
    assert data.shape == (24, 32, 3)
    assert np.all(data[:,:,0] == data[:,:,2])


def test_main_writes_one_image_per_file(tmp_path, capsysbinary):
    ifname = image_file(tmp_path)
    main([ifname, ifname])
    out = capsysbinary.readouterr().out
    assert out.count(b'\x1bP') == 2
    # Each image is terminated and followed by a newline so that the next one
    # (or the shell prompt) does not land on top of it
    assert out.count(b'\x1b\\\n') == 2
    assert out.endswith(b'\x1b\\\n')
