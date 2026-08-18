import numpy as np
import pytest

from syxel.imcat import _as_uint8, load_image, parse_args, main


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


def image_file(tmp_path, ndim=3, dtype=np.uint8):
    imread = pytest.importorskip('imread')
    h, w = 24, 32
    y, x = np.mgrid[0:h,0:w]
    im = ((x * 8 + y) % 256).astype(np.uint8)
    if ndim == 3:
        im = np.stack([im, (im // 2), 255 - im], axis=2)
    if dtype == np.uint16:
        im = im.astype(np.uint16) * 257
    ofname = str(tmp_path / f'im{ndim}-{np.dtype(dtype).name}.png')
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


@pytest.mark.parametrize('dtype,top', [
                            (np.uint16, 65535),
                            (np.uint32, 2**32 - 1),
                            (np.int16, 2**15 - 1),
                            ])
def test_as_uint8_integer(dtype, top):
    data = np.array([[0, top // 255, top // 2, top]], dtype=dtype)
    converted = _as_uint8(data)
    assert converted.dtype == np.uint8
    assert converted[0,0] == 0
    assert converted[0,1] == 1
    assert converted[0,-1] == 255
    # top // 2 is just under the midpoint, so it rounds down
    assert converted[0,2] == 127


def test_as_uint8_passes_uint8_through():
    data = np.array([[0, 17, 255]], dtype=np.uint8)
    assert _as_uint8(data) is data


def test_as_uint8_clips_negatives():
    data = np.array([[-32768, 0, 32767]], dtype=np.int16)
    assert list(_as_uint8(data)[0]) == [0, 0, 255]


def test_as_uint8_float():
    data = np.array([[-1., 0., 0.5, 1., 2.]])
    converted = _as_uint8(data)
    assert converted.dtype == np.uint8
    assert list(converted[0]) == [0, 0, 128, 255, 255]


def test_as_uint8_bool():
    assert list(_as_uint8(np.array([[False, True]]))[0]) == [0, 255]


def test_load_image_16_bit(tmp_path):
    ifname = image_file(tmp_path, dtype=np.uint16)
    data = load_image(ifname)
    assert data.dtype == np.uint8
    assert data.shape == (24, 32, 3)
    # The values must be the 8-bit originals back again, not truncated ones
    assert np.array_equal(data, load_image(image_file(tmp_path)))


def test_load_image_without_imread(tmp_path, monkeypatch):
    # imread is an optional dependency, so the failure must point at the extra
    import builtins
    real_import = builtins.__import__
    def no_imread(name, *args, **kwargs):
        if name == 'imread':
            raise ImportError('No module named imread')
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_imread)
    with pytest.raises(ImportError, match=r'syxel\[imcat\]'):
        load_image(str(tmp_path / 'nonexistent.png'))
