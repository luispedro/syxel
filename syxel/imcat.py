def _as_uint8(data):
    '''Rescale an image array to uint8.

    Integer types are rescaled from the full range of their dtype (so a uint16
    65535 becomes 255) and floating point ones are assumed to lie in [0, 1] and
    are clipped to it. Negative values are clipped away.
    '''
    import numpy as np
    if data.dtype == np.uint8:
        return data
    if data.dtype == bool:
        return data.astype(np.uint8) * 255
    if np.issubdtype(data.dtype, np.integer):
        top = np.iinfo(data.dtype).max
    else:
        top = 1.
    return (data.clip(0, top) * (255. / top)).round().astype(np.uint8)


def load_image(ifname : str, max_height : int = 800, max_width : int = 1200):
    '''Load an image from a file.

    The image is halved by subsampling until it fits within
    `max_height` x `max_width`.

    The result is always RGB uint8: greyscale images are expanded to three
    channels, an alpha channel, if present, is dropped, and other dtypes are
    rescaled to 0-255 (see `_as_uint8`).
    '''

    import numpy as np
    try:
        import imread as im
    except ImportError:
        raise ImportError(
            'Loading image files requires the imread package, which is an '
            'optional dependency of syxel: install it with '
            '`pip install "syxel[imcat]"` (or `pip install imread`).')
    data = im.imread(ifname)
    while data.shape[0] > max_height or data.shape[1] > max_width:
        data = data[::2,::2]
    data = _as_uint8(data)
    if data.ndim == 2:
        data = data[:,:,None]
    if data.shape[2] in (2, 4):
        data = data[:,:,:-1]
    if data.shape[2] == 1:
        data = np.repeat(data, 3, axis=2)
    return data


def _unknown(what):
    '''How an answer that did not come back is spelled in the report'''
    return f'unknown ({what})'


def format_info(max_colours=None, max_height=800, max_width=1200):
    '''Describe the terminal, as `imcat --info` reports it

    The terminal is asked what it supports (which takes a moment, and may
    time out); the arguments are the limits `imcat` was asked to use, so that
    the report ends with what it would actually do.

    Returns
    -------
    lines : list of str
        Without trailing newlines.
    '''
    import os
    from syxel.sixel import DEFAULT_COLOURS
    from syxel.syxel_version import __version__
    from syxel.terminal import colour_registers, terminal_info, window_size

    info = terminal_info()
    size = window_size()

    supported = {True: 'yes',
                 False: 'no (it answered, but did not list SIXEL)',
                 None: _unknown('the terminal did not answer')}[info['sixel']]

    if info['colours'] is None:
        registers = _unknown('the terminal did not answer')
    else:
        registers = str(info['colours'])

    if size is None:
        geometry = _unknown('not a terminal, or output is redirected')
    else:
        parts = []
        if size['columns'] and size['rows']:
            parts.append(f"{size['columns']}x{size['rows']} characters")
        if size['width'] and size['height']:
            parts.append(f"{size['width']}x{size['height']} pixels")
        else:
            parts.append('pixel size not reported')
        geometry = ', '.join(parts)

    if info['geometry'] is None:
        largest = _unknown('the terminal did not answer')
    else:
        largest = '{}x{} pixels'.format(*info['geometry'])

    if max_colours is not None:
        source = '--max-colours'
    elif os.environ.get('SYXEL_MAX_COLOURS') or os.environ.get('SYXEL_MAX_COLORS'):
        source = 'the environment'
        max_colours = colour_registers()
    else:
        max_colours = colour_registers()
        source = 'the terminal'
        if max_colours is None:
            max_colours = DEFAULT_COLOURS
            source = 'the default, as the terminal did not say'
        elif max_colours != info['colours']:
            source = 'the terminal, clamped to what SIXEL allows'

    rows = [('syxel version', __version__),
            ('Terminal', os.environ.get('TERM') or _unknown('$TERM is not set')),
            ('SIXEL support', supported),
            ('Colour registers', registers),
            ('Terminal size', geometry),
            ('Largest image', largest),
            ('Colours in use', f'{max_colours} (from {source})'),
            ('Images fit into', f'{max_width}x{max_height} pixels '
                                '(--max-width/--max-height)'),
            ]
    width = max(len(label) for label, _ in rows) + 2
    return [f'{label + ":":<{width}}{value}' for label, value in rows]


def parse_args(argv=None):
    '''Parse the command line arguments (`sys.argv[1:]` if argv is None)'''
    import argparse
    from syxel.syxel_version import __version__
    from syxel.terminal import MAX_COLOURS

    parser = argparse.ArgumentParser(
            prog='imcat',
            description='Write images to the terminal using the SIXEL protocol')
    parser.add_argument('images', metavar='IMAGE', nargs='*',
                        help='image file to display (may be repeated)')
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {__version__}')
    parser.add_argument('--info', action='store_true',
                        help='ask the terminal what it supports (SIXEL, how '
                             'many colour registers, how big it is) and exit')
    parser.add_argument('--max-height', type=int, default=800, metavar='N',
                        help='subsample until the image is at most N pixels high (default: %(default)s)')
    parser.add_argument('--max-width', type=int, default=1200, metavar='N',
                        help='subsample until the image is at most N pixels wide (default: %(default)s)')
    parser.add_argument('--max-colours', '--max-colors', type=int, default=None,
                        metavar='N', dest='max_colours',
                        help='quantize to at most N colours (default: ask the '
                             'terminal how many colour registers it has, and '
                             'assume 255 if it does not say)')
    args = parser.parse_args(argv)
    if args.info and args.images:
        parser.error('--info takes no image files')
    if not args.info and not args.images:
        parser.error('the following arguments are required: IMAGE')
    if args.max_height < 1 or args.max_width < 1:
        parser.error('--max-height and --max-width must be positive')
    if args.max_colours is not None and not 1 <= args.max_colours <= MAX_COLOURS:
        parser.error(f'--max-colours must be between 1 and {MAX_COLOURS}')
    return args


def main(argv=None):
    import sys
    from syxel.sixel import rgb_to_palette, write_sixel
    from syxel.terminal import colour_registers
    args = parse_args(argv)
    if args.info:
        for line in format_info(max_colours=args.max_colours,
                                max_height=args.max_height,
                                max_width=args.max_width):
            print(line)
        return
    max_colours = args.max_colours
    if max_colours is None:
        max_colours = colour_registers()
    out = sys.stdout.buffer
    status = 0
    for ifname in args.images:
        try:
            rgb = load_image(ifname,
                             max_height=args.max_height,
                             max_width=args.max_width)
        except ImportError as e:
            # A missing optional dependency will not be there for the next
            # file either, so there is no point in carrying on
            sys.exit(f'imcat: {e}')
        except (OSError, RuntimeError) as e:
            # Files that cannot be read are reported and skipped (as `cat`
            # does), but they still make the exit status non-zero. `imread`
            # signals a missing or unreadable file with OSError and a file it
            # cannot decode (not an image, unknown extension) with RuntimeError.
            # Some of those messages already name the file, some do not
            message = str(e) if ifname in str(e) else f'{ifname}: {e}'
            print(f'imcat: {message}', file=sys.stderr)
            status = 1
            continue
        active, data = rgb_to_palette(rgb, max_colours=max_colours)
        write_sixel(out, data, active)
        # Otherwise the shell prompt (or the next image) lands on top of this one
        out.write(b'\n')
    out.flush()
    if status:
        sys.exit(status)
