def load_image(ifname : str, max_height : int = 800, max_width : int = 1200):
    '''Load an image from a file.

    The image is halved by subsampling until it fits within
    `max_height` x `max_width`.

    The result is always RGB: greyscale images are expanded to three channels
    and an alpha channel, if present, is dropped.
    '''

    import numpy as np
    import imread as im
    data = im.imread(ifname)
    while data.shape[0] > max_height or data.shape[1] > max_width:
        data = data[::2,::2]
    if data.ndim == 2:
        data = data[:,:,None]
    if data.shape[2] in (2, 4):
        data = data[:,:,:-1]
    if data.shape[2] == 1:
        data = np.repeat(data, 3, axis=2)
    return data


def parse_args(argv=None):
    '''Parse the command line arguments (`sys.argv[1:]` if argv is None)'''
    import argparse
    from syxel.syxel_version import __version__

    parser = argparse.ArgumentParser(
            prog='imcat',
            description='Write images to the terminal using the SIXEL protocol')
    parser.add_argument('images', metavar='IMAGE', nargs='+',
                        help='image file to display (may be repeated)')
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {__version__}')
    parser.add_argument('--max-height', type=int, default=800, metavar='N',
                        help='subsample until the image is at most N pixels high (default: %(default)s)')
    parser.add_argument('--max-width', type=int, default=1200, metavar='N',
                        help='subsample until the image is at most N pixels wide (default: %(default)s)')
    args = parser.parse_args(argv)
    if args.max_height < 1 or args.max_width < 1:
        parser.error('--max-height and --max-width must be positive')
    return args


def main(argv=None):
    import sys
    from syxel.sixel import rgb_to_palette, write_sixel
    args = parse_args(argv)
    out = sys.stdout.buffer
    for ifname in args.images:
        rgb = load_image(ifname,
                         max_height=args.max_height,
                         max_width=args.max_width)
        active, data = rgb_to_palette(rgb)
        write_sixel(out, data, active)
        # Otherwise the shell prompt (or the next image) lands on top of this one
        out.write(b'\n')
    out.flush()
