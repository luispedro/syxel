def load_image(ifname : str):
    '''Load an image from a file.

    The result is always RGB: greyscale images are expanded to three channels
    and an alpha channel, if present, is dropped.
    '''

    import numpy as np
    import imread as im
    data = im.imread(ifname)
    while data.shape[0] > 800 or data.shape[1] > 1200:
        data = data[::2,::2]
    if data.ndim == 2:
        data = data[:,:,None]
    if data.shape[2] in (2, 4):
        data = data[:,:,:-1]
    if data.shape[2] == 1:
        data = np.repeat(data, 3, axis=2)
    return data


def main():
    import sys
    from syxel.sixel import rgb_to_palette, write_sixel
    ifname = sys.argv[1]
    active, data = rgb_to_palette(load_image(ifname))
    write_sixel(sys.stdout.buffer, data, active)
