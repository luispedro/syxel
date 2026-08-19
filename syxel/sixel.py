# The number of colour registers assumed when the terminal is not asked, or
# does not answer. 256 is what current SIXEL terminals provide; the palette
# path has always used 255 of them.
DEFAULT_COLOURS = 255


def _cube_size(n):
    '''The number of colours in the n x (2n-1) x n cube'''
    return n * n * (2 * n - 1)


def _levels(n):
    '''`n` levels spanning 0-255, ending at 255'''
    step = -(-256 // (n - 1))
    return [min(i * step, 255) for i in range(n)]


def _fixed_cube(max_colours=DEFAULT_COLOURS):
    '''The fallback palette: the largest RGB cube that fits in `max_colours`

    Green gets about twice as many levels as red and blue, which is where the
    eye resolves most detail, so the cube is n x (2n-1) x n: 5 x 9 x 5 = 225
    colours for the 255 registers assumed by default, and up to 8 x 15 x 8 =
    960 for the 1024 that xterm can be configured with.
    '''
    import numpy as np
    n = 2
    while _cube_size(n + 1) <= max_colours:
        n += 1
    if _cube_size(n) > max_colours:
        # Fewer registers than even the smallest cube (2 x 3 x 2 = 12): a grey
        # ramp is the best that a fixed palette can do
        greys = np.linspace(0, 255, max_colours).round().astype(np.int32)
        return np.repeat(greys[:,None], 3, axis=1)
    active = []
    for r in _levels(n):
        for g in _levels(2 * n - 1):
            for b in _levels(n):
                active.append([r,g,b])
    return np.array(active, dtype=np.int32)


def _nearest(colours, active, max_elements=4*1024*1024):
    '''Map each colour to the index of the nearest entry in active

    Distances are squared Euclidean and ties are broken towards the lowest
    index (as `argmin` does). The `|c|^2` term is dropped as it is constant
    within a row and so does not affect which entry is closest, which leaves a
    matrix product. Everything stays integral and well under 2**53, so float64
    reproduces the exact integer distances.

    The matrix is computed in chunks so that its size stays bounded no matter
    how many distinct colours the image has.

    Parameters
    ----------
    colours : ndarray
        Shape (C,3)
    active : ndarray
        Shape (P,3)
    max_elements : int, optional
        Approximate upper bound on the number of elements in the temporary
        distance matrix.

    Returns
    -------
    lut : ndarray
        Shape (C,) and dtype uint8, or uint16 if `active` has more than 256
        entries
    '''
    import numpy as np
    active = active.astype(np.float64)
    sq_norm = (active**2).sum(1)
    chunk = max(1, max_elements // max(1, len(active)))
    dtype = np.uint8 if len(active) <= 256 else np.uint16
    lut = np.zeros(len(colours), dtype=dtype)
    for start in range(0, len(colours), chunk):
        block = colours[start:start + chunk].astype(np.float64)
        dist = sq_norm[None,:] - 2 * (block @ active.T)
        lut[start:start + chunk] = dist.argmin(1)
    return lut


# Some terminals only accept repeat counts up to 255, so long runs are split
_MAX_REPEAT = 255


def _rle(band):
    '''Run-length encode one colour pass over a band

    Trailing empty sixels are dropped (they set no pixel, and the cursor is
    reset by the `$`/`-` that follows anyway) and each remaining run of equal
    bytes is written as `!<n><byte>` whenever that is shorter than repeating
    the byte.

    Parameters
    ----------
    band : ndarray
        Shape (N,) and dtype uint8, one sixel byte per column.

    Returns
    -------
    encoded : bytes
    '''
    import numpy as np
    nonempty = np.flatnonzero(band != 63)
    if not len(nonempty):
        return b''
    band = band[:nonempty[-1] + 1]

    starts = np.flatnonzero(np.concatenate([[True], band[1:] != band[:-1]]))
    lengths = np.diff(np.concatenate([starts, [len(band)]]))

    parts = []
    for value, n in zip(band[starts].tolist(), lengths.tolist()):
        ch = bytes([value])
        while n > 0:
            k = min(n, _MAX_REPEAT)
            n -= k
            # `!<k><byte>` costs 2 bytes plus the digits of k, so it only
            # pays off for runs longer than that
            if k > len(str(k)) + 2:
                parts.append(b'!%d%s' % (k, ch))
            else:
                parts.append(ch * k)
    return b''.join(parts)


def rgb_to_palette(rgb, max_colours=None):
    '''Convert an RGB image to a palette image.

    Parameters
    ----------
    rgb : ndarray
        An image with shape (M,N,3) and dtype uint8.
    max_colours : int, optional
        How many colour registers the terminal has to spare (default:
        `DEFAULT_COLOURS`). `syxel.terminal.colour_registers()` asks the
        terminal for it.

    Returns
    -------
    active : ndarray
        The active palette colours, with shape (P <= max_colours,3) and dtype
        int32.
    res : ndarray
        The palette image, with shape (M,N) and dtype uint8, or uint16 if the
        palette has more than 256 entries.
    '''
    import numpy as np
    if max_colours is None:
        max_colours = DEFAULT_COLOURS
    flat = rgb.reshape((-1,3)).astype(np.int32)
    keys = (flat[:,0] << 16) | (flat[:,1] << 8) | flat[:,2]
    ukeys, first, inverse, counts = np.unique(keys,
                                              return_index=True,
                                              return_inverse=True,
                                              return_counts=True)
    inverse = inverse.reshape(len(keys))
    colours = np.stack([ukeys >> 16, (ukeys >> 8) & 0xff, ukeys & 0xff], axis=1)

    # Most frequent first, ties broken by order of first appearance (lexsort
    # sorts by the last key first)
    order = np.lexsort((first, -counts))
    active = colours[order[:max_colours]]
    n_pixels = rgb.shape[0] * rgb.shape[1]
    if counts[order[:max_colours]].sum() < 0.5 * n_pixels:
        active = _fixed_cube(max_colours)

    lut = _nearest(colours, active)
    res = lut[inverse].reshape(rgb.shape[:2])
    return active, res


def write_sixel(out, data, active):
    '''Write a palette image to `out` as a SIXEL escape sequence.

    Parameters
    ----------
    out : file-like
        Opened for writing in binary mode (anything with a bytes `write`).
    data : ndarray
        The palette image, with shape (M,N) and an unsigned integer dtype.
    active : ndarray
        The active palette colours, with shape (P,3) and values in 0-255.
    '''
    import numpy as np
    active = active.astype(np.int32) * 100 // 255
    height, width = data.shape
    sixel_header = b'\x1bP0;0;0q"1;1;'
    out.write(sixel_header)

    # Raster attributes are `Ph;Pv`, i.e. width before height
    out.write(f'{width};{height}'.encode('ascii'))
    for i in range(len(active)):
        # 2 is for RGB
        out.write(f'#{i};2;{active[i,0]:};{active[i,1]:};{active[i,2]:}'.encode('ascii'))

    for i in range(-(-height//6)):
        sel = data[i*6:(i+1)*6]
        # The last band can be short, in which case only the bits for the rows
        # that exist are set (the raster attributes above already declare the
        # true height, so the terminal does not draw the missing rows)
        weights = np.array([1,2,4,8,16,32])[:len(sel)]
        is_first = True
        for c in set(sel.ravel()):
            to_write = (sel == c).astype(np.int32)
            to_write = np.dot(to_write.T, weights) + 63
            if not is_first:
                out.write(b'$')
            is_first = False
            out.write(f'#{c}'.encode('ascii'))
            out.write(_rle(to_write.astype(np.uint8)))
        out.write(b'-')
    out.write(b'\x1b\\')  # End of Sixel
