def main():
    import sys

    import numpy as np
    import imread as im
    from collections import Counter
    ifname = sys.argv[1]

    data = im.imread(ifname)
    while data.shape[0] > 800 or data.shape[1] > 1200:
        data = data[::2,::2]

    cs = Counter([tuple(pix) for pix in data.reshape((-1,3))])

    colours = list(cs.keys())
    colours.sort(key=lambda x: -cs[x])
    active = np.array(colours[:255], dtype=np.int32)
    if sum(cs[tuple(c)] for c in active) < 0.5 * data.size:
        active = []
        for r in range(0,257,64):
            if r == 256:
                r = 255
            for g in range(0,257,32):
                if g == 256:
                    g = 255
                for b in range(0,257,64):
                    if b == 256:
                        b = 255
                    active.append([r,g,b])
        active = np.array(active, dtype=np.int32)


    palette = {}
    for c in colours:
        palette[c] = ((active - c)**2).sum(1).argmin()

    res = np.zeros(data.shape[:2], dtype=np.uint8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            res[i,j] = palette[tuple(data[i,j])]
    active = active.astype(np.int32) * 100 // 255

    out = sys.stdout.buffer

    w,h = res.shape
    sixel_header = b'\x1bP0;0;0q"1;1;'
    out.write(sixel_header)

    out.write(f'{h};{w}'.encode('ascii'))
    for i in range(len(active)):
        # 2 is for RGB
        out.write(f'#{i};2;{active[i,0]:};{active[i,1]:};{active[i,2]:}'.encode('ascii'))

    for i in range(res.shape[0]//6):
        sel = res[i*6:(i+1)*6]
        is_first = True
        for c in set(sel.ravel()):
            to_write = (sel == c).astype(np.int32)
            to_write = np.dot(to_write.T, np.array([1,2,4,8,16,32])) + 63
            if not is_first:
                out.write(b'$')
            is_first = False
            out.write(f'#{c}'.encode('ascii'))
            to_write = to_write.astype(np.uint8)
            to_write = to_write.tobytes()
            out.write(to_write)
        out.write(b'-')
    out.write(b'\x1b\\') # End of Sixel

