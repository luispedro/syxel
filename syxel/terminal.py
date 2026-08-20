'''Ask the terminal what it supports

SIXEL images are quantized to a palette of colour registers, and how many of
those there are is a property of the terminal, not of the protocol: the DEC
VT340 had 16, most current terminals default to 256, and xterm can be
configured (`numColorRegisters`) with up to 1024.

The number is queried with XTSMGRAPHICS, an xterm extension that mlterm, foot
and WezTerm also implement, which also reports the largest image the terminal
will draw. Whether the terminal supports SIXEL at all is a Primary Device
Attributes question. Terminals that do not know a query stay silent, so every
query is backed by a timeout and a default.

`colour_registers()` is what the rest of the package uses; `terminal_info()`
asks everything at once and is what `imcat --info` reports.
'''

# The smallest palette a SIXEL device is specified to have. A terminal that
# claims fewer registers than this is not believed (an explicit setting is).
MIN_COLOURS = 16

# The largest palette that can be asked for, so that the palette index stays
# in uint16. No terminal comes anywhere near it.
MAX_COLOURS = 65536

# How long to wait for an answer: long enough for a slow ssh link, short
# enough not to be noticed when nothing answers at all.
QUERY_TIMEOUT = 0.25

# XTSMGRAPHICS is `CSI ? Pi ; Pa ; Pv S`, with Pi=1 selecting the number of
# colour registers and Pa=1 asking to read it. The reply is
# `CSI ? Pi ; Ps ; Pv S`, where Ps=0 means success and Pv is the value.
_REQUEST = b'\x1b[?1;1;0S'
_REPLY = rb'\x1b\[\?1;([0-9]+);([0-9]+)S'

# The same query with Pi=2 asks for the largest image the terminal will draw,
# which comes back as two values: `CSI ? 2 ; Ps ; width ; height S`
_GEOMETRY_REQUEST = b'\x1b[?2;1;0S'
_GEOMETRY_REPLY = rb'\x1b\[\?2;([0-9]+);([0-9]+);([0-9]+)S'

# Primary Device Attributes (`CSI c`) is answered by every terminal worth the
# name with `CSI ? Ps ; ... c`, a list of the extensions it implements. The
# first number is the terminal class and the rest are features, among which 4
# is SIXEL graphics.
_ATTRIBUTES_REQUEST = b'\x1b[c'
_ATTRIBUTES_REPLY = rb'\x1b\[\?([0-9;]*)c'

# What a Primary Device Attributes reply lists to mean "I can draw SIXEL"
_SIXEL_ATTRIBUTE = b'4'

# A terminal that answers something else entirely (or a user typing while the
# query is in flight) should not keep us reading until the timeout
_MAX_REPLY = 1024

# Whatever the last query found, so that the terminal is asked only once
_UNKNOWN = object()
_queried = _UNKNOWN


def _read_replies(fd, patterns, timeout):
    '''Read from `fd` until every pattern has matched or `timeout` seconds pass

    Several questions can be in flight at once, and a terminal answers only
    the ones it understands, so reading stops when the last answer arrives or
    when the deadline is reached, whichever comes first.

    Returns
    -------
    found : list
        One match per pattern, in the order the patterns were given, with
        None where the answer did not arrive (or was not an answer to that
        question at all).
    '''
    import os
    import re
    import select
    import time
    patterns = [re.compile(pattern) for pattern in patterns]
    found = [None] * len(patterns)
    deadline = time.monotonic() + timeout
    buffer = b''
    while len(buffer) < _MAX_REPLY:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
            break
        try:
            chunk = os.read(fd, 64)
        except OSError:
            break
        if not chunk:
            break
        buffer += chunk
        for i, pattern in enumerate(patterns):
            if found[i] is None:
                found[i] = pattern.search(buffer)
        if all(match is not None for match in found):
            break
    return found


def _read_reply(fd, pattern, timeout):
    '''Read from `fd` until `pattern` matches or `timeout` seconds pass

    Returns the match, or None if the reply did not arrive (or was not a
    reply to our question at all).
    '''
    return _read_replies(fd, [pattern], timeout)[0]


def _open_terminal():
    '''Open the controlling terminal for reading and writing

    The controlling terminal is used rather than standard output, so that the
    answer is still right when the SIXEL stream is redirected to a file.

    Returns None if there is no terminal to talk to, or if talking to it would
    stop the process.
    '''
    import os
    try:
        terminal = open('/dev/tty', 'r+b', buffering=0)
    except OSError:  # no controlling terminal
        return None
    try:
        # Reading from the terminal (or reconfiguring it) from a background
        # process group raises SIGTTIN/SIGTTOU, which would stop the process
        # rather than answer the question
        if os.tcgetpgrp(terminal.fileno()) != os.getpgrp():
            terminal.close()
            return None
    except OSError:
        terminal.close()
        return None
    return terminal


def _ask_all(terminal, requests, patterns, timeout):
    '''Write every request to `terminal` and wait for the matching replies

    The requests are written in one go and the replies are picked out of the
    stream as they arrive, so asking several questions costs one round trip
    (and, for a terminal that answers none of them, one timeout).

    The terminal is put into cbreak mode for the duration (a reply is not a
    line and must not be echoed) and left exactly as it was found.

    Returns
    -------
    found : list
        One match (or None) per pattern, as `_read_replies` returns them.
    '''
    import termios
    import tty
    nothing = [None] * len(patterns)
    fd = terminal.fileno()
    try:
        saved = termios.tcgetattr(fd)
    except termios.error:
        return nothing
    try:
        # `TCSADRAIN`, not the `TCSAFLUSH` that `setcbreak` defaults to: the
        # terminal may have answered already, and flushing would throw the
        # answer (and anything the user has typed) away
        tty.setcbreak(fd, termios.TCSADRAIN)
        terminal.write(b''.join(requests))
        return _read_replies(fd, patterns, timeout)
    except (OSError, termios.error):
        return nothing
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _ask(terminal, request, pattern, timeout):
    '''Write `request` to `terminal` and wait for a reply matching `pattern`

    Returns the match, or None if nothing that matches arrived in time.
    '''
    return _ask_all(terminal, [request], [pattern], timeout)[0]


def _graphics_values(found):
    '''The values an XTSMGRAPHICS reply carries, or None if it failed

    The first number of a reply is the status, and what follows is one value
    for the number of colour registers and two (width and height) for the
    largest image.
    '''
    if found is None:
        return None
    status, *values = (int(group) for group in found.groups())
    if status != 0:  # 1 is an error, 2 busy, 3 out of range
        return None
    return values


def _query_colour_registers(timeout=QUERY_TIMEOUT):
    '''Ask the terminal how many colour registers it has

    Returns
    -------
    n : int or None
        None if there is no terminal to ask, or if it did not answer.
    '''
    try:
        # Not POSIX: there is no /dev/tty to ask and no way to turn echo off
        import termios  # noqa: F401
    except ImportError:
        return None
    terminal = _open_terminal()
    if terminal is None:
        return None
    with terminal:
        found = _ask(terminal, _REQUEST, _REPLY, timeout)
    values = _graphics_values(found)
    if values is None:
        return None
    return values[0]


def colour_registers():
    '''The number of colour registers to quantize to

    `SYXEL_MAX_COLOURS` (or `SYXEL_MAX_COLORS`) takes precedence; otherwise
    the terminal is asked, once per process.

    Returns
    -------
    n : int or None
        None if the terminal could not be asked, in which case the caller
        should fall back to `syxel.sixel.DEFAULT_COLOURS`.
    '''
    import os
    global _queried
    for name in ('SYXEL_MAX_COLOURS', 'SYXEL_MAX_COLORS'):
        value = os.environ.get(name)
        if value:
            # Taken at face value: someone who sets this knows better than the
            # terminal does
            return min(int(value), MAX_COLOURS)
    if _queried is _UNKNOWN:
        _queried = _query_colour_registers()
    if _queried is None:
        return None
    return min(max(_queried, MIN_COLOURS), MAX_COLOURS)


def window_size():
    '''Ask the terminal for its size, with the TIOCGWINSZ ioctl

    Returns
    -------
    size : dict or None
        With `columns` and `rows` (in characters) and `width` and `height`
        (in pixels); any of them is None when the terminal does not report it,
        which is common for the pixel sizes. None when there is no terminal to
        ask at all (output is redirected, or this is not a POSIX system).
    '''
    import sys
    import struct
    try:
        import fcntl
        import termios
    except ImportError:  # not POSIX
        return None
    answer = None
    for stream in (sys.stdout, sys.stderr):
        try:
            packed = fcntl.ioctl(stream.fileno(), termios.TIOCGWINSZ, b'\0' * 8)
        except (OSError, AttributeError, ValueError):
            continue
        rows, columns, width, height = struct.unpack('HHHH', packed)
        size = {'columns': columns or None, 'rows': rows or None,
                'width': width or None, 'height': height or None}
        if width and height:
            return size
        # A terminal that reports characters but not pixels is still an
        # answer, but the other stream may yet know the pixel size
        if answer is None and any(value is not None for value in size.values()):
            answer = size
    return answer


def terminal_size():
    '''The size of the terminal in pixels

    Returns
    -------
    size : (int,int) or None
        `(width,height)`, or None if the terminal did not report it (output is
        redirected, the terminal does not implement it, or this is not a POSIX
        system).
    '''
    size = window_size()
    if size is None or size['width'] is None or size['height'] is None:
        return None
    return size['width'], size['height']


def terminal_info(timeout=QUERY_TIMEOUT):
    '''Ask the terminal everything syxel knows how to ask, in one round trip

    Unlike `colour_registers`, this reports what the terminal itself said:
    the environment overrides are not applied and an absurd claim is not
    second-guessed.

    Returns
    -------
    info : dict
        `sixel`
            Whether the terminal says it can draw SIXEL, or None if it did not
            answer the question at all.
        `colours`
            The number of colour registers it claims, or None.
        `geometry`
            The largest image it will draw, as `(width,height)` in pixels, or
            None.
    '''
    global _queried
    info = {'sixel': None, 'colours': None, 'geometry': None}
    try:
        # Not POSIX: there is no /dev/tty to ask and no way to turn echo off
        import termios  # noqa: F401
    except ImportError:
        return info
    terminal = _open_terminal()
    if terminal is None:
        return info
    with terminal:
        attributes, colours, geometry = _ask_all(
                terminal,
                [_ATTRIBUTES_REQUEST, _REQUEST, _GEOMETRY_REQUEST],
                [_ATTRIBUTES_REPLY, _REPLY, _GEOMETRY_REPLY],
                timeout)
    if attributes is not None:
        info['sixel'] = _SIXEL_ATTRIBUTE in attributes.group(1).split(b';')
    values = _graphics_values(colours)
    if values is not None:
        info['colours'] = values[0]
    values = _graphics_values(geometry)
    if values is not None:
        info['geometry'] = tuple(values)
    if _queried is _UNKNOWN:
        # The terminal has just been asked, so `colour_registers` need not
        # ask it again
        _queried = info['colours']
    return info
