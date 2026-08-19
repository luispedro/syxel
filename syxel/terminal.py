'''Ask the terminal what it supports

SIXEL images are quantized to a palette of colour registers, and how many of
those there are is a property of the terminal, not of the protocol: the DEC
VT340 had 16, most current terminals default to 256, and xterm can be
configured (`numColorRegisters`) with up to 1024.

The number is queried with XTSMGRAPHICS, an xterm extension that mlterm, foot
and WezTerm also implement. Terminals that do not know it stay silent, so the
query is backed by a timeout and a default.
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

# A terminal that answers something else entirely (or a user typing while the
# query is in flight) should not keep us reading until the timeout
_MAX_REPLY = 1024

# Whatever the last query found, so that the terminal is asked only once
_UNKNOWN = object()
_queried = _UNKNOWN


def _read_reply(fd, pattern, timeout):
    '''Read from `fd` until `pattern` matches or `timeout` seconds pass

    Returns the match, or None if the reply did not arrive (or was not a
    reply to our question at all).
    '''
    import os
    import re
    import select
    import time
    pattern = re.compile(pattern)
    deadline = time.monotonic() + timeout
    buffer = b''
    while len(buffer) < _MAX_REPLY:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
            return None
        try:
            chunk = os.read(fd, 64)
        except OSError:
            return None
        if not chunk:
            return None
        buffer += chunk
        found = pattern.search(buffer)
        if found is not None:
            return found
    return None


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


def _ask(terminal, request, pattern, timeout):
    '''Write `request` to `terminal` and wait for a reply matching `pattern`

    The terminal is put into cbreak mode for the duration (the reply is not a
    line and must not be echoed) and left exactly as it was found.

    Returns the match, or None if nothing that matches arrived in time.
    '''
    import termios
    import tty
    fd = terminal.fileno()
    try:
        saved = termios.tcgetattr(fd)
    except termios.error:
        return None
    try:
        # `TCSADRAIN`, not the `TCSAFLUSH` that `setcbreak` defaults to: the
        # terminal may have answered already, and flushing would throw the
        # answer (and anything the user has typed) away
        tty.setcbreak(fd, termios.TCSADRAIN)
        terminal.write(request)
        return _read_reply(fd, pattern, timeout)
    except (OSError, termios.error):
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


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
    if found is None:
        return None
    status, value = (int(group) for group in found.groups())
    if status != 0:  # 1 is an error, 2 busy, 3 out of range
        return None
    return value


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
