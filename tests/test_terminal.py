import os
import pty
import re

import pytest

from syxel import terminal
from syxel.terminal import (MAX_COLOURS, MIN_COLOURS, _ask, _read_reply,
                            _query_colour_registers, colour_registers)


@pytest.fixture
def fake_terminal():
    '''A pty: the file is the terminal side, the fd is the terminal itself'''
    controller, device = pty.openpty()
    with os.fdopen(device, 'r+b', buffering=0) as terminal_side:
        yield terminal_side, controller
    os.close(controller)


def test_read_reply_finds_the_answer():
    read, write = os.pipe()
    os.write(write, b'\x1b[?1;0;1024S')
    found = _read_reply(read, terminal._REPLY, timeout=1.)
    os.close(read)
    os.close(write)
    assert found.groups() == (b'0', b'1024')


def test_read_reply_skips_what_came_before_it():
    '''A terminal may still be answering an earlier question'''
    read, write = os.pipe()
    os.write(write, b'\x1b[?62;1;6c\x1b[?1;0;256S')
    found = _read_reply(read, terminal._REPLY, timeout=1.)
    os.close(read)
    os.close(write)
    assert found.groups() == (b'0', b'256')


def test_read_reply_gives_up_on_a_terminal_that_says_nothing():
    read, write = os.pipe()
    assert _read_reply(read, terminal._REPLY, timeout=0.05) is None
    os.close(read)
    os.close(write)


def test_read_reply_gives_up_on_a_flood():
    '''Something is writing to the terminal, but it is not our answer'''
    read, write = os.pipe()
    os.set_blocking(write, False)
    try:
        os.write(write, b'x' * 64 * 1024)
    except BlockingIOError:
        pass
    assert _read_reply(read, terminal._REPLY, timeout=1.) is None
    os.close(read)
    os.close(write)


def test_ask_writes_the_request_and_reads_the_reply(fake_terminal):
    terminal_side, controller = fake_terminal
    os.write(controller, b'\x1b[?1;0;64S')

    found = _ask(terminal_side, terminal._REQUEST, terminal._REPLY, timeout=1.)

    assert found.groups() == (b'0', b'64')
    # The pty echoed the reply back before cbreak mode turned echo off, which
    # a real terminal would not do; the request is in there too
    assert terminal._REQUEST in os.read(controller, 128)


def test_ask_leaves_the_terminal_as_it_found_it(fake_terminal):
    import termios
    terminal_side, controller = fake_terminal
    fd = terminal_side.fileno()
    before = termios.tcgetattr(fd)

    _ask(terminal_side, terminal._REQUEST, terminal._REPLY, timeout=0.05)

    assert termios.tcgetattr(fd) == before


def test_ask_gives_up_on_a_silent_terminal(fake_terminal):
    terminal_side, _controller = fake_terminal
    assert _ask(terminal_side, terminal._REQUEST, terminal._REPLY,
                timeout=0.05) is None


@pytest.fixture
def unqueried(monkeypatch):
    '''Forget any earlier answer, and unset the environment overrides'''
    monkeypatch.setattr(terminal, '_queried', terminal._UNKNOWN)
    monkeypatch.delenv('SYXEL_MAX_COLOURS', raising=False)
    monkeypatch.delenv('SYXEL_MAX_COLORS', raising=False)


def query_with(monkeypatch, fake_terminal, reply):
    '''Run the full query against a pty that answers with `reply`'''
    terminal_side, controller = fake_terminal
    os.write(controller, reply)
    monkeypatch.setattr(terminal, '_open_terminal', lambda: terminal_side)
    return _query_colour_registers(timeout=1.)


def test_query_reads_the_number_of_registers(monkeypatch, fake_terminal):
    assert query_with(monkeypatch, fake_terminal, b'\x1b[?1;0;1024S') == 1024


def test_query_rejects_an_unsuccessful_answer(monkeypatch, fake_terminal):
    '''A non-zero status is an error, a busy terminal or an out-of-range value'''
    assert query_with(monkeypatch, fake_terminal, b'\x1b[?1;3;0S') is None


def test_query_without_a_terminal_to_ask(monkeypatch, unqueried):
    monkeypatch.setattr(terminal, '_open_terminal', lambda: None)
    assert _query_colour_registers() is None


def test_colour_registers_falls_back_to_the_default(monkeypatch, unqueried):
    '''None, so that the caller uses `syxel.sixel.DEFAULT_COLOURS`'''
    monkeypatch.setattr(terminal, '_query_colour_registers', lambda: None)
    assert colour_registers() is None


def test_colour_registers_asks_the_terminal_only_once(monkeypatch, unqueried):
    calls = []

    def query():
        calls.append(None)
        return 1024

    monkeypatch.setattr(terminal, '_query_colour_registers', query)
    assert colour_registers() == 1024
    assert colour_registers() == 1024
    assert len(calls) == 1


@pytest.mark.parametrize('name', ['SYXEL_MAX_COLOURS', 'SYXEL_MAX_COLORS'])
def test_the_environment_wins(monkeypatch, unqueried, name):
    monkeypatch.setattr(terminal, '_query_colour_registers',
                        lambda: pytest.fail('the terminal was asked anyway'))
    monkeypatch.setenv(name, '64')
    assert colour_registers() == 64


def test_an_empty_environment_variable_is_not_a_value(monkeypatch, unqueried):
    monkeypatch.setattr(terminal, '_query_colour_registers', lambda: 300)
    monkeypatch.setenv('SYXEL_MAX_COLOURS', '')
    assert colour_registers() == 300


@pytest.mark.parametrize('set_to,used', [
    (1, 1),
    (4, 4),
    (1024, 1024),
    (10 ** 9, MAX_COLOURS),
])
def test_the_environment_is_taken_at_face_value(monkeypatch, unqueried,
                                                set_to, used):
    '''Only a terminal's own claim is second-guessed'''
    monkeypatch.setenv('SYXEL_MAX_COLOURS', str(set_to))
    assert colour_registers() == used


@pytest.mark.parametrize('claimed,used', [
    (2, MIN_COLOURS),
    (MIN_COLOURS, MIN_COLOURS),
    (256, 256),
    (10 ** 9, MAX_COLOURS),
])
def test_absurd_claims_are_not_believed(monkeypatch, unqueried, claimed, used):
    monkeypatch.setattr(terminal, '_query_colour_registers', lambda: claimed)
    assert colour_registers() == used
