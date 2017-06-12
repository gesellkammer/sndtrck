from __future__ import absolute_import
from emlib.music.event import Chord
from typing import Iterable as Iter, Tuple


def _normalizenote(note):
    """
    note can be a tuple (midinote, dbamp) or a midinote

    Returns: (midinote, dbamp)
    """
    if isinstance(note, (tuple, list)) and len(note) == 2:
        return note
    return note, 0


def newchord(data):
    # type: (Iter[Tuple[float, float]]) -> Chord
    """
    Convert a seq. of (midinote, amp) to a chord.

    amp: 0-1

    :param data: a seq. of (midinote, amp) tuples
    :return: Chord
    """
    return Chord(data)

