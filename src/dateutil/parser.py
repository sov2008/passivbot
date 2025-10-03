"""Minimal drop-in replacement for :mod:`dateutil.parser` used in tests."""

import datetime as _dt


class ParserError(ValueError):
    pass


def parse(value: str) -> _dt.datetime:
    patterns = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ]
    for fmt in patterns:
        try:
            return _dt.datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ParserError(f"Unsupported date format: {value}")

