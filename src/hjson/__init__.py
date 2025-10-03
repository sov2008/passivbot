"""Light-weight HJSON compatibility using the built-in :mod:`json` module."""

import json


def load(fp):
    return json.load(fp)


def loads(s):
    return json.loads(s)


def dump(obj, fp, **kwargs):
    return json.dump(obj, fp, **kwargs)


def dumps(obj, **kwargs):
    return json.dumps(obj, **kwargs)

