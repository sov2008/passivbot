def tqdm(iterable=None, **kwargs):
    if iterable is None:
        return _Dummy()
    return iterable


class _Dummy:
    def update(self, *args, **kwargs):
        return None

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
