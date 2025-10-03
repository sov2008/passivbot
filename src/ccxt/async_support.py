"""Minimal async ccxt compatibility layer for the kata tests."""

class Exchange:
    def __init__(self, *args, **kwargs):
        self.options = {}

    async def load_markets(self, reload=False):  # pragma: no cover - placeholder
        return {}

    async def close(self):  # pragma: no cover - placeholder
        return None


class NetworkError(Exception):
    pass


class RateLimitExceeded(Exception):
    pass


exchanges = []

