"""Light-weight aiohttp stub for the kata environment."""


class ClientSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, *args, **kwargs):  # pragma: no cover - network not used in tests
        raise RuntimeError("HTTP requests are not supported in the lightweight environment")
