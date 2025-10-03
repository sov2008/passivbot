import asyncio
import inspect


def pytest_addoption(parser):
    """Provide a stub asyncio_mode ini option so pytest doesn't warn about it."""

    parser.addini("asyncio_mode", "Dummy asyncio mode handler", default="auto")


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark async tests")


def pytest_pyfunc_call(pyfuncitem):
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        funcargs = pyfuncitem.funcargs
        argnames = pyfuncitem._fixtureinfo.argnames
        kwargs = {name: funcargs[name] for name in argnames}
        asyncio.run(pyfuncitem.obj(**kwargs))
        return True
    return None
