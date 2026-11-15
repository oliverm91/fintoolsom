import pytest

from fintoolsom import settings


@pytest.fixture(autouse=True)
def _restore():
    before = {s: settings.get_backend(s) for s in settings.SUBPACKAGES}
    yield
    for s, b in before.items():
        settings.set_backend(b, s)


def test_set_backend_without_subpackages_reaches_the_ported_ones():
    settings.set_backend("python")
    settings.set_backend("cpp")
    for s in settings.SUBPACKAGES:
        expected = "cpp" if s in settings.PORTED else "python"
        assert settings.get_backend(s) == expected


def test_use_backend_restores_the_previous_backend():
    settings.set_backend("python", "rates")
    with settings.use_backend("cpp", "rates"):
        assert settings.get_backend("rates") == "cpp"
    assert settings.get_backend("rates") == "python"


def test_use_backend_restores_after_an_error():
    settings.set_backend("python", "rates")
    with pytest.raises(RuntimeError), settings.use_backend("cpp", "rates"):
        raise RuntimeError
    assert settings.get_backend("rates") == "python"


@pytest.mark.parametrize(
    "call",
    [
        lambda: settings.set_backend("fortran"),
        lambda: settings.set_backend("python", "nope"),
        lambda: settings.get_backend("nope"),
    ],
)
def test_unknown_names_raise(call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize("subpackage", settings.SUBPACKAGES)
def test_cpp_on_a_subpackage_not_ported_raises(subpackage, monkeypatch):
    # Every subpackage is ported today: pretend this one is not.
    monkeypatch.setattr(settings, "PORTED", settings.PORTED - {subpackage})
    with pytest.raises(ValueError, match="has no C"):
        settings.set_backend("cpp", subpackage)
