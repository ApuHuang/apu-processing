import string

from apu_processing.i18n import _EN, _ZH, Msg, set_language, tr


def _fields(text: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def test_catalogs_have_same_keys_and_placeholders():
    assert _ZH.keys() == _EN.keys()
    for key in _ZH:
        assert _fields(_ZH[key]) == _fields(_EN[key]), key


def test_msg_follows_language():
    msg = Msg("msg.unsupported_format", name="x.xisf")
    set_language("en")
    try:
        assert str(msg) == "Unsupported file format: x.xisf"
    finally:
        set_language("zh")
    assert str(msg) == tr("msg.unsupported_format", name="x.xisf")
