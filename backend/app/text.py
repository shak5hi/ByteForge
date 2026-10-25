import re

from unidecode import unidecode

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(value: str | None) -> str:
    """Searchable form of a name: ASCII, lowercase, punctuation collapsed to spaces."""
    if not value:
        return ""
    return _NON_ALNUM.sub(" ", unidecode(value).lower().replace("&", " and ")).strip()
