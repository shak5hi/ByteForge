import re

from unidecode import unidecode

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
# "Knowledge Park II" and "knowledge park 2" should match. "i" is left alone (too common as a word).
_ROMAN = {"ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10"}


def normalize(value: str | None) -> str:
    """Searchable form of a name: ASCII, lowercase, punctuation collapsed, Roman numerals as digits."""
    if not value:
        return ""
    text = _NON_ALNUM.sub(" ", unidecode(value).lower().replace("&", " and ")).strip()
    return " ".join(_ROMAN.get(tok, tok) for tok in text.split())
