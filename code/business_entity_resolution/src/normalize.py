"""
Normalization for business names and addresses.

Design constraints this module respects:
  - No external gazetteers/dictionaries — the abbreviation map below is
    hand-written from the noise patterns the problem statement itself
    describes (Corp/Corporation, Pvt/Private, Rd/Road, etc.), which is
    explicitly allowed ("small hand-written normalization dictionaries").
  - No hard-coded assumption about which countries/scripts exist. Unicode
    NFKD + diacritic stripping is a *generic* algorithm (works for French
    accents, Vietnamese, transliterated Spanish, etc.) — it is not a lookup
    of any specific country's data, so it stays valid for the unseen
    France slice.
"""

import re
import unicodedata

# Hand-written legal-suffix / common-abbreviation map. Keys and values are
# both stored in the token set during blocking so either spelling matches;
# during scoring, name features are computed on the *expanded* form so
# "Corp" vs "Corporation" reads as identical rather than merely similar.
ABBREVIATION_MAP = {
    "corp": "corporation",
    "co": "company",
    "ltd": "limited",
    "pvt": "private",
    "inc": "incorporated",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "assoc": "associates",
    "intl": "international",
    "mfg": "manufacturing",
    "svcs": "services",
    "svc": "service",
    "grp": "group",
    "&": "and",
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "apt": "apartment",
    "apts": "apartments",
    "fl": "floor",
    "flr": "floor",
    "bldg": "building",
    "twp": "township",
    "nr": "near",
    "opp": "opposite",
}

# Tokens that carry almost no discriminative signal for *blocking* (too
# common across unrelated businesses) — they are still used in similarity
# *features*, just excluded from the token-inverted-index blocking keys.
LEGAL_SUFFIX_TOKENS = {
    "corp", "corporation", "co", "company", "ltd", "limited", "pvt",
    "private", "inc", "incorporated", "llc", "llp", "group", "grp",
    "and", "the", "of", "services", "service",
}

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def strip_diacritics(text: str) -> str:
    """Generic Unicode diacritic stripping. Works on French accents,
    Vietnamese, etc. equally — this is an algorithm, not a language-specific
    lookup table, so it needs no per-country branch."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def basic_clean(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = strip_diacritics(text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def expand_abbreviations(tokens: list[str]) -> list[str]:
    return [ABBREVIATION_MAP.get(tok, tok) for tok in tokens]


def normalize_name(raw_name: str) -> dict:
    """Returns both a cleaned string (for fuzzy-string features) and a
    token list (for blocking + jaccard-style features)."""
    cleaned = basic_clean(raw_name)
    tokens = cleaned.split() if cleaned else []
    expanded_tokens = expand_abbreviations(tokens)
    expanded_str = " ".join(expanded_tokens)
    return {
        "clean": cleaned,
        "expanded": expanded_str,
        "tokens": expanded_tokens,
    }


def normalize_address(raw_address) -> dict:
    if raw_address is None or (isinstance(raw_address, float)):
        # pandas turns missing TSV cells into NaN (float)
        return {"clean": "", "expanded": "", "tokens": [], "present": False}
    cleaned = basic_clean(raw_address)
    if not cleaned:
        return {"clean": "", "expanded": "", "tokens": [], "present": False}
    tokens = cleaned.split()
    expanded_tokens = expand_abbreviations(tokens)
    return {
        "clean": cleaned,
        "expanded": " ".join(expanded_tokens),
        "tokens": expanded_tokens,
        "present": True,
    }


def blocking_tokens(tokens: list[str], min_len: int = 3) -> set:
    """Tokens usable as blocking keys: drop legal-suffix boilerplate and
    very short tokens (too common, would flood the inverted index)."""
    return {
        t for t in tokens
        if len(t) >= min_len and t not in LEGAL_SUFFIX_TOKENS
    }


def char_ngrams(text: str, n: int = 3) -> set:
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}
