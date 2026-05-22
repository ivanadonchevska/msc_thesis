import html
import re

import pandas as pd
from bs4 import BeautifulSoup

_SOCIAL_ATTR = re.compile(r"\(@\w+\).*\b\d{4}\b")
_EMOJI_RE = re.compile(r"[\U0001F000-\U0001FFFF\U00002600-\U000027BF]")
_MEDIA_SUFFIX_RE = re.compile(
    r"\(\s*(?:видео|снимка|снимки)" r"(?:\s*\+\s*(?:видео|снимка|снимки))?" r"\s*\)",
    re.IGNORECASE,
)


def is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    content = re.sub(r"^[—–-]+\s*", "", stripped)
    if _SOCIAL_ATTR.search(stripped):
        return True
    if content.startswith("@"):
        return True
    words = content.split()
    if words and sum(1 for w in words if _EMOJI_RE.search(w)) / len(words) > 0.3:
        return True
    if content and not any("Ѐ" <= c <= "ӿ" for c in content):
        return True
    return False


def drop_lines(text: str, predicate) -> str:
    if not isinstance(text, str):
        return text
    return "\n".join(line for line in text.split("\n") if not predicate(line))


def truncate_at_sentinel(text: str, sentinels: list[str]) -> str:
    if not isinstance(text, str):
        return text
    text_lower = text.lower()
    for sentinel in sentinels:
        idx = text_lower.find(sentinel.lower())
        if idx != -1:
            text = text[:idx].strip()
    return text


def truncate_at_pattern(text: str, pattern) -> str:
    if not isinstance(text, str):
        return text
    match = pattern.search(text)
    if match:
        return text[: match.start()].strip()
    return text


def strip_before_sentinel(text: str, sentinel: str) -> str:
    if not isinstance(text, str):
        return text
    idx = text.find(sentinel)
    if idx != -1:
        return text[idx + len(sentinel) :].strip()
    return text


def strip_prefix(text: str, prefix: str) -> str:
    if not isinstance(text, str):
        return text
    if text.startswith(prefix):
        return text[len(prefix) :].strip()
    return text


def drop_phrase(text: str, phrases: list[str]) -> str:
    if not isinstance(text, str):
        return text
    for phrase in phrases:
        text = text.replace(phrase, "")
    return text.strip()


def fix_split_first_letter(text: str) -> str:
    if not isinstance(text, str):
        return text
    match = re.match(r"^([А-Я]) ([а-я])", text)
    if match:
        return match.group(1) + match.group(2) + text[3:]
    return text


def strip_html(text: str) -> str:
    if not isinstance(text, str):
        return text
    return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)


def decode_html_entities(text: str) -> str:
    if not isinstance(text, str):
        return text
    prev = None
    while prev != text:
        prev = text
        text = html.unescape(text)
    return text


def remove_inline_separators(text: str) -> str:
    if not isinstance(text, str):
        return text
    return re.sub(r"(?:-\s*){5,}", " ", text).strip()


def remove_duplicate_paragraphs(text: str) -> str:
    if not isinstance(text, str):
        return text
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    seen = []
    for p in paragraphs:
        if p not in seen:
            seen.append(p)
    return "\n\n".join(seen)


def strip_media_suffixes(text: str) -> str:
    if not isinstance(text, str):
        return text
    return _MEDIA_SUFFIX_RE.sub("", text).strip()


def apply_to_source(df: pd.DataFrame, fn, source: str | None = None) -> pd.DataFrame:
    if source:
        mask = df["source"] == source
        df.loc[mask, "full_text"] = df.loc[mask, "full_text"].apply(fn)
    else:
        df["full_text"] = df["full_text"].apply(fn)
    return df
