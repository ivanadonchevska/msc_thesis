import re

import pandas as pd

from filters import drop_by_field
from cleaners import (
    apply_to_source,
    drop_lines,
    drop_phrase,
    filter_noise_articles,
    fix_split_first_letter,
    is_noise_line,
    strip_before_sentinel,
    strip_prefix,
    truncate_at_pattern,
    truncate_at_sentinel,
    remove_inline_separators,
    remove_duplicate_paragraphs,
    decode_html_entities,
    strip_html,
    strip_media_suffixes,
)

_CLOSE_PATTERN = re.compile(r"\sЗатвори\s+[А-Я]")
_24CHASA_PROMPT = "Разберете големите новини от последното денонощие накуп Запиши се"
_24CHASA_PHRASES = [
    '"24 часа" в мейла ти. Разберете големите новини от последното денонощие накуп'
]
_NOVA_NAV = re.compile(
    r"^Здравей, България.*?(?:и за сметка на кои партии|конструктивна опозиция)\s+(?=[А-ЯA-Z,„])",
    re.DOTALL,
)


def _strip_nova_nav(text: str) -> str:
    if not isinstance(text, str) or not text.startswith("Здравей, България"):
        return text
    match = _NOVA_NAV.search(text)
    if match:
        return text[match.end() :].strip()
    return text


def clean_vesti(df: pd.DataFrame) -> pd.DataFrame:
    df = apply_to_source(df, fix_split_first_letter, source="vesti")
    df = apply_to_source(
        df, lambda t: truncate_at_sentinel(t, ["Още от автора"]), source="vesti"
    )
    return df


def clean_monitor(df: pd.DataFrame) -> pd.DataFrame:
    df = apply_to_source(
        df,
        lambda t: drop_lines(t, lambda line: "абонирайте се" in line.lower()),
        source="monitor",
    )
    df = apply_to_source(df, fix_split_first_letter, source="monitor")
    return df


def clean_standartnews(df: pd.DataFrame) -> pd.DataFrame:
    df = apply_to_source(
        df,
        lambda t: truncate_at_sentinel(
            t, ["Последвайте ни в Google News Showcase за важните новини"]
        ),
        source="standartnews",
    )
    return df


def clean_segabg(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_by_field(df, ["Спортът по телевизията"], field="title", source="segabg")
    return df


def clean_bta(df: pd.DataFrame) -> pd.DataFrame:
    df = apply_to_source(df, lambda t: strip_prefix(t, "site.bta"), source="bta")
    df = apply_to_source(df, lambda t: truncate_at_sentinel(t, ["/"]), source="bta")
    return df


def clean_actualno(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_by_field(
        df, ["валутен калкулатор"], field="title", source="actualno", match="contains"
    )
    df = apply_to_source(
        df,
        lambda t: drop_lines(
            t, lambda line: line.strip().startswith(("Снимка:", "Източник:", "Още:"))
        ),
        source="actualno",
    )
    return df


def clean_nova(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_by_field(
        df, ["Новините на NOVA", "Прогноза за времето"], field="title", source="nova"
    )

    df = drop_by_field(df, ["Новините на NOVA"], field="full_text", source="nova")

    df = apply_to_source(df, _strip_nova_nav, source="nova")
    df = drop_by_field(df, ["Здравей, България"], field="full_text", source="nova")
    df = apply_to_source(
        df, lambda t: truncate_at_sentinel(t, ["Последвайте ни"]), source="nova"
    )

    df = apply_to_source(
        df,
        lambda t: drop_lines(
            t,
            lambda line: line.strip().startswith(
                (
                    "Снимка:",
                    "Снимки:",
                    "Редактор:",
                    "Повече гледайте във видеото",
                    "Цялото предаване гледайте във видеото",
                )
            ),
        ),
        source="nova",
    )
    return df


def clean_24chasa(df: pd.DataFrame) -> pd.DataFrame:

    df = drop_by_field(
        df,
        ["състоянието на пътищата", "изтегли приложението от тук"],
        field="title",
        source="24chasa",
        match="contains",
    )
    df = drop_by_field(
        df,
        [
            "Спорт по телевизията днес",
            '"24 часа" на 26 април',
            "Вижте отговора на фотозагадката:",
            "Фотозагадка:",
            "Виж отговора на фотозагадката:",
        ],
        field="title",
        source="24chasa",
    )

    df = drop_by_field(
        df, ["Малкия Иванчо"], field="full_text", source="24chasa", match="contains"
    )
    df = apply_to_source(
        df,
        lambda t: (
            strip_before_sentinel(t, _24CHASA_PROMPT)
            if isinstance(t, str) and _24CHASA_PROMPT in re.sub(r"\s+", " ", t)
            else t
        ),
        source="24chasa",
    )
    df = apply_to_source(
        df, lambda t: drop_phrase(t, _24CHASA_PHRASES), source="24chasa"
    )
    df = apply_to_source(
        df,
        lambda t: truncate_at_sentinel(t, ["Последвайте ни в Google News Showcase"]),
        source="24chasa",
    )
    df = apply_to_source(
        df, lambda t: truncate_at_pattern(t, _CLOSE_PATTERN), source="24chasa"
    )
    df = apply_to_source(
        df,
        lambda t: drop_lines(
            t, lambda line: line.strip().lower().startswith("снимка:")
        ),
        source="24chasa",
    )
    return df


def clean_fakti(df: pd.DataFrame) -> pd.DataFrame:
    df = apply_to_source(
        df,
        lambda t: truncate_at_sentinel(t, ["Поставете оценка:", "Напиши коментар:"]),
        source="fakti",
    )
    return df


def clean_global(df: pd.DataFrame) -> pd.DataFrame:
    df["full_text"] = df["full_text"].apply(lambda t: drop_lines(t, is_noise_line))
    df["full_text"] = df["full_text"].apply(remove_inline_separators)
    df["full_text"] = df["full_text"].apply(decode_html_entities)
    df["full_text"] = df["full_text"].apply(remove_duplicate_paragraphs)
    df["title"] = df["title"].apply(strip_html)
    df["title"] = df["title"].apply(decode_html_entities)
    df["title"] = df["title"].apply(strip_media_suffixes)
    df["full_text"] = df["full_text"].apply(strip_media_suffixes)
    df = filter_noise_articles(df)

    return df
