"""Generic prompt snippets for parameter extraction (ordinal/index-based).

Rules mirror ksr-style heuristics: quoted spans, template tails, paths/tokens,
regex keywords — without branching on specific parameter names.
"""

from __future__ import annotations

import re

_DIGIT_NUMBER_RE = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_WORD_NUMBERS: dict[str, str] = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}
_WORD_NUMBER_RE = re.compile(
    r"\b(" + "|".join(_WORD_NUMBERS) + r")\b",
    re.IGNORECASE,
)

_TOKEN_PATTERN = re.compile(
    r"([A-Za-z]:\\[^\s'\"]+)" r"|(/[^\s'\"]{4,})" r"|(\b[a-zA-Z][\w]*-[\w]+\b)"
)

_SYMBOL_WORDS: tuple[tuple[str, str], ...] = (
    ("asterisks", "*"),
    ("asterisk", "*"),
    ("stars", "*"),
    ("star", "*"),
    ("hash", "#"),
    ("pound", "#"),
    ("dash", "-"),
    ("hyphen", "-"),
    ("underscore", "_"),
    ("dot", "."),
    ("period", "."),
    ("slash", "/"),
    ("backslash", "\\"),
    ("space", " "),
)

_REGEX_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("number", "digit", "numeric", "integer"), r"\d+"),
    (("vowel",), r"[aeiouAEIOU]"),
    (("uppercase", "capital letter"), r"[A-Z]"),
    (("lowercase", "small letter"), r"[a-z]"),
    (("space", "whitespace"), r"\s+"),
    (("word", "alphanumeric"), r"\w+"),
    (("punctuation", "special char", "symbol"), r"[^\w\s]"),
)


def quoted_spans(prompt: str) -> list[str]:
    found = re.findall(r"'([^']*)'|\"([^\"]+)\"", prompt)
    return [a or b for a, b in found if (a or b)]


def template_tail_if_braces(prompt: str) -> str | None:
    m = re.search(r":\s*(.+)$", prompt)
    if m and ("{" in m.group(1) or "}" in m.group(1)):
        return m.group(1).strip()
    return None


def first_path_windows(prompt: str) -> str | None:
    pm = re.search(r'([A-Za-z]:\\[^\s\'"]+|/[^\s\'"]+)', prompt)
    if not pm:
        return None
    return pm.group(1).replace("\\\\", "\\")


def token_pattern_values(prompt: str) -> list[str]:
    out: list[str] = []
    for match in _TOKEN_PATTERN.finditer(prompt):
        token_val = next(g for g in match.groups() if g is not None)
        if re.match(r"[A-Za-z]:\\", token_val):
            token_val = token_val.replace("\\\\", "\\")
        out.append(token_val)
    return out


def symbol_from_keywords(prompt: str) -> str | None:
    lower = prompt.lower()
    for word, symbol in _SYMBOL_WORDS:
        if word in lower:
            return symbol
    return None


def plain_text_tail(prompt: str) -> str | None:
    if not prompt.strip():
        return None
    if re.search(r"['\"]", prompt):
        return None
    if re.search(r"([A-Za-z]:\\[^\s'\"]+|/[^\s'\"]+)", prompt):
        return None

    words = re.findall(r"[A-Za-z][\w-]*", prompt)
    if not words:
        return None

    lead_words = {
        "greet",
        "reverse",
        "replace",
        "substitute",
        "read",
        "execute",
        "run",
        "calculate",
        "format",
        "use",
    }
    if words[0].lower() in lead_words and len(words) == 2:
        return words[1].strip() or None

    if len(words) <= 2:
        return " ".join(words).strip()

    return None


def try_non_regex_string(
    prompt: str,
    string_param_index: int,
    last_regex_value: str | None,
) -> str | None:
    tt = template_tail_if_braces(prompt)
    if tt is not None:
        return tt

    values = quoted_spans(prompt)
    if values:
        longest = max(values, key=len)
        remaining = [
            v for v in values if v != longest and v != last_regex_value
        ]
        if string_param_index == 0:
            chosen = longest
        elif string_param_index - 1 < len(remaining):
            chosen = remaining[string_param_index - 1]
        else:
            chosen = None
        if chosen is not None:
            return chosen

    pm = first_path_windows(prompt)
    if pm is not None and string_param_index == 0:
        return pm

    tokens = token_pattern_values(prompt)
    if string_param_index < len(tokens):
        return tokens[string_param_index]

    m = re.search(r":\s*(.+)$", prompt)
    if m:
        return m.group(1).strip()

    if string_param_index == 0:
        plain = plain_text_tail(prompt)
        if plain is not None:
            return plain

    return symbol_from_keywords(prompt)


def regex_pattern_from_prompt(prompt: str) -> str | None:
    text = prompt.lower()
    mq = re.search(r"['\"](\w+)['\"]", prompt)
    if mq:
        return mq.group(1)

    for keywords, pattern in _REGEX_KEYWORDS:
        if any(kw in text for kw in keywords):
            return pattern
    return None


def ordered_numeric_strings(prompt: str) -> list[str]:
    hits_outside_quotes: list[tuple[int, str]] = []
    hits_inside_quotes: list[tuple[int, str]] = []
    quoted_spans = [
        match.span() for match in re.finditer(r"'[^']*'|\"[^\"]*\"", prompt)
    ]

    def _inside_quotes(index: int) -> bool:
        return any(start <= index < end for start, end in quoted_spans)

    for m in _DIGIT_NUMBER_RE.finditer(prompt):
        target = (
            hits_inside_quotes
            if _inside_quotes(m.start())
            else hits_outside_quotes
        )
        target.append((m.start(), m.group()))
    for m in _WORD_NUMBER_RE.finditer(prompt):
        target = (
            hits_inside_quotes
            if _inside_quotes(m.start())
            else hits_outside_quotes
        )
        target.append((m.start(), _WORD_NUMBERS[m.group().lower()]))

    hits = hits_outside_quotes if hits_outside_quotes else hits_inside_quotes
    hits.sort(key=lambda h: h[0])
    return [v for _, v in hits]


def parse_numeric_at_index(
    prompt: str,
    numeric_param_index: int,
    *,
    integer_only: bool,
) -> int | float | None:
    matches = ordered_numeric_strings(prompt)
    if numeric_param_index >= len(matches):
        return None
    value_text = matches[numeric_param_index]
    if integer_only:
        try:
            return int(float(value_text))
        except ValueError:
            return None
    try:
        return float(value_text)
    except ValueError:
        return None


def regex_candidate_patterns() -> list[str]:
    return [
        r"\d+",
        r"[aeiouAEIOU]",
        r"[A-Z]",
        r"[a-z]",
        r"\s+",
        r"\w+",
        r"[^\w\s]",
    ]
