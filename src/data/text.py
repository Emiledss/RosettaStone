"""Comptage des mots tenant compte de l'élision (socle des étapes 1 et 3).

Transpose en snake_case le tokeniseur validé dans le notebook AED (cellule 18,
tests cellules 19-20). Règles préservées :

- l'apostrophe d'élision est une **frontière de mot** (`l'ami` = 2 unités,
  ADR-0002), sauf pour une liste d'exceptions insécables (`aujourd'hui`,
  `quelqu'un`, ...) ;
- le clitique reste attaché **à gauche en FR**, **à droite en EN** (ADR-0004) ;
- la ponctuation seule n'est pas comptée comme un mot, par défaut.

⚠️ Piège pandas 3 / PyArrow (voir CLAUDE.md) : ne jamais utiliser `.str.count()` /
`.str.contains()` pour ce genre de comptage, le moteur RE2 traite `\\w` en ASCII
seul (`"été"` y compterait 1 caractère de mot au lieu de 3). Tout ce module est
écrit en Python pur (`re` + `str.isalnum()`), Unicode-correct.
"""

from __future__ import annotations

import re

# Variantes typographiques de l'apostrophe rencontrées dans le corpus.
APOSTROPHES = "'’‘´"  # droite, courbe, ouvrante, accent aigu
APOSTROPHE_CLASS = f"[{re.escape(APOSTROPHES)}]"
SENTINEL = "\x00"  # marqueur temporaire des exceptions, le temps du découpage

APOSTROPHE_RE = re.compile(APOSTROPHE_CLASS)
# une apostrophe ENCADRÉE par deux caractères de mot est une frontière
BOUNDARY_RE = re.compile(r"(?<=\w)" + APOSTROPHE_CLASS + r"(?=\w)")
# contraction négative anglaise : didn't -> did + n't (convention Penn Treebank)
NT_RE = re.compile(r"(?<=\w)n" + APOSTROPHE_CLASS + r"t(?!\w)", re.IGNORECASE)

# Côté auquel l'apostrophe reste attachée lors du découpage.
#   "left"  -> "l'ami" -> ["l'", "ami"]   le clitique élidé français précède
#   "right" -> "it's"  -> ["it", "'s"]    le clitique contracté anglais suit
APOSTROPHE_ATTACH = {"en": "right", "fr": "left"}

# Langues où "n't" est un clitique à part entière.
NT_SPLIT_LANGS = {"en"}

# Mots où l'apostrophe est INTERNE et ne sépare pas deux unités (vérifiés sur le
# corpus : aujourd'hui et quelqu'un sont les seules exceptions vraiment fréquentes
# en français ; presqu'à / presqu'une restent de vraies élisions).
APOSTROPHE_EXCEPTIONS = {
    "en": ["o'clock", "o'er", "ma'am"],
    "fr": [
        "aujourd'hui",
        "quelqu'un", "quelqu'une",
        "chef-d'œuvre", "chefs-d'œuvre", "hors-d'œuvre",
        "main-d'œuvre", "mains-d'œuvre",
        "grand'mère", "grand'père", "grand'route",
        "prud'homme", "prud'hommes", "prud'homie",
        "presqu'île", "presqu'îles",
        "entr'acte", "entr'actes", "entr'aide",
    ],
}

# Motifs supplémentaires : patronymes irlandais (O'Brien, O'Connor...).
APOSTROPHE_EXCEPTION_PATTERNS = {
    "en": [r"[Oo]" + APOSTROPHE_CLASS + r"[A-Z]\w*"],
    "fr": [],
}

_EXCEPTION_REGEX_CACHE: dict[str, re.Pattern[str] | None] = {}


def build_exception_regex(words: list[str], patterns: list[str]) -> re.Pattern[str] | None:
    """Construit la regex des mots dont l'apostrophe ne doit pas servir de séparateur."""
    alternatives = [
        APOSTROPHE_CLASS.join(re.escape(seg) for seg in re.split(APOSTROPHE_CLASS, word))
        for word in words
    ]
    alternatives.extend(patterns)
    if not alternatives:
        return None
    return re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)", re.IGNORECASE)


def tokenizer_config(lang: str) -> dict[str, object]:
    """Réglages de tokenisation d'une langue, à passer à `tokenize_sentence`."""
    if lang not in _EXCEPTION_REGEX_CACHE:
        _EXCEPTION_REGEX_CACHE[lang] = build_exception_regex(
            APOSTROPHE_EXCEPTIONS.get(lang, []),
            APOSTROPHE_EXCEPTION_PATTERNS.get(lang, []),
        )
    return {
        "exception_regex": _EXCEPTION_REGEX_CACHE[lang],
        "split_apostrophe": True,
        "keep_punctuation": False,
        "attach": APOSTROPHE_ATTACH.get(lang, "left"),
        "nt_split": lang in NT_SPLIT_LANGS,
    }


def tokenize_sentence(
    text: str,
    exception_regex: re.Pattern[str] | None = None,
    split_apostrophe: bool = True,
    keep_punctuation: bool = False,
    attach: str = "left",
    nt_split: bool = False,
) -> list[str]:
    """Découpe une phrase en unités lexicales et retourne la liste des tokens.

    split_apostrophe : une apostrophe ENTRE deux caractères de mot sépare deux
                        unités ("l'ami" -> l' + ami), sauf pour les exceptions
                        ("aujourd'hui"). En bord de mot elle ne sépare pas
                        ("parents'", "'em" -> 1 token).
    attach            : côté auquel l'apostrophe reste collée ("left" pour le
                        français, "right" pour l'anglais). N'affecte QUE la forme
                        des tokens, jamais leur nombre.
    nt_split          : traite "n't" comme un clitique ("didn't" -> did + n't).
    keep_punctuation  : si False, les tokens sans aucun caractère alphanumérique
                        ("!", "»", "--") ne sont pas comptés.
    """
    if split_apostrophe:
        if exception_regex is not None:
            # neutralise l'apostrophe des exceptions le temps du découpage
            text = exception_regex.sub(
                lambda m: APOSTROPHE_RE.sub(SENTINEL, m.group(0)), text
            )
        if nt_split:
            # "didn't" -> "did n't" ; l'apostrophe de n't est neutralisée pour que
            # la règle générale ne redécoupe pas le clitique une seconde fois.
            text = NT_RE.sub(lambda m: " " + APOSTROPHE_RE.sub(SENTINEL, m.group(0)), text)
        if attach == "right":
            text = BOUNDARY_RE.sub(lambda m: " " + m.group(0), text)
        else:
            text = BOUNDARY_RE.sub(lambda m: m.group(0) + " ", text)
        text = text.replace(SENTINEL, "'")

    tokens = text.split()
    if not keep_punctuation:
        tokens = [t for t in tokens if any(ch.isalnum() for ch in t)]
    return tokens


def tokenize_words(text: str, lang: str) -> list[str]:
    """Découpe `text` en mots pour la langue `lang` ("fr" ou "en")."""
    config = tokenizer_config(lang)
    return tokenize_sentence(str(text).strip(), **config)


def count_words(text: str, lang: str) -> int:
    """Nombre de mots de `text` pour la langue `lang` ("fr" ou "en").

    Comptage en Python pur (voir avertissement en tête de module) : ne pas
    remplacer par `.str.count()` / `.str.contains()` sur une colonne pandas.
    """
    return len(tokenize_words(text, lang))


# Retire la ponctuation en début/fin de token mais garde l'apostrophe :
# "chose." -> "chose", "l'" -> "l'" (l'élision reste une forme à part entière).
# Reproduit EXACTEMENT `TOKEN_STRIP_RE` / `normalizeToken` du notebook AED
# (cellule 34) : `\w` est ici évalué par le module `re` de Python (Unicode-
# correct), jamais par une méthode `.str.*` de pandas (moteur RE2, ASCII seul
# sur `\w` -- piège documenté dans CLAUDE.md).
TOKEN_STRIP_RE = re.compile(
    r"^[^\w" + re.escape(APOSTROPHES) + r"]+|[^\w" + re.escape(APOSTROPHES) + r"]+$"
)


def normalize_token(token: str) -> str:
    """Minuscule + retrait de la ponctuation de bord (identique à `normalizeToken`
    du notebook AED)."""
    return TOKEN_STRIP_RE.sub("", token.lower())


def normalize_text(text: str, lang: str) -> str:
    """Normalise une phrase pour la langue `lang` : tokenise (`tokenize_words`),
    normalise chaque token (`normalize_token` : minuscule + ponctuation de bord
    retirée, apostrophe préservée), écarte les tokens devenus vides, et rejoint
    le résultat par un espace."""
    tokens = (normalize_token(t) for t in tokenize_words(text, lang))
    return " ".join(t for t in tokens if t)
