"""Génère `reports/soutenance/Rosetta_soutenance.pptx`.

Source de vérité pour la présentation de soutenance : ce script est idempotent,
il peut être relancé à tout moment (par exemple quand de nouveaux résultats
Optuna seront disponibles) et régénère intégralement le `.pptx` et les figures
qui en dépendent.

Étapes :
1. Extraction des figures PNG stockées dans les outputs de
   `notebooks/Rosetta_AED.ipynb` (cellules de code contenant un output
   `image/png`) vers `reports/soutenance/figures/aed_<n>_<slug>.png`.
2. Copie des figures déjà produites par d'autres notebooks/scripts
   (tokenisation, entraînement, Optuna) vers `reports/soutenance/figures/`.
3. Génération du diaporama (22 diapositives).

Lecture seule sur `checkpoints/`, `reports/optuna/` et `reports/runs/` : une
étude Optuna peut y écrire en parallèle. Ce script ne fait qu'y lire.
"""

from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_AED = ROOT / "notebooks" / "Rosetta_AED.ipynb"
FIGURES_DIR = ROOT / "reports" / "soutenance" / "figures"
OUTPUT_PPTX = ROOT / "reports" / "soutenance" / "Rosetta_soutenance.pptx"

# Cellules de code de l'AED contenant une figure PNG à extraire, avec un
# slug lisible pour le nom de fichier. Ordre = ordre d'apparition dans le
# notebook.
AED_FIGURE_CELLS: dict[int, str] = {
    26: "distribution_longueurs",
    27: "dispersion_longueurs",
    28: "ratio_longueur",
    35: "zipf",
    37: "couverture_oov",
    39: "couverture_seuils",
    41: "hapax_ttr",
    42: "croissance_vocabulaire",
}

# Figures déjà produites ailleurs, à recopier (lecture seule sur la source).
# fig_importance_gru.png : préférer la source runs_hf (étude v4) si présente,
# sinon replier sur reports/optuna/ (voir résolution ci-dessous).
_FIG_IMPORTANCE_GRU_HF = ROOT / "reports" / "runs_hf" / "reports" / "optuna" / "fig_importance_gru.png"
_FIG_IMPORTANCE_GRU_FALLBACK = ROOT / "reports" / "optuna" / "fig_importance_gru.png"

REUSED_FIGURES: dict[str, Path] = {
    "fig_zipf_bpe_vs_unigram.png": ROOT / "reports" / "tokenization" / "fig_zipf_bpe_vs_unigram.png",
    "fig_longueur_tokens_caracteres.png": ROOT / "reports" / "tokenization" / "fig_longueur_tokens_caracteres.png",
    "fig_fragmentation.png": ROOT / "reports" / "tokenization" / "fig_fragmentation.png",
    "courbes_entrainement.png": ROOT / "reports" / "runs" / "courbes_entrainement.png",
    "fig_importance_gru.png": (
        _FIG_IMPORTANCE_GRU_HF if _FIG_IMPORTANCE_GRU_HF.exists() else _FIG_IMPORTANCE_GRU_FALLBACK
    ),
}


# --------------------------------------------------------------------------- #
# Étape 1 — extraction des figures de l'AED
# --------------------------------------------------------------------------- #


def extract_aed_figures(notebook_path: Path, out_dir: Path) -> dict[int, Path]:
    """Extrait les PNG des outputs `image/png` des cellules listées.

    Ne modifie jamais le notebook (lecture seule) ; écrit uniquement dans
    `out_dir`.
    """
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[int, Path] = {}

    for cell_index, slug in AED_FIGURE_CELLS.items():
        cell = cells[cell_index]
        if cell.get("cell_type") != "code":
            raise ValueError(f"cellule {cell_index} n'est pas une cellule de code")

        png_b64 = None
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "image/png" in data:
                png_b64 = data["image/png"]
                break

        if png_b64 is None:
            raise ValueError(
                f"cellule {cell_index} : aucun output image/png trouvé "
                "(le notebook a-t-il été exécuté avec ses outputs stockés ?)"
            )

        if isinstance(png_b64, list):
            png_b64 = "".join(png_b64)

        raw = base64.b64decode(png_b64)
        out_path = out_dir / f"aed_{cell_index}_{slug}.png"
        out_path.write_bytes(raw)
        written[cell_index] = out_path

    return written


def copy_reused_figures(out_dir: Path) -> dict[str, Path]:
    """Copie (lecture seule sur la source) les figures déjà produites ailleurs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    for name, source in REUSED_FIGURES.items():
        if not source.exists():
            raise FileNotFoundError(f"figure attendue absente : {source}")
        dest = out_dir / name
        shutil.copy2(source, dest)
        copied[name] = dest
    return copied


# --------------------------------------------------------------------------- #
# Constantes de style
# --------------------------------------------------------------------------- #

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

COLOR_TITLE = RGBColor(0x1A, 0x1A, 0x1A)
COLOR_TEXT = RGBColor(0x2B, 0x2B, 0x2B)
COLOR_ACCENT = RGBColor(0x2E, 0x5C, 0x8A)
COLOR_MUTED = RGBColor(0x6E, 0x6E, 0x6E)
COLOR_HEADER_BG = RGBColor(0x2E, 0x5C, 0x8A)
COLOR_HEADER_TEXT = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Calibri"
TITLE_SIZE = Pt(30)
SUBTITLE_SIZE = Pt(20)
BODY_SIZE = Pt(18)  # plancher imposé pour tout le texte de corps (>= 18 pt)
TABLE_SIZE = Pt(18)
TABLE_HEADER_SIZE = Pt(18)

MARGIN = Inches(0.5)


def add_blank_slide(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


TITLE_TOP = Inches(0.35)


def add_title(slide, text: str, *, size=TITLE_SIZE, top=None):
    if top is None:
        top = TITLE_TOP
    box = slide.shapes.add_textbox(MARGIN, top, SLIDE_W - 2 * MARGIN, Inches(0.9))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = size
    p.font.bold = True
    p.font.color.rgb = COLOR_TITLE
    p.font.name = FONT
    return box


def add_bullets(slide, bullets, left, top, width, height, *, size=BODY_SIZE, bullet_char="•"):
    """Ajoute une zone de texte à puces. `bullets` : liste de (texte, niveau).

    Toutes les puces restent à `size` (>= 18 pt) : pas de sous-niveau réduit
    sous le plancher de lisibilité imposé.
    """
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    for item in bullets:
        if isinstance(item, tuple):
            text, level = item
        else:
            text, level = item, 0
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        indent = "    " * level
        marker = bullet_char if level == 0 else "–"
        p.text = f"{indent}{marker} {text}"
        p.font.size = size
        p.font.color.rgb = COLOR_TEXT
        p.font.name = FONT
        p.space_after = Pt(8)
    return box


def add_picture_fit(slide, image_path: Path, left, top, max_width, max_height):
    """Insère une image en respectant son ratio, dans une boîte max_width x max_height."""
    from PIL import Image

    with Image.open(image_path) as im:
        img_w, img_h = im.size
    ratio = img_w / img_h
    box_ratio = max_width / max_height

    if ratio > box_ratio:
        width = max_width
        height = Emu(int(max_width / ratio))
    else:
        height = max_height
        width = Emu(int(max_height * ratio))

    x = Emu(int(left + (max_width - width) / 2))
    y = Emu(int(top + (max_height - height) / 2))
    return slide.shapes.add_picture(str(image_path), x, y, width=width, height=height)


def add_caption(slide, text: str, left, top, width):
    """Légende sous une figure. Reste au plancher de 18 pt (texte de corps)."""
    box = slide.shapes.add_textbox(left, top, width, Inches(0.4))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = BODY_SIZE
    p.font.italic = True
    p.font.color.rgb = COLOR_MUTED
    p.font.name = FONT
    p.alignment = PP_ALIGN.CENTER
    return box


def add_table(slide, headers, rows, left, top, width, height):
    n_rows = len(rows) + 1
    n_cols = len(headers)
    shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, height)
    table = shape.table

    for c, header in enumerate(headers):
        cell = table.cell(0, c)
        cell.text = str(header)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_HEADER_BG
        for p in cell.text_frame.paragraphs:
            p.font.size = TABLE_HEADER_SIZE
            p.font.bold = True
            p.font.color.rgb = COLOR_HEADER_TEXT
            p.font.name = FONT
            p.alignment = PP_ALIGN.CENTER

    for r, row in enumerate(rows, start=1):
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(value)
            for p in cell.text_frame.paragraphs:
                p.font.size = TABLE_SIZE
                p.font.color.rgb = COLOR_TEXT
                p.font.name = FONT
                p.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT

    return shape


def add_arrow_flow(slide, labels, left, top, width, height):
    """Diagramme simple : rectangles reliés par des flèches, sur une ligne."""
    n = len(labels)
    gap = Inches(0.35)
    box_w = Emu(int((width - gap * (n - 1)) / n))
    arrow_w = gap

    x = left
    shapes = []
    for i, label in enumerate(labels):
        rect = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, top, box_w, height)
        rect.fill.solid()
        rect.fill.fore_color.rgb = COLOR_ACCENT
        rect.line.color.rgb = COLOR_ACCENT
        tf = rect.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = label
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = COLOR_HEADER_TEXT
        p.font.name = FONT
        p.alignment = PP_ALIGN.CENTER
        shapes.append(rect)
        x = Emu(int(x + box_w))
        if i < n - 1:
            arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x, Emu(int(top + height / 2 - Inches(0.12))), arrow_w, Inches(0.24))
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = COLOR_MUTED
            arrow.line.color.rgb = COLOR_MUTED
            shapes.append(arrow)
            x = Emu(int(x + arrow_w))
    return shapes


# --------------------------------------------------------------------------- #
# Références bibliographiques (diapositive 22)
# --------------------------------------------------------------------------- #
# NB : titres canoniques (publiés) des 3 premières références -- à vérifier
# si une autre formulation était visée.
REFERENCES = [
    "Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H. & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.",
    "Bahdanau, D., Cho, K. & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate.",
    "Sennrich, R., Haddow, B. & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units.",
    "Bengio, S., Vinyals, O., Jaitly, N. & Shazeer, N. (2015). Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.",
    "Post, M. (2018). A Call for Clarity in Reporting BLEU Scores.",
    "Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q. & Artzi, Y. (2020). BERTScore: Evaluating Text Generation with BERT.",
    "Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. (2019). Optuna: A Next-Generation Hyperparameter Optimization Framework.",
]

# Effondrement de mode (RNN) : trois sources FR distinctes produisant la même
# sortie en génération libre, recopié en dur depuis `reports/optuna/trials_rnn.csv`.
MODE_COLLAPSE_EXAMPLES = [
    "« il se peut que le bonheur qui nous attend là-bas ne soit pas du tout le genre de bonheur que nous voudrions »",
    "« c'est dommage quand quelqu'un meurt »",
    "« je ne peux pas réfléchir avec ce bruit dit-elle en fixant des yeux la machine à écrire »",
]

# Désaccords BLEU/BERTScore (diapositive 17) : 3 paires où SacreBLEU est bas
# (< 10) mais BERTScore F1 est élevé (> 0,85), recopiées en dur (calcul coûteux)
# depuis les hypothèses du run gru v4 (`reports/runs_hf/reports/runs/v4/gru/hypotheses.jsonl`).
DISAGREEMENT_EXAMPLES = [
    # (source FR, hypothèse, référence, SacreBLEU, BERTScore F1)
    ("faux", "false", "wrong", 0.00, 0.854),
    (
        "on dirait qu' elle est saoule",
        "it looks like she is drunk",
        "she looks as if she were drunk",
        9.04,
        0.892,
    ),
    (
        "je n' en ai pas la moindre idée",
        "i do n't have the faintest idea",
        "i have n't the foggiest idea",
        9.82,
        0.891,
    ),
]


# --------------------------------------------------------------------------- #
# Diapositives
# --------------------------------------------------------------------------- #


def slide_01_titre(prs):
    slide = add_blank_slide(prs)
    box = slide.shapes.add_textbox(Inches(1.0), Inches(2.3), Inches(11.33), Inches(1.2))
    p = box.text_frame.paragraphs[0]
    p.text = "Rosetta"
    p.font.size = Pt(54)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT
    p.font.name = FONT
    p.alignment = PP_ALIGN.CENTER

    box2 = slide.shapes.add_textbox(Inches(1.0), Inches(3.5), Inches(11.33), Inches(0.8))
    tf2 = box2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "Traduction automatique FR→EN (Seq2Seq)"
    p2.font.size = Pt(28)
    p2.font.color.rgb = COLOR_TITLE
    p2.font.name = FONT
    p2.alignment = PP_ALIGN.CENTER

    box3 = slide.shapes.add_textbox(Inches(1.0), Inches(4.4), Inches(11.33), Inches(0.7))
    tf3 = box3.text_frame
    tf3.word_wrap = True
    p3 = tf3.paragraphs[0]
    p3.text = "RNN simple vs GRU, avec attention additive (Bahdanau) en option"
    p3.font.size = Pt(20)
    p3.font.color.rgb = COLOR_MUTED
    p3.font.name = FONT
    p3.alignment = PP_ALIGN.CENTER

    box4 = slide.shapes.add_textbox(Inches(1.0), Inches(5.3), Inches(11.33), Inches(0.6))
    tf4 = box4.text_frame
    p4 = tf4.paragraphs[0]
    p4.text = "Soutenance"
    p4.font.size = Pt(18)
    p4.font.color.rgb = COLOR_MUTED
    p4.font.name = FONT
    p4.alignment = PP_ALIGN.CENTER
    return slide


def slide_02_objectif(prs, figures):
    slide = add_blank_slide(prs)
    add_title(slide, "Objectif et périmètre")
    add_bullets(
        slide,
        [
            "Traduire automatiquement du français vers l'anglais avec un modèle seq2seq (encodeur-décodeur récurrent).",
            "Comparaison contrôlée : RNN simple vs GRU, avec attention additive de Bahdanau en option.",
            "Périmètre : corpus Tatoeba → nettoyage et anti-fuite → 6 configurations de tokenisation → recherche d'hyperparamètres (Optuna) → entraînement final → évaluation en génération libre (SacreBLEU, BERTScore).",
        ],
        Inches(0.6), Inches(1.5), Inches(6.4), Inches(5.4),
    )
    add_picture_fit(slide, figures["couverture_oov"], Inches(7.3), Inches(1.5), Inches(5.4), Inches(4.6))
    add_caption(
        slide,
        "Contexte : coût de la couverture lexicale (détail diapositive 7)",
        Inches(7.3), Inches(6.15), Inches(5.4),
    )
    return slide


def slide_03_corpus(prs, figures):
    slide = add_blank_slide(prs)
    add_title(slide, "Le corpus Tatoeba")
    add_bullets(
        slide,
        [
            "264 905 phrases brutes → 625 filtrées (0,24 %) → 264 280 nettoyées.",
            "1 402 doublons exacts retirés après normalisation → 262 878 paires finales.",
            "Longueur médiane 6–7 mots ; p99 ≈ 19–21 mots ; max 40 mots après coupe.",
            "Le corpus est très majoritairement sous le seuil de décrochage de l'attention de Bahdanau (~15–20 mots) → d'où les bins d'évaluation par longueur (diapositive 12).",
        ],
        Inches(0.6), Inches(1.5), Inches(6.4), Inches(5.4),
    )
    add_picture_fit(slide, figures["longueurs"], Inches(7.3), Inches(1.5), Inches(5.4), Inches(4.6))
    add_caption(
        slide,
        "Distribution des longueurs (caractères et mots), FR vs EN",
        Inches(7.3), Inches(6.15), Inches(5.4),
    )
    return slide


def slide_04_zipf_couverture(prs, figures):
    slide = add_blank_slide(prs)
    add_title(slide, "Zipf et couverture lexicale")
    add_bullets(
        slide,
        [
            "Distribution de Zipf : peu de formes très fréquentes, longue traîne de formes rares.",
            "95 % des occurrences couvertes par ≈ 3 700 formes EN / 7 000 formes FR.",
            "Sert au dimensionnement du vocabulaire (config words95, diapositive 7).",
        ],
        Inches(0.6), Inches(1.5), Inches(6.4), Inches(5.4),
    )
    add_picture_fit(slide, figures["zipf"], Inches(7.3), Inches(1.5), Inches(5.4), Inches(4.6))
    add_caption(
        slide,
        "Courbe de Zipf (rang / fréquence, log-log), FR et EN",
        Inches(7.3), Inches(6.15), Inches(5.4),
    )
    return slide


# Géométrie commune aux diapositives « une seule figure pleine largeur »
# (hapax, Zipf BPE/Unigram, longueur des tokens, courbes d'entraînement) :
# la figure occupe toute la largeur utile, avec au plus 2 puces en dessous.
_FULL_WIDTH_IMG_LEFT = MARGIN
_FULL_WIDTH_IMG_TOP = Inches(1.25)
_FULL_WIDTH_IMG_MAX_W = SLIDE_W - 2 * MARGIN
_FULL_WIDTH_IMG_MAX_H = Inches(4.5)
_FULL_WIDTH_CAPTION_TOP = Inches(5.85)
_FULL_WIDTH_BULLETS_TOP = Inches(6.35)
_FULL_WIDTH_BULLETS_H = Inches(1.1)


def slide_05_hapax(prs, figures):
    slide = add_blank_slide(prs)
    add_title(slide, "Hapax et richesse lexicale")
    add_picture_fit(
        slide, figures["hapax"],
        _FULL_WIDTH_IMG_LEFT, _FULL_WIDTH_IMG_TOP, _FULL_WIDTH_IMG_MAX_W, _FULL_WIDTH_IMG_MAX_H,
    )
    add_caption(
        slide,
        "Hapax legomena et type-token ratio (TTR)",
        _FULL_WIDTH_IMG_LEFT, _FULL_WIDTH_CAPTION_TOP, _FULL_WIDTH_IMG_MAX_W,
    )
    add_bullets(
        slide,
        [
            "Les hapax (formes vues une seule fois) : 38 % du vocabulaire pour ~0,6 % de couverture — coût du dernier pourcent.",
            "C'est cet argument qui justifie la coupe à 95 % de couverture (vocabulaire words95, diapositive 7).",
        ],
        _FULL_WIDTH_IMG_LEFT, _FULL_WIDTH_BULLETS_TOP, _FULL_WIDTH_IMG_MAX_W, _FULL_WIDTH_BULLETS_H,
    )
    return slide


def slide_06_pipeline(prs, figures):
    slide = add_blank_slide(prs)
    add_title(slide, "Pipeline de données et anti-fuite")
    add_bullets(
        slide,
        [
            "Nettoyage → normalisation → dédoublonnage → split groupé par phrase cible EN.",
            "264 905 brutes → 625 filtrées → 264 280 nettoyées → 1 402 doublons exacts retirés (après normalisation) → 262 878 paires finales.",
            "Split 70/15/15 (seed 42) : train 183 994 / val 39 367 / test 39 517.",
            "Non-fuite = 0 sur les trois paires (vérifiée sur cibles EN normalisées).",
            "Pourquoi grouper par cible EN : 25 % des phrases EN sont dupliquées (alignements 1→N légitimes) — sans regroupement, une même cible se retrouverait dans plusieurs splits.",
        ],
        Inches(0.6), Inches(1.5), Inches(6.4), Inches(5.5),
    )
    add_picture_fit(slide, figures["ratio"], Inches(7.3), Inches(1.5), Inches(5.4), Inches(4.6))
    add_caption(
        slide,
        "Ratio de longueur FR/EN par paire",
        Inches(7.3), Inches(6.15), Inches(5.4),
    )
    return slide


def slide_07_tokenisation_configs(prs, figures):
    slide = add_blank_slide(prs)
    add_title(slide, "Tokenisation : 6 configurations")
    add_table(
        slide,
        ["Config", "Type", "Taille vocab.", "Régime"],
        [
            ["full", "mots entiers", "35 282 / 23 365", "séparé"],
            ["words95", "mots entiers", "6 753 / 3 520", "séparé"],
            ["bpe4k", "BPE", "4 000", "conjoint"],
            ["bpe8k", "BPE", "8 000", "conjoint"],
            ["unigram4k", "Unigram", "4 000", "conjoint"],
            ["unigram8k", "Unigram", "8 000", "conjoint"],
        ],
        Inches(0.5), Inches(1.3), Inches(4.5), Inches(3.6),
    )
    # Figure agrandie : >= 55 % de la largeur de la diapo (13,333 in).
    add_picture_fit(slide, figures["couverture_seuils"], Inches(5.4), Inches(1.3), Inches(7.4), Inches(3.4))
    add_caption(
        slide,
        "Couverture cumulée (seuils 95 %)",
        Inches(5.4), Inches(4.7), Inches(7.4),
    )
    add_bullets(
        slide,
        [
            "Taille vocab. : FR/EN si séparé, total si conjoint.",
            "Conjoint : un seul modèle SentencePiece FR+EN — cognats mutualisés (Jaccard 54,0 % BPE / 39,9 % Unigram à 4k).",
            "Séparé : vocabulaires asymétriques (ex. words95 : 6 753 FR vs 3 520 EN).",
            "Encodeur = FR seul, décodeur = EN seul → pas de risque d'homographes inter-langues (Sennrich et al., 2016).",
        ],
        Inches(0.5), Inches(5.2), Inches(12.3), Inches(2.0),
    )
    return slide


def slide_08_bpe_vs_unigram_zipf(prs, figures):
    slide = add_blank_slide(prs)
    add_title(slide, "BPE vs Unigram — distribution de Zipf")
    add_picture_fit(
        slide, figures["zipf_bpe"],
        _FULL_WIDTH_IMG_LEFT, _FULL_WIDTH_IMG_TOP, _FULL_WIDTH_IMG_MAX_W, _FULL_WIDTH_IMG_MAX_H,
    )
    add_caption(
        slide,
        "Zipf superposé (BPE vs Unigram)",
        _FULL_WIDTH_IMG_LEFT, _FULL_WIDTH_CAPTION_TOP, _FULL_WIDTH_IMG_MAX_W,
    )
    add_bullets(
        slide,
        [
            "BPE fusionne les paires fréquentes (bottom-up) ; Unigram élague un grand vocabulaire (top-down) — les deux convergent vers la même Zipf.",
            "Exploration, pas comparaison contrôlée (un seul run par config).",
        ],
        _FULL_WIDTH_IMG_LEFT, _FULL_WIDTH_BULLETS_TOP, _FULL_WIDTH_IMG_MAX_W, _FULL_WIDTH_BULLETS_H,
    )
    return slide


def slide_09_bpe_vs_unigram_longueur(prs, figures):
    slide = add_blank_slide(prs)
    add_title(slide, "BPE vs Unigram — longueur des tokens")
    add_picture_fit(
        slide, figures["longueur_tokens"],
        _FULL_WIDTH_IMG_LEFT, _FULL_WIDTH_IMG_TOP, _FULL_WIDTH_IMG_MAX_W, _FULL_WIDTH_IMG_MAX_H,
    )
    add_caption(
        slide,
        "Longueur des tokens (caractères), FR et EN",
        _FULL_WIDTH_IMG_LEFT, _FULL_WIDTH_CAPTION_TOP, _FULL_WIDTH_IMG_MAX_W,
    )
    add_bullets(
        slide,
        [
            "C'est sur la longueur des tokens, pas sur Zipf, que l'écart entre BPE et Unigram se lit vraiment.",
            "BPE partage davantage de tokens entre FR et EN que Unigram (Jaccard 54,0 % vs 39,9 % à 4k).",
        ],
        _FULL_WIDTH_IMG_LEFT, _FULL_WIDTH_BULLETS_TOP, _FULL_WIDTH_IMG_MAX_W, _FULL_WIDTH_BULLETS_H,
    )
    return slide


def slide_10_encodeur_decodeur(prs):
    slide = add_blank_slide(prs)
    add_title(slide, "Encodeur-décodeur et goulot d'étranglement")
    add_arrow_flow(
        slide,
        ["Embedding", "Cellule récurrente\n(RNN / GRU)", "Vecteur de contexte\nunique", "Décodeur → sortie"],
        Inches(0.6), Inches(1.5), Inches(12.1), Inches(1.3),
    )
    add_bullets(
        slide,
        [
            "Sans attention : toute l'information de la phrase source est compressée dans un unique vecteur de contexte — le goulot d'étranglement (Cho et al., 2014).",
            "Avec attention (Bahdanau, Cho & Bengio, 2015) : à chaque pas, le décodeur consulte l'ensemble des états de l'encodeur, pondérés dynamiquement.",
            "Nuance : ici, l'attention est isolée sur un encodeur unidirectionnel identique à la version sans attention — protocole plus restreint que celui de Bahdanau et al. (encodeur bidirectionnel dans l'article original).",
        ],
        Inches(0.6), Inches(3.2), Inches(12.1), Inches(3.8),
    )
    return slide


def slide_11_teacher_forcing(prs):
    slide = add_blank_slide(prs)
    add_title(slide, "Teacher forcing et scheduled sampling")
    add_bullets(
        slide,
        [
            "En teacher forcing (l'entrée du décodeur à t+1 est le mot cible réel), l'accuracy mesurée pendant l'entraînement est gonflée : le modèle ne voit jamais ses erreurs s'accumuler.",
            "Biais d'exposition : en génération libre, le modèle doit se recopier lui-même — une erreur précoce peut se propager.",
            "Scheduled sampling : probabilité de teacher forcing décroissante au fil des epochs, ratio(e) = decay^(e-1).",
            "decay cherché dans [0,7 ; 1,0] ; decay = 1,0 = témoin sans décroissance (teacher forcing constant).",
            "Meilleurs essais observés à decay ≈ 0,96–0,98.",
        ],
        Inches(0.6), Inches(1.5), Inches(12.1), Inches(5.5),
    )
    return slide


def slide_12_evaluation(prs, figures):
    slide = add_blank_slide(prs)
    add_title(slide, "Évaluation : génération libre et deux métriques")
    add_bullets(
        slide,
        [
            "Génération libre uniquement (jamais teacher forcing) — seule condition comparable à un usage réel.",
            "Deux métriques : SacreBLEU (lowercase, corpus déjà en minuscules) et BERTScore (backbone distilbert, 3 000 paires).",
            "Plancher de référence : sortie constante « i do n't know » = 0,99 BLEU.",
            "Bins par longueur : courte ≤ 18 mots, longue > 18 mots.",
        ],
        Inches(0.6), Inches(1.5), Inches(4.6), Inches(5.5),
    )
    # Figure agrandie : >= 55 % de la largeur de la diapo (13,333 in).
    add_picture_fit(slide, figures["dispersion"], Inches(5.4), Inches(1.5), Inches(7.4), Inches(4.0))
    add_caption(
        slide,
        "Dispersion des longueurs (mots) par langue — base des bins",
        Inches(5.4), Inches(5.5), Inches(7.4),
    )
    return slide


def slide_13_optuna(prs, figures):
    slide = add_blank_slide(prs)
    add_title(slide, "Recherche d'hyperparamètres (Optuna)")
    add_bullets(
        slide,
        [
            "Espace de recherche : tokeniseur, taux d'apprentissage (1e-4 → 5e-3), taille cachée, taille d'embedding, dropout, decay.",
            "Objectif = SacreBLEU en génération libre, pas la loss (non comparable entre tokenisations : 4 000 classes vs 23 365).",
            "MedianPruner ; 60 essais par étude, 12 epochs par essai ; sous-échantillon stratifié 40 % du train ; une étude par architecture, reprise sur interruption.",
        ],
        Inches(0.6), Inches(1.5), Inches(7.0), Inches(5.6),
    )
    add_picture_fit(slide, figures["importance_gru"], Inches(7.9), Inches(1.6), Inches(4.8), Inches(3.4))
    add_caption(
        slide,
        "Importance des hyperparamètres — étude GRU (étude 7, complète)",
        Inches(7.9), Inches(5.1), Inches(4.8),
    )
    return slide


def slide_14_resultat_central(prs):
    slide = add_blank_slide(prs)
    add_title(slide, "Résultat central : RNN vs GRU")
    bullets = [
        "Test complet, bin courte (39 092 phrases) : RNN 9,74 vs GRU 31,64 — facteur ×3,2.",
        "Bin longue (425 phrases) : RNN 0,70 vs GRU 5,53.",
        "Les deux architectures apprennent : nettement au-dessus du plancher de la sortie constante (≈ 0,99).",
        "RNN : lr 2,17e-4, 50 epochs, non convergé — meilleur BLEU val 9,91. Convergence lente et plafond bas, caractéristiques d'un RNN sans portes.",
        "Effondrement de mode observé sur les premiers essais de recherche (RNN) : trois phrases FR différentes → même sortie « i 'll try to help you » :",
        (MODE_COLLAPSE_EXAMPLES[0], 1),
        (MODE_COLLAPSE_EXAMPLES[1], 1),
        (MODE_COLLAPSE_EXAMPLES[2], 1),
        "Explication : disparition du gradient sur les séquences longues, limitée par les portes du GRU (Cho et al., 2014).",
    ]
    add_bullets(slide, bullets, Inches(0.6), Inches(1.4), Inches(12.1), Inches(5.9))
    return slide


def slide_15_entrainement_final(prs):
    slide = add_blank_slide(prs)
    add_title(slide, "Entraînement final : évaluation par bin de longueur")
    add_table(
        slide,
        ["Run", "Bin", "n", "SacreBLEU", "BERTScore F1"],
        [
            ["rnn", "courte", "39 092", "9,74", "0,786"],
            ["rnn", "longue", "425", "0,70", "0,682"],
            ["gru", "courte", "39 092", "31,64", "0,882"],
            ["gru", "longue", "425", "5,53", "0,777"],
            ["gru_attention", "courte", "39 092", "34,10", "0,888"],
            ["gru_attention", "longue", "425", "7,60", "0,782"],
        ],
        Inches(0.5), Inches(1.4), Inches(12.3), Inches(3.6),
    )
    add_bullets(
        slide,
        [
            "Le bin long (> 18 mots, 425 paires) est nettement plus dur que le bin court, pour les trois architectures.",
            "Le RNN apprend mais reste nettement sous le GRU sur les deux bins (facteur ×3,2 sur les courtes).",
        ],
        Inches(0.5), Inches(5.3), Inches(12.3), Inches(1.9),
    )
    return slide


def slide_16_attention_bahdanau(prs):
    slide = add_blank_slide(prs)
    add_title(slide, "L'attention de Bahdanau : avec ou sans, par bin")
    # Plus de figure sur cette diapo (courbes déplacées diapositive 20) :
    # table et puces recentrées.
    add_table(
        slide,
        ["Bin", "Sans attention", "Avec attention", "Gain relatif"],
        [
            ["courte (≤ 18 mots)", "31,64", "34,10", "+8 %"],
            ["longue (> 18 mots)", "5,53", "7,60", "+37 %"],
        ],
        Inches(1.67), Inches(1.7), Inches(10.0), Inches(1.6),
    )
    add_bullets(
        slide,
        [
            "Gain relatif 3× plus fort sur le bin long : prédiction de Bahdanau et al. (2015) vérifiée.",
            "Comparaison propre : mêmes hyperparamètres, même tokeniseur, 11 epochs — seule variable, la présence de l'attention.",
        ],
        Inches(0.9), Inches(3.9), Inches(11.5), Inches(2.6),
    )
    return slide


def slide_17_trois_metriques(prs):
    slide = add_blank_slide(prs)
    add_title(slide, "Quand BLEU et BERTScore ne sont pas d'accord")
    add_bullets(
        slide,
        [
            "BERTScore reste compressé entre 0,68 et 0,89 quand SacreBLEU varie de 0,7 à 34 — anisotropie de l'espace d'embeddings (Zhang et al., 2020).",
        ],
        Inches(0.6), Inches(1.35), Inches(12.1), Inches(1.0),
    )
    add_table(
        slide,
        ["FR (source)", "Hypothèse", "Référence", "SacreBLEU", "BERTScore F1"],
        [
            [src, hyp, ref, f"{bleu:.2f}".replace(".", ","), f"{f1:.3f}".replace(".", ",")]
            for src, hyp, ref, bleu, f1 in DISAGREEMENT_EXAMPLES
        ],
        Inches(0.6), Inches(2.35), Inches(12.1), Inches(2.6),
    )
    add_bullets(
        slide,
        [
            "BLEU compte des n-grammes exacts ; BERTScore compare des embeddings contextuels — une reformulation ou un synonyme valide est pénalisé par BLEU mais pas par BERTScore.",
            "Exemples calculés sur les 3 000 premières paires du test, run gru (v4).",
        ],
        Inches(0.6), Inches(5.2), Inches(12.1), Inches(1.8),
    )
    return slide


def slide_18_limites(prs):
    slide = add_blank_slide(prs)
    add_title(slide, "Limites et honnêteté")
    add_bullets(
        slide,
        [
            "Corpus relativement court (phrases Tatoeba), pas représentatif de tous les registres.",
            "Hyperparamètres réglés sur un sous-échantillon stratifié à 20 % du train (pas le corpus complet).",
            "SacreBLEU non strictement comparable aux valeurs publiées : corpus en minuscules, clitiques séparés par le tokeniseur.",
            "Recherche RNN fragile : 6 essais complets sur 60 (pruning sévère, convergence lente).",
            "Scheduled sampling : optima proches de 1,0 (0,84 / 0,94) — effet non concluant sur ce corpus.",
            "Exploration des 6 configurations de tokenisation, pas une comparaison contrôlée (pas de recherche d'hyperparamètres dédiée à chacune).",
        ],
        Inches(0.6), Inches(1.5), Inches(12.1), Inches(5.7),
    )
    return slide


def slide_19_conclusion(prs):
    slide = add_blank_slide(prs)
    add_title(slide, "Conclusion")
    add_bullets(
        slide,
        [
            "Le GRU et le RNN simple apprennent tous deux, mais le GRU l'emporte nettement (×3,2 sur les phrases courtes) ; le RNN n'a pas convergé en 50 epochs — plafond bas d'un modèle sans portes.",
            "L'attention de Bahdanau améliore la traduction sur les deux bins, avec un gain relatif 3× plus fort sur les phrases longues (+37 % vs +8 % sur les courtes) — la prédiction du goulot d'étranglement est vérifiée.",
            "Les vocabulaires sous-mots de petite taille (BPE/Unigram) surpassent nettement les vocabulaires par mots entiers ; Optuna a retenu bpe4340 (RNN) et unigram6539 (GRU).",
            "Le bin long (phrases > 18 mots) reste nettement plus dur pour les trois architectures, cohérent avec un corpus de phrases courtes.",
        ],
        Inches(0.6), Inches(1.6), Inches(12.1), Inches(5.4),
    )
    return slide


def slide_20_courbes_entrainement(prs, figures):
    slide = add_blank_slide(prs)
    add_title(slide, "Courbes d'entraînement (v4)")
    add_picture_fit(
        slide, figures["courbes"],
        _FULL_WIDTH_IMG_LEFT, _FULL_WIDTH_IMG_TOP, _FULL_WIDTH_IMG_MAX_W, _FULL_WIDTH_IMG_MAX_H,
    )
    add_caption(
        slide,
        "Courbes d'entraînement (loss / perplexité) des trois architectures",
        _FULL_WIDTH_IMG_LEFT, _FULL_WIDTH_CAPTION_TOP, _FULL_WIDTH_IMG_MAX_W,
    )
    add_bullets(
        slide,
        [
            "Le GRU converge en une dizaine d'epochs (meilleur essai retenu : 11 epochs).",
            "Le RNN n'a pas convergé en 50 epochs — plafond bas, cohérent avec un modèle sans portes.",
        ],
        _FULL_WIDTH_IMG_LEFT, _FULL_WIDTH_BULLETS_TOP, _FULL_WIDTH_IMG_MAX_W, _FULL_WIDTH_BULLETS_H,
    )
    return slide


def slide_21_hyperparametres(prs):
    slide = add_blank_slide(prs)
    add_title(slide, "Hyperparamètres retenus (v4)")
    rows = [
        ["rnn", "bpe4340", "512", "128", "0,05", "0,84", "2,17e-4", "50 (non convergé)"],
        ["gru", "unigram6539", "1024", "256", "0,25", "0,94", "9,53e-4", "11"],
        ["gru_attention", "unigram6539", "1024", "256", "0,25", "0,94", "9,53e-4", "11"],
    ]
    add_table(
        slide,
        ["run", "tokeniseur", "hidden", "emb", "dropout", "decay", "lr", "epochs"],
        rows,
        Inches(0.5), Inches(1.4), Inches(12.3), Inches(2.6),
    )
    add_bullets(
        slide,
        [
            "Optuna : 60 essais par étude ; études complètes 6 (rnn, best 3,69) et 7 (gru, best 25,49).",
            "gru_attention reprend exactement les hyperparamètres retenus pour gru — seule variable : la présence de l'attention.",
        ],
        Inches(0.5), Inches(4.4), Inches(12.3), Inches(2.5),
    )
    return slide


def slide_22_references(prs):
    slide = add_blank_slide(prs)
    add_title(slide, "Références")
    bullets = [(ref, 0) for ref in REFERENCES]
    add_bullets(slide, bullets, Inches(0.6), Inches(1.4), Inches(12.1), Inches(5.9), size=Pt(18))
    return slide


# --------------------------------------------------------------------------- #
# Construction du diaporama
# --------------------------------------------------------------------------- #


def build_figure_map(figures_dir: Path) -> dict[str, Path]:
    return {
        "longueurs": figures_dir / "aed_26_distribution_longueurs.png",
        "dispersion": figures_dir / "aed_27_dispersion_longueurs.png",
        "ratio": figures_dir / "aed_28_ratio_longueur.png",
        "zipf": figures_dir / "aed_35_zipf.png",
        "couverture_oov": figures_dir / "aed_37_couverture_oov.png",
        "couverture_seuils": figures_dir / "aed_39_couverture_seuils.png",
        "hapax": figures_dir / "aed_41_hapax_ttr.png",
        "croissance": figures_dir / "aed_42_croissance_vocabulaire.png",
        "zipf_bpe": figures_dir / "fig_zipf_bpe_vs_unigram.png",
        "longueur_tokens": figures_dir / "fig_longueur_tokens_caracteres.png",
        "fragmentation": figures_dir / "fig_fragmentation.png",
        "courbes": figures_dir / "courbes_entrainement.png",
        "importance_gru": figures_dir / "fig_importance_gru.png",
    }


def build_presentation(figures_dir: Path) -> Presentation:
    figures = build_figure_map(figures_dir)
    for name, path in figures.items():
        if not path.exists():
            raise FileNotFoundError(f"figure manquante ({name}) : {path}")

    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    slide_01_titre(prs)
    slide_02_objectif(prs, figures)
    slide_03_corpus(prs, figures)
    slide_04_zipf_couverture(prs, figures)
    slide_05_hapax(prs, figures)
    slide_06_pipeline(prs, figures)
    slide_07_tokenisation_configs(prs, figures)
    slide_08_bpe_vs_unigram_zipf(prs, figures)
    slide_09_bpe_vs_unigram_longueur(prs, figures)
    slide_10_encodeur_decodeur(prs)
    slide_11_teacher_forcing(prs)
    slide_12_evaluation(prs, figures)
    slide_13_optuna(prs, figures)
    slide_14_resultat_central(prs)
    slide_15_entrainement_final(prs)
    slide_16_attention_bahdanau(prs)
    slide_17_trois_metriques(prs)
    slide_18_limites(prs)
    slide_19_conclusion(prs)
    slide_20_courbes_entrainement(prs, figures)
    slide_21_hyperparametres(prs)
    slide_22_references(prs)

    return prs


# --------------------------------------------------------------------------- #
# Entrée principale
# --------------------------------------------------------------------------- #


def main() -> None:
    print(f"[1/3] Extraction des figures AED depuis {NOTEBOOK_AED} ...")
    written = extract_aed_figures(NOTEBOOK_AED, FIGURES_DIR)
    for cell_index, path in sorted(written.items()):
        print(f"       cellule {cell_index} -> {path.relative_to(ROOT)}")

    print("[2/3] Copie des figures déjà produites (tokenisation, entraînement, Optuna) ...")
    copied = copy_reused_figures(FIGURES_DIR)
    for name, path in copied.items():
        print(f"       {name} -> {path.relative_to(ROOT)}")

    print("[3/3] Construction du diaporama ...")
    prs = build_presentation(FIGURES_DIR)
    OUTPUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPUT_PPTX))
    print(f"       {len(prs.slides)} diapositives écrites -> {OUTPUT_PPTX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
