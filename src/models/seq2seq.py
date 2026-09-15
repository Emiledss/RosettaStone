"""Ossature encodeur-décodeur seq2seq (étape 6) : RNN|GRU paramétrable, teacher
forcing intégré dès le départ, attention additive de Bahdanau greffable sans
refonte de l'encodeur ni de la boucle principale.

Contrat imposé par `notebooks/Rosetta_Modelisation.ipynb` (`makeObjective`) :

- `build_model(fr_vocab_size, en_vocab_size, emb_dim, hidden_dim, cell_type,
  dropout, use_attention=False, tie_embeddings=False)` -- les 6 premiers
  arguments sont appelés POSITIONNELLEMENT dans cet ordre ; `use_attention`
  et `tie_embeddings` restent des mots-clés à défaut.
- `Seq2Seq.forward(src, tgt, teacher_forcing_ratio)` renvoie
  `[batch, seq_len, vocab_size]`, **batch-first** : la boucle d'entraînement
  (étape 7) fait `output[:, 1:, :].reshape(-1, outputDim)` contre
  `tgt[:, 1:].reshape(-1)`.

`tie_embeddings=True` lie l'embedding de l'encodeur, l'embedding du décodeur
et la projection de sortie (une seule matrice partagée, comme MarianMT) --
exige `fr_vocab_size == en_vocab_size` (vocabulaire conjoint, voir
`src/tokenization/vocab_builder.py`). Quand `emb_dim != hidden_dim`, une
projection linéaire `hidden_dim -> emb_dim` est insérée avant de réutiliser
la matrice d'embedding transposée (Optuna tire `embDim`/`hiddenDim`
indépendamment, ils diffèrent la plupart du temps).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Indice du <pad> -- imposé par `src.tokenization.vocab_builder.Vocabulary.pad_id`.
PAD_ID = 0

_CELL_CLASSES: dict[str, type[nn.RNN | nn.GRU]] = {"rnn": nn.RNN, "gru": nn.GRU}


def _resolve_cell(cell_type: str) -> type[nn.RNN | nn.GRU]:
    try:
        return _CELL_CLASSES[cell_type]
    except KeyError as exc:
        raise ValueError(
            f"cell_type inconnu: {cell_type!r} (attendu: {sorted(_CELL_CLASSES)})"
        ) from exc


class Encoder(nn.Module):
    """Embedding -> cellule récurrente paramétrable (`nn.RNN` ou `nn.GRU`,
    bascule = 1 paramètre `cell_type`).

    Renvoie TOUS les états cachés (`outputs`, forme `(batch, seq_len,
    hidden_dim)`) -- pas seulement le dernier -- prérequis pour greffer
    l'attention plus tard sans refonte de cet encodeur.

    Les positions de `<pad>` sont exclues du calcul récurrent via
    `pack_padded_sequence`/`pad_packed_sequence` : sans ça, `hidden`
    est l'état après avoir "lu" du padding -- le contexte encodé est effacé
    sur des batchs paddés à des longueurs très supérieures à la phrase réelle
    (RNN sans portes en particulier, incapable d'y résister).
    """

    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, cell_type: str) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.rnn = _resolve_cell(cell_type)(emb_dim, hidden_dim, batch_first=True)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """src: (batch, src_len) -> outputs: (batch, src_len, hidden_dim),
        hidden: (num_layers, batch, hidden_dim). `hidden` est l'état à la
        VRAIE dernière position de chaque séquence (pas après le padding)."""
        embedded = self.embedding(src)
        pad_id = self.embedding.padding_idx
        # `pack_padded_sequence` exige des longueurs CPU int64 ; `enforce_sorted=False`
        # évite d'avoir à trier le batch par longueur décroissante en amont.
        lengths = (src != pad_id).sum(dim=1).clamp(min=1).to(device="cpu", dtype=torch.int64)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=src.size(1)
        )
        return outputs, hidden


class BahdanauAttention(nn.Module):
    """Attention additive de Bahdanau -- module d'alignement séparé, greffé sur
    le décodeur sans toucher à l'encodeur.

    À chaque pas du décodeur : scores entre l'état courant du décodeur et TOUS
    les états de l'encodeur -> softmax -> vecteur de contexte pondéré.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn_decoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_encoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """decoder_hidden: (batch, hidden_dim) -- état caché courant du décodeur
        (couche du haut). encoder_outputs: (batch, src_len, hidden_dim).
        `src_mask` (optionnel) : (batch, src_len), `True` sur les VRAIES
        positions -- les positions `False` (padding) reçoivent un score
        `-inf`, donc un poids exactement nul après softmax.

        Renvoie `(context, weights)` : `context` (batch, hidden_dim), `weights`
        (batch, src_len) -- poids d'alignement, somme 1 sur `src_len`.
        """
        scores = self.attn_score(
            torch.tanh(
                self.attn_decoder(decoder_hidden).unsqueeze(1) + self.attn_encoder(encoder_outputs)
            )
        ).squeeze(-1)  # (batch, src_len)
        if src_mask is not None:
            scores = scores.masked_fill(~src_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, hidden_dim)
        return context, weights


class Decoder(nn.Module):
    """Embedding -> cellule récurrente -> linéaire. Un pas de décodage à la
    fois, pour permettre le teacher forcing token par token dans
    `Seq2Seq.forward`.

    Si `attention` est fourni, le vecteur de contexte est concaténé à
    l'embedding d'entrée avant la cellule récurrente -- seule modification
    nécessaire pour greffer l'attention.

    Si `tied_embedding` est fourni (module d'embedding de l'encodeur, même
    vocabulaire), le décodeur **réutilise ce module** au lieu d'en créer un
    -- l'embedding encodeur/décodeur devient une seule et même matrice. La
    couche de sortie réutilise alors cette matrice transposée au lieu d'une
    `nn.Linear(hidden_dim, vocab_size)` indépendante ; comme `emb_dim` et
    `hidden_dim` diffèrent la plupart du temps (tirés indépendamment par
    Optuna), une projection `hidden_dim -> emb_dim` (`pre_output_proj`) est
    insérée avant cette réutilisation -- `nn.Identity()` si les deux
    dimensions coïncident déjà.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        cell_type: str,
        dropout: float,
        attention: BahdanauAttention | None = None,
        tied_embedding: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        self.embedding = (
            tied_embedding if tied_embedding is not None else nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        )
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        rnn_input_dim = emb_dim + hidden_dim if attention is not None else emb_dim
        self.rnn = _resolve_cell(cell_type)(rnn_input_dim, hidden_dim, batch_first=True)

        self.tie_embeddings = tied_embedding is not None
        if self.tie_embeddings:
            self.pre_output_proj: nn.Module = (
                nn.Identity() if hidden_dim == emb_dim else nn.Linear(hidden_dim, emb_dim, bias=False)
            )
            self.out_bias = nn.Parameter(torch.zeros(vocab_size))
            self.out: nn.Linear | None = None
        else:
            self.pre_output_proj = None  # type: ignore[assignment]
            self.out_bias = None
            self.out = nn.Linear(hidden_dim, vocab_size)

    def forward_step(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """input_token: (batch, 1). `src_mask` (optionnel, (batch, src_len),
        `True` sur les vraies positions) est transmis à l'attention -- défaut
        `None` pour ne pas casser les appelants existants (`generate.py`)
        qui ne le fournissent pas encore. Renvoie `(prediction, hidden)` avec
        `prediction`: (batch, vocab_size)."""
        embedded = self.dropout(self.embedding(input_token))  # (batch, 1, emb_dim)
        if self.attention is not None:
            if encoder_outputs is None:
                raise ValueError("encoder_outputs requis quand l'attention est activée")
            # état caché de la couche du haut : hidden[-1] -> (batch, hidden_dim)
            context, _weights = self.attention(hidden[-1], encoder_outputs, src_mask)
            rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        else:
            rnn_input = embedded
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(1)  # (batch, hidden_dim)
        if self.tie_embeddings:
            projected = self.pre_output_proj(output)  # (batch, emb_dim)
            prediction = nn.functional.linear(projected, self.embedding.weight, self.out_bias)
        else:
            prediction = self.out(output)
        return prediction, hidden


class Seq2Seq(nn.Module):
    """Ossature encodeur-décodeur avec `teacher_forcing_ratio` intégré et
    attention additive de Bahdanau greffable via `use_attention`.

    `teacher_forcing_ratio` : `1.0` = teacher forcing pur (toujours le vrai
    token précédent) ; `0.0` = génération libre (toujours la prédiction du
    modèle) ; valeur intermédiaire = scheduled sampling.
    """

    def __init__(
        self,
        fr_vocab_size: int,
        en_vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        cell_type: str,
        dropout: float,
        use_attention: bool = False,
        tie_embeddings: bool = False,
    ) -> None:
        super().__init__()
        if tie_embeddings and fr_vocab_size != en_vocab_size:
            raise ValueError(
                "tie_embeddings=True exige fr_vocab_size == en_vocab_size (vocabulaire "
                f"conjoint) -- reçu fr_vocab_size={fr_vocab_size}, en_vocab_size={en_vocab_size}. "
                "Les configs à vocabulaires séparés (full, words95) ne peuvent pas lier "
                "leurs embeddings."
            )
        self.encoder = Encoder(fr_vocab_size, emb_dim, hidden_dim, cell_type)
        attention = BahdanauAttention(hidden_dim) if use_attention else None
        tied_embedding = self.encoder.embedding if tie_embeddings else None
        self.decoder = Decoder(
            en_vocab_size,
            emb_dim,
            hidden_dim,
            cell_type,
            dropout,
            attention=attention,
            tied_embedding=tied_embedding,
        )
        self.en_vocab_size = en_vocab_size
        self.use_attention = use_attention
        self.tie_embeddings = tie_embeddings

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float
    ) -> torch.Tensor:
        """src: (batch, src_len), tgt: (batch, tgt_len) -- `tgt[:, 0]` = `<sos>`.

        Renvoie `[batch, tgt_len, en_vocab_size]` (batch-first). La position 0
        reste à zéro (jamais prédite, `tgt[:, 0]` sert d'entrée initiale) --
        c'est pour cela que la boucle d'entraînement calcule la loss sur
        `output[:, 1:, :]`.
        """
        batch_size, tgt_len = tgt.shape
        device = src.device
        encoder_outputs, hidden = self.encoder(src)
        # (batch, src_len), True sur les vraies positions -- même device que
        # `encoder_outputs`/`scores` pour l'attention.
        src_mask = (src != self.encoder.embedding.padding_idx).to(device)

        outputs = torch.zeros(batch_size, tgt_len, self.en_vocab_size, device=device)
        input_token = tgt[:, 0:1]  # <sos>
        for t in range(1, tgt_len):
            prediction, hidden = self.decoder.forward_step(input_token, hidden, encoder_outputs, src_mask)
            outputs[:, t, :] = prediction
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            top1 = prediction.argmax(1, keepdim=True)
            input_token = tgt[:, t : t + 1] if use_teacher_forcing else top1
        return outputs


def build_model(
    fr_vocab_size: int,
    en_vocab_size: int,
    emb_dim: int,
    hidden_dim: int,
    cell_type: str,
    dropout: float,
    use_attention: bool = False,
    tie_embeddings: bool = False,
) -> nn.Module:
    """Construit le modèle seq2seq (étape 6).

    Signature imposée par `makeObjective` du notebook Optuna -- les 6 premiers
    arguments sont appelés positionnellement dans cet ordre, `use_attention`
    et `tie_embeddings` restant des mots-clés à défaut.

    `tie_embeddings=True` lie embedding encodeur, embedding décodeur et
    projection de sortie (variante fixée par run, hors espace de recherche
    Optuna, cf. `useAttention`) -- lève `ValueError` si
    `fr_vocab_size != en_vocab_size` (voir `Seq2Seq.__init__`).
    """
    return Seq2Seq(
        fr_vocab_size,
        en_vocab_size,
        emb_dim,
        hidden_dim,
        cell_type,
        dropout,
        use_attention=use_attention,
        tie_embeddings=tie_embeddings,
    )
