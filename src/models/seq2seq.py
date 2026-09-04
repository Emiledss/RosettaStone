"""Ossature encodeur-décodeur seq2seq (étape 6) : RNN|GRU paramétrable, teacher
forcing intégré dès le départ, attention additive de Bahdanau greffable sans
refonte de l'encodeur ni de la boucle principale.

Contrat imposé par `notebooks/Rosetta_Optuna.ipynb` (`makeObjective`) :

- `build_model(fr_vocab_size, en_vocab_size, emb_dim, hidden_dim, cell_type,
  dropout, use_attention=False)` -- appelé POSITIONNELLEMENT dans cet ordre ;
  `use_attention` reste un mot-clé à défaut.
- `Seq2Seq.forward(src, tgt, teacher_forcing_ratio)` renvoie
  `[batch, seq_len, vocab_size]`, **batch-first** : la boucle d'entraînement
  (étape 7) fait `output[:, 1:, :].reshape(-1, outputDim)` contre
  `tgt[:, 1:].reshape(-1)`.
"""

from __future__ import annotations

import torch
from torch import nn

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
    """

    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, cell_type: str) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.rnn = _resolve_cell(cell_type)(emb_dim, hidden_dim, batch_first=True)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """src: (batch, src_len) -> outputs: (batch, src_len, hidden_dim),
        hidden: (num_layers, batch, hidden_dim)."""
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
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
        self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """decoder_hidden: (batch, hidden_dim) -- état caché courant du décodeur
        (couche du haut). encoder_outputs: (batch, src_len, hidden_dim).

        Renvoie `(context, weights)` : `context` (batch, hidden_dim), `weights`
        (batch, src_len) -- poids d'alignement, somme 1 sur `src_len`.
        """
        scores = self.attn_score(
            torch.tanh(
                self.attn_decoder(decoder_hidden).unsqueeze(1) + self.attn_encoder(encoder_outputs)
            )
        ).squeeze(-1)  # (batch, src_len)
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
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        cell_type: str,
        dropout: float,
        attention: BahdanauAttention | None = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        rnn_input_dim = emb_dim + hidden_dim if attention is not None else emb_dim
        self.rnn = _resolve_cell(cell_type)(rnn_input_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward_step(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """input_token: (batch, 1). Renvoie `(prediction, hidden)` avec
        `prediction`: (batch, vocab_size)."""
        embedded = self.dropout(self.embedding(input_token))  # (batch, 1, emb_dim)
        if self.attention is not None:
            if encoder_outputs is None:
                raise ValueError("encoder_outputs requis quand l'attention est activée")
            # état caché de la couche du haut : hidden[-1] -> (batch, hidden_dim)
            context, _weights = self.attention(hidden[-1], encoder_outputs)
            rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        else:
            rnn_input = embedded
        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.out(output.squeeze(1))
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
    ) -> None:
        super().__init__()
        self.encoder = Encoder(fr_vocab_size, emb_dim, hidden_dim, cell_type)
        attention = BahdanauAttention(hidden_dim) if use_attention else None
        self.decoder = Decoder(
            en_vocab_size, emb_dim, hidden_dim, cell_type, dropout, attention=attention
        )
        self.en_vocab_size = en_vocab_size
        self.use_attention = use_attention

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

        outputs = torch.zeros(batch_size, tgt_len, self.en_vocab_size, device=device)
        input_token = tgt[:, 0:1]  # <sos>
        for t in range(1, tgt_len):
            prediction, hidden = self.decoder.forward_step(input_token, hidden, encoder_outputs)
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
) -> nn.Module:
    """Construit le modèle seq2seq (étape 6).

    Signature imposée par `makeObjective` du notebook Optuna -- appelée
    positionnellement dans cet ordre, `use_attention` restant un mot-clé à
    défaut.
    """
    return Seq2Seq(
        fr_vocab_size, en_vocab_size, emb_dim, hidden_dim, cell_type, dropout, use_attention=use_attention
    )
