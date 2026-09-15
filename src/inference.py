"""Chargement des modèles entraînés et traduction d'une phrase française."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import torch
from torch import nn

from src.data.text import normalize_text
from src.models.seq2seq import build_model
from src.tokenization.vocab_builder import SentencePieceVocabulary


@dataclass(frozen=True)
class ModelSpec:
    """Paramètres d'architecture qui ne figurent pas dans ``state.json``."""

    cell_type: str
    use_attention: bool


MODEL_SPECS = {
    "rnn": ModelSpec(cell_type="rnn", use_attention=False),
    "gru": ModelSpec(cell_type="gru", use_attention=False),
    "gru_attention": ModelSpec(cell_type="gru", use_attention=True),
}


@dataclass
class LoadedModel:
    model: nn.Module
    vocab: SentencePieceVocabulary


class TranslationService:
    """Charge paresseusement les trois modèles et les garde en mémoire."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        max_output_tokens: int = 80,
    ) -> None:
        self.project_root = (
            Path(project_root)
            if project_root is not None
            else Path(__file__).resolve().parents[1]
        )
        self.device = torch.device("cpu")
        self.max_output_tokens = max_output_tokens
        self._models: dict[str, LoadedModel] = {}
        self._load_lock = Lock()

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(MODEL_SPECS)

    def _load_model(self, model_name: str) -> LoadedModel:
        if model_name not in MODEL_SPECS:
            raise ValueError(f"modèle inconnu : {model_name!r}")

        if model_name in self._models:
            return self._models[model_name]

        with self._load_lock:
            if model_name in self._models:
                return self._models[model_name]

            run_dir = self.project_root / "reports" / "runs" / model_name
            state_path = run_dir / "state.json"
            weights_path = run_dir / "best.pt"
            if not state_path.exists() or not weights_path.exists():
                raise FileNotFoundError(
                    f"artefacts du modèle '{model_name}' introuvables dans {run_dir}"
                )

            with open(state_path, encoding="utf-8") as file:
                state = json.load(file)
            params = state.get("params")
            if not isinstance(params, dict):
                raise TypeError(f"paramètres absents ou invalides dans {state_path}")

            config_name = str(params["tokenizationConfig"])
            tokenizer_path = (
                self.project_root
                / "data"
                / "processed"
                / "tokenizers"
                / f"{config_name}_joint.model"
            )
            if not tokenizer_path.exists():
                raise FileNotFoundError(
                    f"tokeniseur du modèle '{model_name}' introuvable : {tokenizer_path}"
                )

            vocab = SentencePieceVocabulary(config_name, "joint", tokenizer_path)
            spec = MODEL_SPECS[model_name]
            model = build_model(
                len(vocab),
                len(vocab),
                int(params["embDim"]),
                int(params["hiddenDim"]),
                spec.cell_type,
                float(params["dropout"]),
                use_attention=spec.use_attention,
            )
            state_dict = torch.load(
                weights_path,
                map_location=self.device,
                weights_only=False,
            )
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            loaded_model = LoadedModel(model=model, vocab=vocab)
            self._models[model_name] = loaded_model
            return loaded_model

    def translate(self, text: str, model_name: str) -> str:
        """Traduit une phrase par décodage glouton, sans teacher forcing."""

        normalized_text = normalize_text(text, "fr")
        if not normalized_text:
            raise ValueError("la phrase à traduire est vide")

        loaded = self._load_model(model_name)
        vocab = loaded.vocab
        source_ids = [vocab.sos_id, *vocab.encode(normalized_text), vocab.eos_id]
        source = torch.tensor(
            [source_ids],
            dtype=torch.long,
            device=self.device,
        )

        generated_ids: list[int] = []
        with torch.no_grad():
            encoder_outputs, hidden = loaded.model.encoder(source)
            input_token = torch.tensor(
                [[vocab.sos_id]],
                dtype=torch.long,
                device=self.device,
            )
            for _ in range(self.max_output_tokens):
                prediction, hidden = loaded.model.decoder.forward_step(
                    input_token,
                    hidden,
                    encoder_outputs,
                )
                token_id = int(prediction.argmax(dim=1).item())
                if token_id == vocab.eos_id:
                    break
                generated_ids.append(token_id)
                input_token = torch.tensor(
                    [[token_id]],
                    dtype=torch.long,
                    device=self.device,
                )

        return vocab.decode(generated_ids)
