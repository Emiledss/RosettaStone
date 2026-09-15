"""Tests légers de l'interface Flask, sans charger les poids PyTorch."""

from __future__ import annotations

from app import create_app


class FakeTranslationService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def translate(self, text: str, model_name: str) -> str:
        self.calls.append((text, model_name))
        return "hello world"


def test_page_affiche_les_trois_modeles() -> None:
    app = create_app(FakeTranslationService())
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"RNN" in response.data
    assert b"GRU" in response.data
    assert b"GRU + attention" in response.data


def test_formulaire_lance_la_traduction_avec_le_modele_choisi() -> None:
    service = FakeTranslationService()
    app = create_app(service)
    client = app.test_client()

    response = client.post(
        "/",
        data={"sourceText": "Bonjour le monde", "modelName": "gru_attention"},
    )

    assert response.status_code == 200
    assert service.calls == [("Bonjour le monde", "gru_attention")]
    assert b"hello world" in response.data


def test_phrase_vide_affiche_une_erreur_sans_appeler_le_service() -> None:
    service = FakeTranslationService()
    app = create_app(service)
    client = app.test_client()

    response = client.post(
        "/",
        data={"sourceText": "   ", "modelName": "gru"},
    )

    assert response.status_code == 200
    assert service.calls == []
    assert "Saisissez une phrase française.".encode() in response.data


def test_modele_invalide_est_refuse() -> None:
    service = FakeTranslationService()
    app = create_app(service)
    client = app.test_client()

    response = client.post(
        "/",
        data={"sourceText": "Bonjour", "modelName": "inconnu"},
    )

    assert response.status_code == 200
    assert service.calls == []
    assert b"mod\xc3\xa8le s\xc3\xa9lectionn\xc3\xa9 est invalide" in response.data
