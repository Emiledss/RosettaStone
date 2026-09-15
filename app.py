"""Interface Flask minimale pour tester les trois modèles de traduction."""

from __future__ import annotations

from flask import Flask, render_template, request

from src.inference import TranslationService

MODEL_LABELS = {
    "rnn": "RNN",
    "gru": "GRU",
    "gru_attention": "GRU + attention",
}


def create_app(translationService: TranslationService | None = None) -> Flask:
    app = Flask(__name__)
    service = translationService or TranslationService()

    @app.route("/", methods=["GET", "POST"])
    def index():
        sourceText = ""
        selectedModel = "gru"
        translation = None
        error = None

        if request.method == "POST":
            sourceText = request.form.get("sourceText", "").strip()
            selectedModel = request.form.get("modelName", "gru")

            if selectedModel not in MODEL_LABELS:
                error = "Le modèle sélectionné est invalide."
            elif not sourceText:
                error = "Saisissez une phrase française."
            elif len(sourceText) > 1000:
                error = "La phrase est trop longue (1 000 caractères maximum)."
            else:
                try:
                    translation = service.translate(sourceText, selectedModel)
                except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
                    error = f"La traduction a échoué : {exc}"

        return render_template(
            "index.html",
            modelLabels=MODEL_LABELS,
            sourceText=sourceText,
            selectedModel=selectedModel,
            translation=translation,
            error=error,
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=False)
