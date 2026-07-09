import os

import joblib
from flask import Flask, jsonify, request, send_from_directory
from flask_swagger_ui import get_swaggerui_blueprint

from train_model import MODEL_DIR, clean_text

app = Flask(__name__)

SWAGGER_URL = "/docs"
API_SPEC_URL = "/static/openapi.yaml"

swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL, API_SPEC_URL, config={"app_name": "API d'Analyse de Sentiments"}
)
app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)


@app.route("/static/openapi.yaml")
def openapi_spec():
    return send_from_directory("static", "openapi.yaml")

model = joblib.load(os.path.join(MODEL_DIR, "sentiment_model.joblib"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.joblib"))


def compute_score(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    probabilities = model.predict_proba(vector)[0]
    class_to_proba = dict(zip(model.classes_, probabilities))
    score = class_to_proba.get(1, 0.0) - class_to_proba.get(-1, 0.0)
    return round(float(score), 4)


@app.route("/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(silent=True)

    if payload is None or "tweets" not in payload:
        return jsonify({"error": "Le corps de la requête doit contenir un champ 'tweets'."}), 400

    tweets = payload["tweets"]

    if not isinstance(tweets, list):
        return jsonify({"error": "'tweets' doit être une liste de chaînes de caractères."}), 400

    if len(tweets) == 0:
        return jsonify({"error": "La liste de tweets ne peut pas être vide."}), 400

    if not all(isinstance(tweet, str) for tweet in tweets):
        return jsonify({"error": "Chaque élément de 'tweets' doit être une chaîne de caractères."}), 400

    if any(tweet.strip() == "" for tweet in tweets):
        return jsonify({"error": "Les tweets ne peuvent pas être des chaînes vides."}), 400

    results = {tweet: compute_score(tweet) for tweet in tweets}
    return jsonify(results), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
