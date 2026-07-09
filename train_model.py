import re

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from db import fetch_tweets

MODEL_PATH = "model/sentiment_model.joblib"
VECTORIZER_PATH = "model/vectorizer.joblib"
REPORTS_DIR = "reports"


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def to_label(row):
    if row["positive"] == 1:
        return 1
    if row["negative"] == 1:
        return -1
    return 0


def plot_confusion_matrix(y_true, y_pred, labels, display_labels, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/{filename}")
    plt.close()
    return cm


def main():
    rows = fetch_tweets()
    if not rows:
        raise RuntimeError("Aucun tweet trouvé dans la base de données.")

    texts = [clean_text(r["text"]) for r in rows]
    labels = [to_label(r) for r in rows]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Rapport de classification (labels: -1=négatif, 0=neutre, 1=positif) :")
    print(classification_report(y_test, y_pred, labels=[-1, 0, 1], zero_division=0))

    y_true_positive = [1 if y == 1 else 0 for y in y_test]
    y_pred_positive = [1 if y == 1 else 0 for y in y_pred]
    print("Rapport - classe positive :")
    print(classification_report(y_true_positive, y_pred_positive, zero_division=0))
    plot_confusion_matrix(
        y_true_positive,
        y_pred_positive,
        labels=[0, 1],
        display_labels=["non-positif", "positif"],
        title="Matrice de confusion - Positive",
        filename="confusion_matrix_positive.png",
    )

    y_true_negative = [1 if y == -1 else 0 for y in y_test]
    y_pred_negative = [1 if y == -1 else 0 for y in y_pred]
    print("Rapport - classe négative :")
    print(classification_report(y_true_negative, y_pred_negative, zero_division=0))
    plot_confusion_matrix(
        y_true_negative,
        y_pred_negative,
        labels=[0, 1],
        display_labels=["non-négatif", "négatif"],
        title="Matrice de confusion - Négative",
        filename="confusion_matrix_negative.png",
    )

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Modèle sauvegardé dans {MODEL_PATH}")
    print(f"Vectorizer sauvegardé dans {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
