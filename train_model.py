import os
import re

import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from db import fetch_tweets

MODEL_DIR = "model"
REPORTS_DIR = "reports"


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def label_from_flags(positive, negative):
    if positive == 1:
        return 1
    if negative == 1:
        return -1
    return 0


def save_confusion_matrix(y_true, y_pred, target_label, display_labels, filename):
    y_true_binary = [1 if y == target_label else 0 for y in y_true]
    y_pred_binary = [1 if y == target_label else 0 for y in y_pred]

    print(classification_report(y_true_binary, y_pred_binary, target_names=display_labels))

    cm = confusion_matrix(y_true_binary, y_pred_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap="Blues")
    plt.title(f"Matrice de confusion - {display_labels[1]} vs reste")
    plt.savefig(os.path.join(REPORTS_DIR, filename))
    plt.close()


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    tweets = fetch_tweets()

    texts = [clean_text(t["text"]) for t in tweets]
    labels = [label_from_flags(t["positive"], t["negative"]) for t in tweets]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.25, random_state=42, stratify=labels
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("=== Rapport global (3 classes) ===")
    print(classification_report(y_test, y_pred))

    print("=== Rapport classe positive (positif vs reste) ===")
    save_confusion_matrix(
        y_test, y_pred, target_label=1,
        display_labels=["reste", "positif"],
        filename="confusion_matrix_positive.png",
    )

    print("=== Rapport classe negative (negatif vs reste) ===")
    save_confusion_matrix(
        y_test, y_pred, target_label=-1,
        display_labels=["reste", "negatif"],
        filename="confusion_matrix_negative.png",
    )

    joblib.dump(model, os.path.join(MODEL_DIR, "sentiment_model.joblib"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))

    print(f"Modele et vectorizer sauvegardes dans {MODEL_DIR}/")
    print(f"Matrices de confusion sauvegardees dans {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
