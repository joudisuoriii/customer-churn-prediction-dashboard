# -*- coding: utf-8 -*-
# Bachelorarbeit - Evaluation des finalen XGBoost-Modells
# Berechnung der Kennzahlen sowie Export von Konfusionsmatrix und ROC-Kurve

import pickle

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# Datensatz laden und denselben Testsplit wiederherstellen
kunden_rohdaten = pd.read_csv("data/Churn_Modelling.csv", sep=";")
if kunden_rohdaten.shape[1] == 1:
    kunden_rohdaten = pd.read_csv("data/Churn_Modelling.csv", sep=",")

ziel_spalte = "Exited"
id_merkmale = ["RowNumber", "CustomerId", "Surname"]

X_daten = kunden_rohdaten.drop(columns=id_merkmale + [ziel_spalte])
y_labels = kunden_rohdaten[ziel_spalte]

_, X_test, _, y_test = train_test_split(
    X_daten,
    y_labels,
    test_size=0.30,
    random_state=42,
    stratify=y_labels
)

# Gespeichertes Modell und Transformer laden
with open("prototype/model/final_xgboost.pkl", "rb") as datei:
    eval_klassifikator = pickle.load(datei)

with open("prototype/model/final_transformer.pkl", "rb") as datei:
    eval_transformer = pickle.load(datei)

# Testdaten transformieren und Vorhersagen berechnen
test_daten_bereit = eval_transformer.transform(X_test)

vorhersage_labels = eval_klassifikator.predict(test_daten_bereit)
abwanderung_scores = eval_klassifikator.predict_proba(
    test_daten_bereit
)[:, 1]

# Kennzahlen
wert_accuracy = accuracy_score(y_test, vorhersage_labels)
wert_precision = precision_score(
    y_test,
    vorhersage_labels,
    zero_division=0
)
wert_recall = recall_score(
    y_test,
    vorhersage_labels,
    zero_division=0
)
wert_f1 = f1_score(
    y_test,
    vorhersage_labels,
    zero_division=0
)
wert_auc = roc_auc_score(y_test, abwanderung_scores)

print("\n--- ERGEBNISSE DER MODELL-EVALUATION ---")
print(f"Accuracy-Wert :  {wert_accuracy:.4f}")
print(f"Precision-Wert:  {wert_precision:.4f}")
print(f"Recall-Wert   :  {wert_recall:.4f}")
print(f"F1-Score      :  {wert_f1:.4f}")
print(f"ROC-AUC       :  {wert_auc:.4f}")

print("\n--- KLASSIFIKATIONSBERICHT ---")
print(classification_report(
    y_test,
    vorhersage_labels,
    target_names=["Verblieben", "Abgewandert"],
    zero_division=0
))

print("\n--- KONFUSIONSMATRIX ---")
berechnete_matrix = confusion_matrix(y_test, vorhersage_labels)
print(berechnete_matrix)

# Konfusionsmatrix speichern
fig_matrix, ax_matrix = plt.subplots(figsize=(7, 6))

ConfusionMatrixDisplay(
    confusion_matrix=berechnete_matrix,
    display_labels=["Verblieben", "Abgewandert"]
).plot(ax=ax_matrix)

ax_matrix.set_title("Konfusionsmatrix des XGBoost-Modells")
plt.tight_layout()
plt.savefig(
    "prototype/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig_matrix)

# ROC-Kurve speichern
fig_roc, ax_roc = plt.subplots(figsize=(7, 6))

RocCurveDisplay.from_predictions(
    y_test,
    abwanderung_scores,
    ax=ax_roc
)

ax_roc.set_title(
    f"ROC-Kurve des XGBoost-Modells (AUC = {wert_auc:.4f})"
)
plt.tight_layout()
plt.savefig(
    "prototype/roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig_roc)

print("\nEvaluation abgeschlossen.")
print("Gespeicherte Dateien:")
print("- prototype/confusion_matrix.png")
print("- prototype/roc_curve.png")
