# -*- coding: utf-8 -*-
# Bachelorarbeit - Vergleich von Machine-Learning-Modellen
# Schritt 4: Modellierung, SMOTE-Balancierung und Evaluation

import os
import warnings

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# 1. Konfiguration und Pfade
quell_datei = os.path.join("data", "Churn_Modelling.csv")
ziel_ordner = "ml_results"
os.makedirs(ziel_ordner, exist_ok=True)

# 2. Datensatz laden
print("\n=== SCHRITT 4: MODELLIERUNG UND VERGLEICH ===")
print("\n1. Datensatz einlesen...")

try:
    daten_basis = pd.read_csv(quell_datei, sep=';')
    if daten_basis.shape[1] == 1:
        daten_basis = pd.read_csv(quell_datei, sep=',')
    print(f"Datei erfolgreich geladen: {quell_datei}")
except FileNotFoundError:
    print(f"Fehler: Datei konnte nicht gefunden werden unter '{quell_datei}'")
    exit()

print(f"Zeilenanzahl: {daten_basis.shape[0]}")
print(f"Spaltenanzahl: {daten_basis.shape[1]}")

print("\nVerteilung der Zielvariable (Exited):")
print(daten_basis["Exited"].value_counts())
print(f"Gesamte Abwanderungsrate: {daten_basis['Exited'].mean() * 100:.2f}%")

# 3. Bereinigung von Identifikatoren
print("\n2. Vorbereitung der Merkmale...")
irrelevante_spalten = ["RowNumber", "CustomerId", "Surname"]
aktuelle_merkmale = [spalte for spalte in irrelevante_spalten if spalte in daten_basis.columns]

X_daten = daten_basis.drop(columns=aktuelle_merkmale + ["Exited"])
y_ziel = daten_basis["Exited"]

print(f"Entfernte ID-Spalten: {aktuelle_merkmale}")
print("Verbleibende Merkmale für das Modell:", list(X_daten.columns))

# 4. Train-Test-Split (Aufteilung)
print("\n3. Aufteilung in Training- und Testdaten...")
X_train, X_test, y_train, y_test = train_test_split(
    X_daten,
    y_ziel,
    test_size=0.30,
    random_state=42,
    stratify=y_ziel
)

print(f"Trainingsdaten-Größe: {len(X_train)}")
print(f"Testdaten-Größe:     {len(X_test)}")
print(f"Abwanderungsrate im Training: {y_train.mean() * 100:.2f}%")
print(f"Abwanderungsrate im Testset:  {y_test.mean() * 100:.2f}%")

# 5. Spatentypen identifizieren
num_merkmale = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_merkmale = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()

print("\n4. Analyse der Datentypen...")
print(f"Numerische Merkmale: {num_merkmale}")
print(f"Kategorische Merkmale: {cat_merkmale}")

# 6. Preprocessing Pipelines definieren
print("\n5. Erstellung der Preprocessing-Pipeline...")
num_ablauf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_ablauf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

vorverarbeitung = ColumnTransformer(transformers=[
    ("numeric", num_ablauf, num_merkmale),
    ("categorical", cat_ablauf, cat_merkmale)
])

# 7. Modelle definieren
print("\n6. Definition των Algorithmen...")
algorithmen = {
    "Logistische Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss", n_jobs=-1)
}

# 8. Training und Evaluation
print("\n7. Modelltraining und Evaluation mit SMOTE...")
zusammenfassung = []

for name, algorithmus in algorithmen.items():
    print(f"\nStarte Training für: {name}")

    modell_ablauf = ImbPipeline(steps=[
        ("preprocessing", vorverarbeitung),
        ("smote", SMOTE(random_state=42)),
        ("model", algorithmus)
    ])

    # Modell fitten
    modell_ablauf.fit(X_train, y_train)

    # Vorhersagen generieren
    vorhersage_y = modell_ablauf.predict(X_test)
    wahrscheinlichkeit_y = modell_ablauf.predict_proba(X_test)[:, 1]

    # Metriken berechnen
    acc = accuracy_score(y_test, vorhersage_y)
    prec = precision_score(y_test, vorhersage_y, zero_division=0)
    rec = recall_score(y_test, vorhersage_y, zero_division=0)
    f1 = f1_score(y_test, vorhersage_y, zero_division=0)
    auc = roc_auc_score(y_test, wahrscheinlichkeit_y)

    zusammenfassung.append({
        "Modell": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "ROC-AUC": auc
    })

    print(f"Accuracy:  {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {auc:.4f}")
    
    print("\nKlassifikationsbericht:")
    print(classification_report(y_test, vorhersage_y, target_names=["Verblieben", "Abgewandert"], zero_division=0))

    # Konfusionsmatrix erstellen und speichern
    cm_matrix = confusion_matrix(y_test, vorhersage_y)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Konfusionsmatrix - {name}")
    plt.colorbar()

    plt.xticks([0, 1], ["Verblieben", "Abgewandert"])
    plt.yticks([0, 1], ["Verblieben", "Abgewandert"])
    plt.xlabel("Vorhergesagt")
    plt.ylabel("Tatsächlich")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm_matrix[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    datei_name = name.lower().replace(" ", "_")
    plt.savefig(os.path.join(ziel_ordner, f"konfusionsmatrix_{datei_name}.png"), dpi=150, bbox_inches="tight")
    plt.close()

# 9. Vergleichstabelle ausgeben
vergleich_df = pd.DataFrame(zusammenfassung)
print("\n=== ABSCHLIESSENDER MODELLVERGLEICH ===")
print(vergleich_df.to_string(index=False))
vergleich_df.to_csv(os.path.join(ziel_ordner, "modell_vergleich.csv"), index=False, sep=";")
print(f"\nVergleichstabelle wurde unter '{ziel_ordner}/modell_vergleich.csv' gespeichert.")
