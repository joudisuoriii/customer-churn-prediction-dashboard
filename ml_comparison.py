# -*- coding: utf-8 -*-
# Bachelorarbeit - Vergleich der Klassifikationsmodelle
# Logistische Regression, Random Forest und XGBoost mit SMOTE

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

# Pfade
quell_datei = os.path.join("data", "Churn_Modelling.csv")
ziel_ordner = "ml_results"
os.makedirs(ziel_ordner, exist_ok=True)

print("\n=== MODELLVERGLEICH ===")
print("\nDatensatz wird eingelesen...")

try:
    daten_basis = pd.read_csv(quell_datei, sep=";")
    if daten_basis.shape[1] == 1:
        daten_basis = pd.read_csv(quell_datei, sep=",")
except FileNotFoundError:
    print(f"Fehler: Datei wurde unter '{quell_datei}' nicht gefunden.")
    raise SystemExit(1)

print(f"Datei erfolgreich geladen: {quell_datei}")
print(f"Zeilenanzahl: {daten_basis.shape[0]}")
print(f"Spaltenanzahl: {daten_basis.shape[1]}")
print("\nVerteilung der Zielvariable (Exited):")
print(daten_basis["Exited"].value_counts())
print(f"Gesamte Abwanderungsrate: {daten_basis['Exited'].mean() * 100:.2f}%")

# Identifikationsspalten entfernen
id_spalten = ["RowNumber", "CustomerId", "Surname"]
vorhandene_id_spalten = [
    spalte for spalte in id_spalten
    if spalte in daten_basis.columns
]

X_daten = daten_basis.drop(columns=vorhandene_id_spalten + ["Exited"])
y_ziel = daten_basis["Exited"]

print("\nVorbereitung der Merkmale...")
print(f"Entfernte ID-Spalten: {vorhandene_id_spalten}")
print("Verbleibende Merkmale:", list(X_daten.columns))

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    X_daten,
    y_ziel,
    test_size=0.30,
    random_state=42,
    stratify=y_ziel
)

print("\nTrain-Test-Aufteilung:")
print(f"Training: {len(X_train)} Beobachtungen")
print(f"Test:     {len(X_test)} Beobachtungen")
print(f"Abwanderungsrate im Training: {y_train.mean() * 100:.2f}%")
print(f"Abwanderungsrate im Testset:  {y_test.mean() * 100:.2f}%")

# Numerische und kategoriale Merkmale
numerische_merkmale = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

kategorische_merkmale = X_train.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

# Gemeinsame Vorverarbeitung für alle drei Modelle
numerische_vorverarbeitung = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

kategorische_vorverarbeitung = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "encoder",
        OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

vorverarbeitung = ColumnTransformer(transformers=[
    ("numeric", numerische_vorverarbeitung, numerische_merkmale),
    ("categorical", kategorische_vorverarbeitung, kategorische_merkmale)
])

# Zu vergleichende Modelle
algorithmen = {
    "Logistische Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}

print("\nModelltraining und Evaluation mit SMOTE...")
zusammenfassung = []

for name, algorithmus in algorithmen.items():
    print(f"\nStarte Training für: {name}")

    modell_ablauf = ImbPipeline(steps=[
        ("preprocessing", vorverarbeitung),
        ("smote", SMOTE(random_state=42)),
        ("model", algorithmus)
    ])

    modell_ablauf.fit(X_train, y_train)

    vorhersage_y = modell_ablauf.predict(X_test)
    wahrscheinlichkeit_y = modell_ablauf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, vorhersage_y)
    prec = precision_score(y_test, vorhersage_y, zero_division=0)
    rec = recall_score(y_test, vorhersage_y, zero_division=0)
    f1 = f1_score(y_test, vorhersage_y, zero_division=0)
    auc = roc_auc_score(y_test, wahrscheinlichkeit_y)

    zusammenfassung.append({
        "Modell": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc
    })

    print(
        f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | "
        f"Recall: {rec:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {auc:.4f}"
    )

    print("\nKlassifikationsbericht:")
    print(classification_report(
        y_test,
        vorhersage_y,
        target_names=["Verblieben", "Abgewandert"],
        zero_division=0
    ))

    # Konfusionsmatrix speichern
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
            plt.text(
                j,
                i,
                str(cm_matrix[i, j]),
                ha="center",
                va="center",
                color="black"
            )

    plt.tight_layout()
    datei_name = name.lower().replace(" ", "_")
    plt.savefig(
        os.path.join(
            ziel_ordner,
            f"konfusionsmatrix_{datei_name}.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

# Vergleichstabelle speichern
vergleich_df = pd.DataFrame(zusammenfassung)

print("\n=== ABSCHLIESSENDER MODELLVERGLEICH ===")
print(vergleich_df.to_string(index=False))

vergleich_df.to_csv(
    os.path.join(ziel_ordner, "modell_vergleich.csv"),
    index=False,
    sep=";"
)

print(
    f"\nModellvergleich abgeschlossen. Die Ergebnisse wurden in "
    f"'{ziel_ordner}/' gespeichert."
)
