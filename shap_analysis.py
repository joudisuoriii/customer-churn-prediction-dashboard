# -*- coding: utf-8 -*-
# Bachelorarbeit - SHAP-Analyse des finalen XGBoost-Modells
# Globale Feature-Wichtigkeit und lokale Erklärung einer Testbeobachtung

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split

# Pfade
basis_ordner = Path(".")
daten_pfad = basis_ordner / "data" / "Churn_Modelling.csv"
modell_ordner = basis_ordner / "prototype" / "model"
ergebnis_ordner = Path("shap_results")
ergebnis_ordner.mkdir(parents=True, exist_ok=True)


def modell_dateien_laden():
    """Lädt Modell, Transformer und die Namen der transformierten Merkmale."""
    with open(modell_ordner / "final_xgboost.pkl", "rb") as datei:
        modell = pickle.load(datei)

    with open(modell_ordner / "final_transformer.pkl", "rb") as datei:
        transformer = pickle.load(datei)

    with open(modell_ordner / "final_feature_names.pkl", "rb") as datei:
        merkmale = pickle.load(datei)

    return modell, transformer, merkmale


def main():
    print("Datensatz wird geladen und Testdaten werden vorbereitet...")

    daten = pd.read_csv(daten_pfad, sep=";")
    if daten.shape[1] == 1:
        daten = pd.read_csv(daten_pfad, sep=",")

    id_und_ziel = ["RowNumber", "CustomerId", "Surname", "Exited"]
    X_daten = daten.drop(columns=id_und_ziel)
    y_ziel = daten["Exited"]

    # Gleicher Testsplit wie beim Training
    _, X_test, _, _ = train_test_split(
        X_daten,
        y_ziel,
        test_size=0.30,
        random_state=42,
        stratify=y_ziel
    )

    modell, transformer, merkmalsnamen = modell_dateien_laden()

    X_test_transformiert = pd.DataFrame(
        transformer.transform(X_test),
        columns=merkmalsnamen,
        index=X_test.index
    )

    churn_wahrscheinlichkeiten = modell.predict_proba(
        X_test_transformiert
    )[:, 1]

    print(
        f"Testbeobachtungen: {len(X_test)} | "
        f"Maximale Churn-Wahrscheinlichkeit: "
        f"{churn_wahrscheinlichkeiten.max():.4f}"
    )

    print("Berechne SHAP-Werte mit TreeExplainer...")
    explainer = shap.TreeExplainer(modell)
    shap_werte = explainer.shap_values(X_test_transformiert)

    if isinstance(shap_werte, list):
        shap_werte = shap_werte[1]

    shap_werte = np.asarray(shap_werte)

    # SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_werte,
        X_test_transformiert,
        show=False
    )
    plt.title("SHAP Summary Plot – Kundenabwanderung")
    plt.savefig(
        ergebnis_ordner / "shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Globale Feature-Wichtigkeit
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_werte,
        X_test_transformiert,
        plot_type="bar",
        show=False
    )
    plt.title("Globale Feature-Wichtigkeit (SHAP)")
    plt.savefig(
        ergebnis_ordner / "shap_feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Lokale Erklärung für den Testfall mit dem höchsten Churn-Risiko
    top_index = int(np.argmax(churn_wahrscheinlichkeiten))

    basiswert = explainer.expected_value
    if isinstance(basiswert, (list, np.ndarray)):
        basiswert = np.asarray(basiswert).flatten()[0]

    erklaerung = shap.Explanation(
        values=shap_werte[top_index],
        base_values=basiswert,
        data=X_test_transformiert.iloc[top_index].values,
        feature_names=merkmalsnamen
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(
        erklaerung,
        max_display=10,
        show=False
    )
    plt.title(
        "SHAP Waterfall Plot "
        f"(Churn-Wahrscheinlichkeit: "
        f"{churn_wahrscheinlichkeiten[top_index]:.2%})"
    )
    plt.savefig(
        ergebnis_ordner / "shap_waterfall_kunde.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # SHAP-Werte des Testdatensatzes speichern
    pd.DataFrame(
        shap_werte,
        columns=merkmalsnamen
    ).to_csv(
        ergebnis_ordner / "shap_werte_testdatensatz.csv",
        index=False
    )

    # Ranking nach mittlerem absolutem SHAP-Wert
    importance_df = pd.DataFrame({
        "Feature": merkmalsnamen,
        "Mean_Abs_SHAP": np.abs(shap_werte).mean(axis=0)
    }).sort_values(
        by="Mean_Abs_SHAP",
        ascending=False
    )

    print("\n--- TOP 10 WICHTIGSTE FEATURES (SHAP) ---")
    print(importance_df.head(10).to_string(index=False))
    print(
        f"\nSHAP-Analyse abgeschlossen. Die Ergebnisse wurden in "
        f"'{ergebnis_ordner}/' gespeichert."
    )


if __name__ == "__main__":
    main()
