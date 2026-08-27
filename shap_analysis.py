import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split

# Pfadkonfiguration
BASE_DIR = Path(".")
DATA_PATH = BASE_DIR / "data" / "Churn_Modelling.csv"
MODEL_DIR = BASE_DIR / "prototype" / "model"
OUT_DIR = Path("shap_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_artifacts():
    """Lädt das trainierte Modell, den Preprocessing-Transformer und die Feature-Namen."""
    with open(MODEL_DIR / "final_xgboost.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "final_transformer.pkl", "rb") as f:
        transformer = pickle.load(f)
    with open(MODEL_DIR / "final_feature_names.pkl", "rb") as f:
        features = pickle.load(f)
    return model, transformer, features

def main():
    # 1. Daten laden und Testsplit replizieren
    print("Datensatz wird geladen und Testdaten werden vorbereitet...")
    df = pd.read_csv(DATA_PATH, sep=";")
    
    # Nicht relevante Spalten und Zielvariable trennen
    drop_cols = ["RowNumber", "CustomerId", "Surname", "Exited"]
    X = df.drop(columns=drop_cols)
    y = df["Exited"]

    # Stratifizierter Split (äquivalent zum Trainings-Setup)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # 2. Modellkomponenten laden & Testdaten transformieren
    model, transformer, feature_names = load_artifacts()
    
    X_test_trans = pd.DataFrame(
        transformer.transform(X_test),
        columns=feature_names,
        index=X_test.index
    )

    # Vorhersagen und Wahrscheinlichkeiten berechnen
    probs = model.predict_proba(X_test_trans)[:, 1]
    print(f"Testbeobachtungen: {len(X_test)} | Maximale Churn-Wahrscheinlichkeit: {probs.max():.4f}")

    # 3. Globale SHAP-Analyse
    print("Berechne SHAP-Werte mittels TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test_trans)
    
    # Sicherstellen des korrekten Formats bei binärer Klassifikation
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_vals = np.asarray(shap_vals)

    # Summary Plot (Bienenwarm-Diagramm)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_test_trans, show=False)
    plt.title("SHAP Summary Plot – Kundenabwanderung")
    plt.savefig(OUT_DIR / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Globale Feature-Wichtigkeit (Balkendiagramm)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_test_trans, plot_type="bar", show=False)
    plt.title("Globale Feature-Wichtigkeit (SHAP)")
    plt.savefig(OUT_DIR / "shap_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Lokale Erklärung (Kunde mit höchstem Churn-Risiko)
    top_idx = int(np.argmax(probs))
    expected_val = explainer.expected_value
    if isinstance(expected_val, (list, np.ndarray)):
        expected_val = np.asarray(expected_val).flatten()[0]

    explanation = shap.Explanation(
        values=shap_vals[top_idx],
        base_values=expected_val,
        data=X_test_trans.iloc[top_idx].values,
        feature_names=feature_names
    )

    # Waterfall Plot für Einzelfallanalyse
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=10, show=False)
    plt.title(f"SHAP Waterfall Plot (Index: {top_idx}, Churn-Wsk: {probs[top_idx]:.2%})")
    plt.savefig(OUT_DIR / "shap_waterfall_kunde.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5. Export der SHAP-Werte und Feature-Rankings
    pd.DataFrame(shap_vals, columns=feature_names).to_csv(
        OUT_DIR / "shap_werte_testdatensatz.csv", index=False
    )

    # Top-Features nach mittlerem absolutem SHAP-Wert sortieren
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": np.abs(shap_vals).mean(axis=0)
    }).sort_values(by="Mean_Abs_SHAP", ascending=False)

    print("\n--- TOP 10 WICHTIGSTE FEATURES (SHAP) ---")
    print(importance_df.head(10).to_string(index=False))
    print(f"\nAlle Grafiken und Tabellen wurden erfolgreich in '{OUT_DIR}/' gespeichert.")

if __name__ == "__main__":
    main()

