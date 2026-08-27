# -*- coding: utf-8 -*-
# Bachelorarbeit - Modelltraining und Export (Schritt 8.1)
# Vorbereitung der Dateien fuer die Web-App (app.py)

import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Pfade fuer Daten und Prototyp-Modell
pfad_daten_csv = "data/Churn_Modelling.csv"
ordner_prototyp = "prototype/model"
os.makedirs(ordner_prototyp, exist_ok=True)

# 1. Daten laden
print("Lade Datensatz...")
kunden_daten = pd.read_csv(pfad_daten_csv, sep=";")
print(f"Daten erfolgreich geladen. Zeilen: {kunden_daten.shape[0]}, Spalten: {kunden_daten.shape[1]}")

# 2. Features und Target trennen
zielwert_name = "Exited"
id_spalten_liste = ["RowNumber", "CustomerId", "Surname"]

X_merkmale = kunden_daten.drop(columns=id_spalten_liste + [zielwert_name])
y_zielwert = kunden_daten[zielwert_name]

print(f"Entfernte Spalten: {id_spalten_liste}")
print("Verbleibende Merkmale:", list(X_merkmale.columns))
print("Klassenverteilung:")
print(y_zielwert.value_counts())
print(f"Abwanderungsrate gesamt: {y_zielwert.mean() * 100:.2f}%")

# 3. Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    X_merkmale,
    y_zielwert,
    test_size=0.30,
    random_state=42,
    stratify=y_zielwert
)

print(f"Größe Training: {len(X_train)} | Größe Test: {len(X_test)}")
print(f"Abwanderung im Training: {y_train.mean() * 100:.2f}%")
print(f"Abwanderung im Testset: {y_test.mean() * 100:.2f}%")

# 4. Spaltentypen festlegen
kategorische_merkmale = ["Geography", "Gender"]
numerische_merkmale = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]

# 5. Pipeline fuer Preprocessing bauen
daten_transformer = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numerische_merkmale),
        ("categorical", OneHotEncoder(drop="first", handle_unknown="ignore"), kategorische_merkmale)
    ]
)

# 6. XGBoost-Klassifikator erstellen
finales_modell = XGBClassifier(
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

# 7. Daten transformieren
X_train_transformed = daten_transformer.fit_transform(X_train)
X_test_transformed = daten_transformer.transform(X_test)
print("Preprocessing fuer Training und Test abgeschlossen.")
# 8. SMOTE auf Trainingsdaten anwenden
print("Starte SMOTE-Balancierung...")
print("Klassenverteilung vor SMOTE:")
print(y_train.value_counts())

smote_werkzeug = SMOTE(random_state=42)
X_train_balanciert, y_train_balanciert = smote_werkzeug.fit_resample(
    X_train_transformed, 
    y_train
)

print("Klassenverteilung nach SMOTE:")
print(y_train_balanciert.value_counts())
print("Hinweis: Testdaten wurden nicht mit SMOTE veraendert.")

# 9. Finales Modell trainieren
print("XGBoost-Modell wird trainiert...")
finales_modell.fit(X_train_balanciert, y_train_balanciert)
print("Modell erfolgreich trainiert.")

# 10. Namen der transformierten Features ausgeben
print("Extrahiere Feature-Namen...")
liste_merkmal_namen = daten_transformer.get_feature_names_out()
print(f"Anzahl der Merkmale nach Transformation: {len(liste_merkmal_namen)}")

print("Verfuegbare Merkmale:")
for merkmal in liste_merkmal_namen:
    print(f" - {merkmal}")

# 11. Speicherpfade definieren
pfad_modell_pkl = os.path.join(ordner_prototyp, "final_xgboost.pkl")
pfad_transformer_pkl = os.path.join(ordner_prototyp, "final_transformer.pkl")
pfad_merkmale_pkl = os.path.join(ordner_prototyp, "final_feature_names.pkl")

# 12. Dateien exportieren
print("Speichere Modell-Dateien fuer app.py...")

with open(pfad_modell_pkl, "wb") as datei:
    pickle.dump(finales_modell, datei)
print(f"Modell gespeichert: {pfad_modell_pkl}")

with open(pfad_transformer_pkl, "wb") as datei:
    pickle.dump(daten_transformer, datei)
print(f"Transformer gespeichert: {pfad_transformer_pkl}")

with open(pfad_merkmale_pkl, "wb") as datei:
    pickle.dump(list(liste_merkmal_namen), datei)
print(f"Feature-Namen gespeichert: {pfad_merkmale_pkl}")

# 13. Exportierte Dateien kurz validieren
print("Ueberpruefe gespeicherte Dateien auf der Festplatte:")
gespeicherte_dateien = [pfad_modell_pkl, pfad_transformer_pkl, pfad_merkmale_pkl]

for dateipfad in gespeicherte_dateien:
    if os.path.exists(dateipfad):
        groesse_kb = os.path.getsize(dateipfad) / 1024
        print(f" OK: {dateipfad} ({groesse_kb:.1f} KB)")
    else:
        print(f" FEHLER: Datei fehlt -> {dateipfad}")

# 14. Testlauf zur Funktionskontrolle
print("Starte kurzen Testlauf mit den exportierten Dateien...")

with open(pfad_modell_pkl, "rb") as datei:
    gespeichertes_modell = pickle.load(datei)

with open(pfad_transformer_pkl, "rb") as datei:
    gespeicherter_transformer = pickle.load(datei)

# Erste 5 Zeilen fuer den Test nutzen
X_test_probe = X_test.iloc[:5]
y_test_probe = y_test.iloc[:5]

X_test_probe_transformed = gespeicherter_transformer.transform(X_test_probe)

wahrscheinlichkeiten = gespeichertes_modell.predict_proba(X_test_probe_transformed)[:, 1]
vorhergesagte_klassen = (wahrscheinlichkeiten >= 0.50).astype(int)

# Ergebnistabelle fuer die Konsole bauen
pruef_tabelle = pd.DataFrame({
    "Wahrscheinlichkeit_Churn": wahrscheinlichkeiten,
    "Vorhergesagt_Exited": vorhergesagte_klassen,
    "Tatsaechlich_Exited": y_test_probe.values
})

print("\nErgebnisse des Testlaufs:")
print(pruef_tabelle.to_string(index=False))

print("\nSchritt 8.1 erfolgreich beendet.")

