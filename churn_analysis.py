# -*- coding: utf-8 -*-
# Bachelorarbeit - Explorative Datenanalyse
# Erste Prüfung und Beschreibung des verwendeten Bankkunden-Datensatzes

import os

import pandas as pd
import matplotlib.pyplot as plt

# Pfade
pfad_zur_datei = os.path.join("data", "Churn_Modelling.csv")
ordner_ergebnisse = "eda_results"
os.makedirs(ordner_ergebnisse, exist_ok=True)

# Datensatz einlesen
print("Lade Datensatz aus:", pfad_zur_datei)

try:
    bank_daten = pd.read_csv(pfad_zur_datei, sep=";")
    if bank_daten.shape[1] == 1:
        bank_daten = pd.read_csv(pfad_zur_datei, sep=",")
except FileNotFoundError:
    print(f"Fehler: Die Datei wurde unter '{pfad_zur_datei}' nicht gefunden.")
    raise SystemExit(1)

print("Daten erfolgreich geladen.")
print(f"Anzahl der Zeilen: {bank_daten.shape[0]}")
print(f"Anzahl der Spalten: {bank_daten.shape[1]}")
print("Verfügbare Merkmale:", bank_daten.columns.tolist())

# Datentypen und fehlende Werte prüfen
print("\n--- Übersicht der Datentypen ---")
print(bank_daten.dtypes)

print("\n--- Überprüfung auf fehlende Werte ---")
fehlende_werte = bank_daten.isnull().sum()
null_uebersicht = pd.DataFrame({
    "Anzahl_Fehlend": fehlende_werte,
    "Prozent_Fehlend": (fehlende_werte / len(bank_daten)) * 100
})
print(null_uebersicht)

if fehlende_werte.sum() == 0:
    print("Keine fehlenden Werte im Datensatz gefunden.")

# Duplikate prüfen
print("\n--- Überprüfung auf Duplikate ---")
doppelte_zeilen = bank_daten.duplicated().sum()
print(f"Anzahl doppelter Einträge: {doppelte_zeilen}")

if doppelte_zeilen > 0:
    bank_daten = bank_daten.drop_duplicates().reset_index(drop=True)
    print("Doppelte Einträge wurden entfernt.")

# Benötigte Spalten kontrollieren
notwendige_spalten = [
    "Exited", "CreditScore", "Geography", "Gender", "Age",
    "Tenure", "Balance", "NumOfProducts", "HasCrCard",
    "IsActiveMember", "EstimatedSalary"
]

fehlende_spalten = [
    spalte for spalte in notwendige_spalten
    if spalte not in bank_daten.columns
]

if fehlende_spalten:
    raise ValueError(f"Wichtige Spalten fehlen im Datensatz: {fehlende_spalten}")

# Einfache Plausibilitätsprüfung
print("\n--- Logische Datenprüfung ---")
print(
    "Ungültiges Alter (außerhalb 0-120):",
    ((bank_daten["Age"] <= 0) | (bank_daten["Age"] > 120)).sum()
)
print("Negative Kontostände:", (bank_daten["Balance"] < 0).sum())
print(
    "Negatives geschätztes Gehalt:",
    (bank_daten["EstimatedSalary"] < 0).sum()
)
print(
    "Unerwartete Werte in Zielvariable (Exited):",
    (~bank_daten["Exited"].isin([0, 1])).sum()
)

# Verteilung der Zielvariable
print("\n--- Verteilung der Zielvariable (Churn) ---")
anzahl_target = bank_daten["Exited"].value_counts().sort_index()
prozent_target = (
    bank_daten["Exited"].value_counts(normalize=True).sort_index() * 100
)

for wert in anzahl_target.index:
    status_label = "Verblieben" if wert == 0 else "Abgewandert"
    print(
        f"Klasse {wert} ({status_label}): {anzahl_target[wert]} Kunden "
        f"({prozent_target[wert]:.2f}%)"
    )

# EDA-Diagramme

# 1. Verteilung der Kundenabwanderung
plt.figure(figsize=(7, 5))
bank_daten["Exited"].value_counts().sort_index().plot(kind="bar")
plt.title("Verteilung der Kundenabwanderung (Churn)")
plt.xlabel("Kundenstatus (0 = Verblieben, 1 = Abgewandert)")
plt.ylabel("Anzahl der Kunden")
plt.tight_layout()
plt.savefig(
    os.path.join(ordner_ergebnisse, "01_churn_verteilung.png"),
    dpi=300
)
plt.close()

# 2. Alter nach Kundenstatus
plt.figure(figsize=(8, 5))
bank_daten.boxplot(column="Age", by="Exited", grid=False)
plt.suptitle("")
plt.title("Altersverteilung nach Kundenstatus")
plt.xlabel("Kundenstatus (0 = Verblieben, 1 = Abgewandert)")
plt.ylabel("Alter")
plt.tight_layout()
plt.savefig(
    os.path.join(ordner_ergebnisse, "02_alter_vs_churn.png"),
    dpi=300
)
plt.close()

# 3. Aktivitätsstatus und Churn
aktivitaet_tabelle = pd.crosstab(
    bank_daten["IsActiveMember"],
    bank_daten["Exited"],
    normalize="index"
) * 100
print("\nAbwanderungsrate nach Aktivitätsstatus:\n", aktivitaet_tabelle)

fig, ax = plt.subplots(figsize=(7, 5))
aktivitaet_tabelle.plot(kind="bar", stacked=True, ax=ax)
plt.title("Abwanderungsrate nach Aktivität der Mitglieder")
plt.xlabel("Aktives Mitglied (0 = Inaktiv, 1 = Aktiv)")
plt.ylabel("Prozent (%)")
plt.legend(["Verblieben", "Abgewandert"], title="Kundenstatus")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    os.path.join(ordner_ergebnisse, "03_aktivitaet_vs_churn.png"),
    dpi=300
)
plt.close()

# 4. Anzahl der Produkte und Churn
produkte_tabelle = pd.crosstab(
    bank_daten["NumOfProducts"],
    bank_daten["Exited"],
    normalize="index"
) * 100
print("\nAbwanderungsrate nach Anzahl der Produkte:\n", produkte_tabelle)

fig, ax = plt.subplots(figsize=(8, 5))
produkte_tabelle.plot(kind="bar", stacked=True, ax=ax)
plt.title("Abwanderungsrate nach Anzahl der Produkte")
plt.xlabel("Anzahl der Produkte")
plt.ylabel("Prozent (%)")
plt.legend(["Verblieben", "Abgewandert"], title="Kundenstatus")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    os.path.join(ordner_ergebnisse, "04_produkte_vs_churn.png"),
    dpi=300
)
plt.close()

# 5. Kontostand nach Kundenstatus
plt.figure(figsize=(8, 5))
bank_daten.boxplot(column="Balance", by="Exited", grid=False)
plt.suptitle("")
plt.title("Kontostand nach Kundenstatus")
plt.xlabel("Kundenstatus (0 = Verblieben, 1 = Abgewandert)")
plt.ylabel("Kontostand")
plt.tight_layout()
plt.savefig(
    os.path.join(ordner_ergebnisse, "05_kontostand_vs_churn.png"),
    dpi=300
)
plt.close()

print(f"\nEDA abgeschlossen. Die Grafiken wurden in '{ordner_ergebnisse}/' gespeichert.")
