# -*- coding: utf-8 -*-
# Bachelorarbeit - Prototyp Web-App (Schritt 8.2)
# Frontend-Dashboard fuer interaktive Vorhersagen und SHAP-Erklaerungen

import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# Konfiguration und Datenpfade einrichten
pfad_modell_datei = "prototype/model/final_xgboost.pkl"
pfad_transformer_datei = "prototype/model/final_transformer.pkl"
pfad_merkmale_datei = "prototype/model/final_feature_names.pkl"

st.set_page_config(
    page_title="Kundenabwanderung Dashboard",
    page_icon="🏦",
    layout="wide"
)

# Gespeicherte Objekte laden
@st.cache_resource
def geladene_dateien_einlesen():
    with open(pfad_modell_datei, "rb") as f:
        modell_objekt = pickle.load(f)
    with open(pfad_transformer_datei, "rb") as f:
        transformer_objekt = pickle.load(f)
    with open(pfad_merkmale_datei, "rb") as f:
        merkmale_namen = pickle.load(f)
    return modell_objekt, transformer_objekt, merkmale_namen

finales_modell, daten_transformer, merkmal_namen_liste = geladene_dateien_einlesen()

# Header der Webseite aufbauen
st.title("🏦 Kundenabwanderung Vorhersage-System")
st.subheader("KI-gestütztes Dashboard zur Vorhersage und Erklärung von Kunden-Churn")
st.write(
    "Dieser funktionale Prototyp verwendet ein trainiertes "
    "XGBoost-Modell zur Schätzung der Abwanderungswahrscheinlichkeit. "
    "SHAP wird eingesetzt, um die individuelle Modellentscheidung "
    "verständlich zu erklären."
)
st.divider()

# Eingabemaske fuer Benutzereingaben erstellen
st.header("1. Eingabe der Kundendaten")
spalte1, spalte2, spalte3 = st.columns(3)

with spalte1:
    kredit_score = st.number_input("Kredit-Score", min_value=300, max_value=900, value=650)
    alter_kunde = st.number_input("Alter", min_value=18, max_value=100, value=40)
    laufzeit_jahre = st.number_input("Kundendauer (Jahre)", min_value=0, max_value=10, value=5)

with spalte2:
    kontostand_wert = st.number_input("Kontostand (€)", min_value=0.0, value=50000.0, step=1000.0)
    anzahl_produkte = st.number_input("Anzahl Produkte", min_value=1, max_value=4, value=2)
    gehalt_geschaetzt = st.number_input("Geschätztes Jahresgehalt (€)", min_value=0.0, value=50000.0, step=1000.0)

with spalte3:
    herkunftsland = st.selectbox("Geografische Herkunft", ["France", "Germany", "Spain"])
    geschlecht_kunde = st.selectbox("Geschlecht", ["Female", "Male"])
    besitzt_kreditkarte = st.selectbox("Besitzt eine Kreditkarte?", [0, 1], format_func=lambda x: "Ja" if x == 1 else "Nein")
    ist_aktiv = st.selectbox("Aktives Bankmitglied?", [0, 1], format_func=lambda x: "Ja" if x == 1 else "Nein")

# Eingaben in ein DataFrame transformieren
neuer_kunde_df = pd.DataFrame({
    "CreditScore": [kredit_score],
    "Geography": [herkunftsland],
    "Gender": [geschlecht_kunde],
    "Age": [alter_kunde],
    "Tenure": [laufzeit_jahre],
    "Balance": [kontostand_wert],
    "NumOfProducts": [anzahl_produkte],
    "HasCrCard": [besitzt_kreditkarte],
    "IsActiveMember": [ist_aktiv],
    "EstimatedSalary": [gehalt_geschaetzt]
})

st.divider()
st.header("2. Analyse der Abwanderungsgefahr")

# Berechnung starten bei Button-Klick
if st.button("🔍 Abwanderungsrisiko berechnen", type="primary", use_container_width=True):

    # Daten transformieren
    daten_transformiert = daten_transformer.transform(neuer_kunde_df)
    transformierter_kunde_df = pd.DataFrame(daten_transformiert, columns=merkmal_namen_liste)

    # Churn-Wahrscheinlichkeit vorhersagen
    abwanderung_wahrscheinlichkeit = finales_modell.predict_proba(transformierter_kunde_df)[0][1]
    prozent_wahrscheinlichkeit = abwanderung_wahrscheinlichkeit * 100

    # Klasse ermitteln
    vorhergesagte_klasse = int(finales_modell.predict(transformierter_kunde_df)[0])

    # Risikostufe bestimmen (30 / 60 Thresholds)
    if prozent_wahrscheinlichkeit >= 60:
        risiko_titel = "HOCH"
        risiko_symbol = "🔴"
    elif prozent_wahrscheinlichkeit >= 30:
        risiko_titel = "MITTEL"
        risiko_symbol = "🟠"
    else:
        risiko_titel = "GERING"
        risiko_symbol = "🟢"

    # Ergebnisse im Dashboard anzeigen
    res_spalte1, res_spalte2, res_spalte3 = st.columns(3)

    with res_spalte1:
        st.metric("Churn-Wahrscheinlichkeit", f"{prozent_wahrscheinlichkeit:.2f}%")

    with res_spalte2:
        if vorhergesagte_klasse == 1:
            st.error("⚠️ Modell prognostiziert: Kunde wird abwandern.")
        else:
            st.success("✅ Modell prognostiziert: Kunde bleibt.")

    with res_spalte3:
        st.metric("Risikostufe", f"{risiko_symbol} {risiko_titel}")

    # Fortschrittsbalken anzeigen
    st.subheader("Risiko-Indikator")
    st.progress(min(int(prozent_wahrscheinlichkeit), 100))
    st.caption("Die Prozentzahl entspricht der vom XGBoost-Modell geschätzten Wahrscheinlichkeit einer Abwanderung.")

    # 3. SHAP Erklärung aufbauen
    st.divider()
    st.header("3. Erklärbare KI (SHAP)")
    st.write(
        "SHAP zeigt, welche Merkmale die individuelle Modellentscheidung beeinflussen "
        "und ob sie das Churn-Risiko erhöhen oder reduzieren."
    )

    try:
        # SHAP-Werte fuer den Kunden berechnen
        shap_analysator = shap.TreeExplainer(finales_modell)
        shap_raw = shap_analysator.shap_values(transformierter_kunde_df)

        if isinstance(shap_raw, list):
            kunden_shap_raw = shap_raw[1][0]
        else:
            kunden_shap_raw = np.asarray(shap_raw)[0]

        # SHAP-Werte den Merkmalen zuordnen
        shap_dict = dict(zip(merkmal_namen_liste, kunden_shap_raw))

        # --- DUMMY-REDUKTION & SAUBERES RE-MAPPING ---
        # Zusammenfassen der geografischen Dummy-Effekte (Deutschland + Spanien) zu einem Gesamteffekt
        geo_shap_total = shap_dict.get("categorical__Geography_Germany", 0.0) + shap_dict.get("categorical__Geography_Spain", 0.0)
        gender_shap_val = shap_dict.get("categorical__Gender_Male", 0.0)

        # Bereinigte Feature-Liste & Display-Werte
        geo_shap_total = shap_dict.get("categorical__Geography_Germany", 0.0) + shap_dict.get("categorical__Geography_Spain", 0.0)
        gender_shap_val = shap_dict.get("categorical__Gender_Male", 0.0)
        bereinigte_features = [
            ("Credit Score", shap_dict.get("numeric__CreditScore", 0.0), str(kredit_score)),
            ("Alter", shap_dict.get("numeric__Age", 0.0), f"{alter_kunde} J."),
            ("Kundendauer", shap_dict.get("numeric__Tenure", 0.0), f"{laufzeit_jahre} J."),
            ("Kontostand", shap_dict.get("numeric__Balance", 0.0), f"{kontostand_wert:,.0f} €"),
            ("Anzahl Produkte", shap_dict.get("numeric__NumOfProducts", 0.0), str(anzahl_produkte)),
            ("Kreditkarte", shap_dict.get("numeric__HasCrCard", 0.0), "Ja" if besitzt_kreditkarte == 1 else "Nein"),
            ("Aktivität", shap_dict.get("numeric__IsActiveMember", 0.0), "Ja" if ist_aktiv == 1 else "Nein"),
            ("Gehalt", shap_dict.get("numeric__EstimatedSalary", 0.0), f"{gehalt_geschaetzt:,.0f} €"),
            ("Herkunft", geo_shap_total, str(herkunftsland)),
            ("Geschlecht", gender_shap_val, str(geschlecht_kunde))
        ]

        display_names = [item[0] for item in bereinigte_features]
        display_shap = np.array([item[1] for item in bereinigte_features])
        display_data = np.array([item[2] for item in bereinigte_features], dtype=object)

        # Tabelle für textuelle Zusammenfassung
        erklaerung_tabelle = pd.DataFrame({
            "Anzeige": display_names,
            "SHAP-Wert": display_shap,
            "Absoluter Einfluss": np.abs(display_shap)
        }).sort_values("Absoluter Einfluss", ascending=False)

        # Wichtigste Faktoren ausgeben
        st.subheader("Wichtigste Einflussfaktoren")
        top_merkmale = erklaerung_tabelle.head(5)

        for _, zeile in top_merkmale.iterrows():
            merkmal_name = zeile["Anzeige"]
            wert_shap = zeile["SHAP-Wert"]

            if wert_shap > 0.001:
                st.write(f"🔴 **{merkmal_name}** → wirkt im Modell risikosteigernd")
            elif wert_shap < -0.001:
                st.write(f"🟢 **{merkmal_name}** → wirkt im Modell risikoreduzierend")

        # SHAP Waterfall-Diagramm anzeigen
        st.subheader("SHAP Waterfall – Erklärung der individuellen Vorhersage")
        st.caption("Positive Beiträge erhöhen die Churn-Prognose, negative Beiträge reduzieren sie.")

        base_val = shap_analysator.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = np.asarray(base_val).flatten()[0]

        waterfall_erklaerung = shap.Explanation(
            values=display_shap,
            base_values=base_val,
            data=display_data,
            feature_names=display_names
        )

        fig_waterfall, ax_wf = plt.subplots(figsize=(10, 6.5))
        shap.plots.waterfall(waterfall_erklaerung, max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig_waterfall, clear_figure=True)
        plt.close(fig_waterfall)

        # Textliche Interpretation ausgeben
        st.subheader("Interpretation")
        positive_faktoren = erklaerung_tabelle[erklaerung_tabelle["SHAP-Wert"] > 0.001].head(3)
        negative_faktoren = erklaerung_tabelle[erklaerung_tabelle["SHAP-Wert"] < -0.001].head(3)

        if len(positive_faktoren) > 0:
            st.write("**Faktoren, die das Churn-Risiko erhöhen:**")
            for _, zeile in positive_faktoren.iterrows():
                st.write(f"- {zeile['Anzeige']}")

        if len(negative_faktoren) > 0:
            st.write("**Faktoren, die das Churn-Risiko reduzieren:**")
            for _, zeile in negative_faktoren.iterrows():
                st.write(f"- {zeile['Anzeige']}")

        # Methodischer Hinweis am Ende
        st.info(
            "Hinweis: SHAP erklärt die Modellentscheidung. Ein positiver SHAP-Wert erhöht den Beitrag "
            "zur Churn-Klasse, während ein negativer SHAP-Wert den Beitrag zur Churn-Klasse reduziert. "
            "SHAP ersetzt nicht das Machine-Learning-Modell, sondern ergänzt die Vorhersage."
        )

    except Exception as error:
        st.error("Die SHAP-Erklärung konnte nicht berechnet werden.")
        st.code(str(error))

# Footer
st.divider()
st.caption("Prototyp – Bank Customer Churn Prediction | XGBoost + SHAP | Bachelor Thesis")

