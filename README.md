# Customer-Churn-Prediction-Dashboard

Dieses Projekt stellt den aktuellen Prototyp eines Entscheidungsunterstützungssystems zur Vorhersage von Kundenabwanderung im Bankensektor dar und wird im Rahmen einer Bachelorarbeit im Studiengang Wirtschaftsinformatik entwickelt.

## Ziel des Projekts

Ziel ist die Entwicklung eines interpretierbaren Decision Support Systems (DSS), das Machine-Learning-basierte Churn-Prognosen mit lokalen SHAP-Erklärungen in einem interaktiven Streamlit-Dashboard verbindet.

Der Prototyp kombiniert:
- Logistische Regression
- Random Forest
- XGBoost
- SMOTE zur Behandlung des Klassenungleichgewichts
- SHAP zur Interpretation der Modellvorhersagen
- Streamlit zur interaktiven Darstellung der Ergebnisse

## Datensatz

Für die Untersuchung wird ein Bankkunden-Datensatz mit insgesamt 10.000 Kunden verwendet.

Die Zielvariable verteilt sich wie folgt:
- 7.963 Kunden ohne Abwanderung (79,63 %)
- 2.037 abgewanderte Kunden (20,37 %)

## Methodisches Vorgehen

Der technische Workflow umfasst folgende Schritte:

1. Explorative Datenanalyse (EDA)
2. Datenaufbereitung und Vorverarbeitung
3. Aufteilung in Trainings- und Testdaten
4. Anwendung von SMOTE ausschließlich auf die Trainingsdaten
5. Training und Vergleich der drei Klassifikationsmodelle
6. Evaluation anhand mehrerer Metriken:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - ROC-AUC
7. Auswahl eines geeigneten Modells anhand der Evaluationsergebnisse
8. Interpretation der Modellvorhersagen mit SHAP
9. Integration des ausgewählten Modells und der SHAP-Erklärungen in ein Streamlit-Dashboard

## Modellvergleich

In der Untersuchung werden drei Klassifikationsmodelle miteinander verglichen:

- Logistische Regression
- Random Forest
- XGBoost

Die Modellauswahl erfolgt auf Grundlage der Ergebnisse des empirischen Vergleichs auf derselben Datenbasis.

## Explainable Artificial Intelligence

Zur Interpretation der Modellvorhersagen wird SHAP (SHapley Additive exPlanations) eingesetzt.

SHAP ermöglicht es, den Beitrag einzelner Merkmale zu einer konkreten Vorhersage darzustellen. Dadurch kann neben dem prognostizierten Abwanderungsrisiko nachvollzogen werden, welche Merkmale die jeweilige Vorhersage beeinflusst haben.

## Streamlit-Prototyp

Das interaktive Dashboard dient als Entscheidungsunterstützung und stellt unter anderem folgende Informationen dar:

- prognostiziertes Abwanderungsrisiko
- Risikokategorie
- lokale SHAP-Erklärung der jeweiligen Vorhersage

Das System dient ausschließlich der Entscheidungsunterstützung und trifft keine automatischen Entscheidungen über Kundenbindungsmaßnahmen.

## Projektstruktur

```text
.
├── churn_analysis.py
├── evaluate_model.py
├── ml_comparison.py
├── save_final_model.py
├── shap_analysis.py
├── data/
├── eda_results/
├── ml_results/
├── shap_results/
└── prototype/
    └── app.py
Status

Dieses Repository enthält den aktuellen technischen Prototyp des Projekts.

Die Implementierung befindet sich im Rahmen der Bachelorarbeit in Weiterentwicklung und kann auf Grundlage methodischer Anforderungen sowie des Feedbacks der Betreuungsperson weiter angepasst werden.
