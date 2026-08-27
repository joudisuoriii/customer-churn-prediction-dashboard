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

### Datenquelle

Für die Untersuchung wird der öffentlich verfügbare Kaggle-Datensatz „Customer Churn from a Bank“ verwendet:

https://www.kaggle.com/datasets/murilozangari/customer-churn-from-a-bank

Die verwendete Datei `Churn_Modelling.csv` enthält 10.000 Bankkundendatensätze. Die CSV-Datei wird nicht direkt in diesem Repository bereitgestellt.

Für die lokale Ausführung muss die Datei unter folgendem Pfad abgelegt werden:

```text
data/Churn_Modelling.csv
```

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

In der Untersuchung werden drei Klassifikationsmodelle auf derselben Datenbasis miteinander verglichen:

- Logistische Regression
- Random Forest
- XGBoost

Die Modelle werden anhand von Accuracy, Precision, Recall, F1-Score und ROC-AUC evaluiert.

Die Ergebnisse des Modellvergleichs sind:

| Modell | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistische Regression | 0.7257 | 0.4013 | 0.7054 | 0.5116 | 0.7912 |
| Random Forest | 0.8510 | 0.6507 | 0.5794 | 0.6130 | 0.8628 |
| XGBoost | 0.8610 | 0.6830 | 0.5925 | 0.6345 | 0.8807 |

Auf Grundlage des Modellvergleichs wurde XGBoost für den finalen Prototyp ausgewählt. Das Modell erzielt im Vergleich die höchste Accuracy, den höchsten F1-Score sowie den höchsten ROC-AUC-Wert.

## Explainable Artificial Intelligence

Zur Interpretation der Modellvorhersagen wird SHAP (SHapley Additive exPlanations) eingesetzt.

SHAP ermöglicht es, den Beitrag einzelner Merkmale zu einer konkreten Modellvorhersage darzustellen. Dadurch kann neben dem prognostizierten Abwanderungsrisiko nachvollzogen werden, welche Merkmale die jeweilige Vorhersage des Modells beeinflusst haben.

Die SHAP-Analyse wird sowohl zur Untersuchung der globalen Merkmalsrelevanz als auch zur lokalen Erklärung einzelner Kundenprognosen eingesetzt.

## Streamlit-Prototyp

Das interaktive Dashboard dient als Entscheidungsunterstützung und stellt unter anderem folgende Informationen dar:

- prognostizierte Abwanderungswahrscheinlichkeit
- Risikokategorie
- wichtigste Einflussfaktoren der individuellen Vorhersage
- lokale SHAP-Erklärung der jeweiligen Vorhersage

Die Risikokategorien werden im Prototyp anhand der prognostizierten Abwanderungswahrscheinlichkeit wie folgt dargestellt:

- geringes Risiko: unter 30 %
- mittleres Risiko: 30 % bis unter 60 %
- hohes Risiko: ab 60 %

Das System dient ausschließlich der Entscheidungsunterstützung und trifft keine automatischen Entscheidungen über Kundenbindungsmaßnahmen.

## Installation und lokale Ausführung

Die für das Projekt benötigten Python-Bibliotheken sind in der Datei `requirements.txt` aufgeführt und können mit folgendem Befehl installiert werden:

```bash
pip install -r requirements.txt
```

Die einzelnen Analyseschritte können anschließend über die entsprechenden Python-Skripte ausgeführt werden:

```bash
python churn_analysis.py
python ml_comparison.py
python save_final_model.py
python evaluate_model.py
python shap_analysis.py
```

Der Streamlit-Prototyp befindet sich unter:

```text
prototype/app.py
```

Nach dem Training und Export des finalen Modells kann das Dashboard mit folgendem Befehl gestartet werden:

```bash
streamlit run prototype/app.py
```

Für die Ausführung des Dashboards werden die zuvor durch `save_final_model.py` erzeugten Modell- und Vorverarbeitungsdateien benötigt. Diese werden lokal im Verzeichnis `prototype/model/` generiert und sind nicht Bestandteil des Repositories.

## Projektstruktur

```text
.
├── churn_analysis.py
├── evaluate_model.py
├── ml_comparison.py
├── save_final_model.py
├── shap_analysis.py
├── requirements.txt
├── data/                       # lokal, enthält Churn_Modelling.csv
├── eda_results/                # Ergebnisse der explorativen Datenanalyse
├── ml_results/                 # Ergebnisse des Modellvergleichs
├── shap_results/               # Ergebnisse der SHAP-Analyse
└── prototype/
    ├── app.py
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── model/                  # lokal erzeugte Modell-Dateien
```

## Status

Dieses Repository enthält den aktuellen technischen Prototyp des Projekts.

Die Datenanalyse, der Vergleich der Klassifikationsmodelle, das Training des ausgewählten XGBoost-Modells, die Modellevaluation, die SHAP-Analyse sowie die Integration in das Streamlit-Dashboard sind im aktuellen Prototyp umgesetzt.

Die Implementierung befindet sich im Rahmen der Bachelorarbeit in Weiterentwicklung und kann auf Grundlage methodischer Anforderungen sowie des Feedbacks der Betreuungsperson weiter angepasst werden.
