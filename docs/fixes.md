# V3 Findings und Fixes

## Findings

### 1. Lag-Features werden im Ranking nicht konsistent genutzt

Das Share-Modell wird mit Lag-Features trainiert. Im eigentlichen Top-K-Ranking werden diese Features aber teilweise nicht befuellt und fallen dadurch auf 0 zurueck.

Auswirkung:

- Die Share-Validation-Metriken und die Top-K-Backtest-Metriken messen nicht exakt dieselbe Feature-Situation.
- Die Preview nach dem finalen Training kann schlechter oder anders ranken als erwartet.
- Das Modell kann im Training Signale lernen, die ihm beim Ranking nicht zur Verfuegung stehen.

Fix:

- Fuer jedes Ranking-Datum einen Lag-Lookup aus der bis dahin bekannten Historie bauen.
- Diesen Lag-Lookup konsequent an die Ranking-Funktion uebergeben.
- Das fuer Backtest, Preview und spaetere Prediction identisch behandeln.

### 2. Prediction-Lags nutzen ein anderes Datenuniversum als das Training

Das Training basiert auf gefilterten Orten, validen Kategorien und exakten Zeitangaben. Bei Prediction mit uebergebenen Rohdaten koennen die Lag-Features aber aus ungefilterten Daten gebaut werden.

Auswirkung:

- Meta-Orte, unsichere Zeiten oder Kategorien ausserhalb des Trainingsraums koennen in die Lag-Features geraten.
- Das erzeugt einen Serving-Skew: Training und Prediction sehen strukturell unterschiedliche Daten.

Fix:

- Vor dem Bau der Prediction-Lags dieselben Filter anwenden wie im Training.
- Also Orte filtern, valide Kategorien anwenden und nur exakte Zeitangaben fuer die bucket-basierten Lags verwenden.
- Idealerweise eine gemeinsame Helper-Funktion fuer den V3-Trainingsdatensatz verwenden, damit Training und Serving nicht auseinanderlaufen.

### 3. Kategorie-Encoder kann Klassen enthalten, die im Train-Split fehlen

Die validen Kategorien werden auf dem Gesamtdatensatz bestimmt. Das Backtest-Share-Modell wird aber nur auf dem Train-Split trainiert. Seltene Kategorien koennen dadurch im Encoder vorhanden sein, ohne im Train-Split wirklich trainiert worden zu sein.

Auswirkung:

- Die Anzahl der Modell-Proba-Spalten kann im Randfall nicht sauber zur Encoder-Klassenliste passen.
- Priors und Modellwahrscheinlichkeiten koennen dann falsch ausgerichtet werden.

Fix:

- Fuer den Backtest sicherstellen, dass alle Encoder-Klassen im Train-Split vorkommen.
- Alternativ die Backtest-Kategorien aus dem Train-Split ableiten und Validation-Events unbekannter Kategorien separat behandeln.
- Vor dem Blending eine harte Assertion einbauen: Anzahl Modell-Proba-Spalten muss der Anzahl Encoder-Klassen entsprechen.
