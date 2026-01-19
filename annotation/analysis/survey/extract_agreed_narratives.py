import pandas as pd
import ast
import unicodedata

# Daten laden
df_task2_annotation = pd.read_pickle(
    "./annotation/analysis/survey/export/task_2_annotation_survey.pkl"
)

df_task1_annotation = pd.read_csv(
    "./annotation/analysis/survey/export/survey_task_1_annotation.csv"
)

df_task1_annotation = df_task1_annotation[["inner_id", "label"]].rename(columns={"inner_id": "item_id"})

df_task2_annotation = pd.merge(df_task2_annotation, df_task1_annotation, on="item_id")
df_task2_annotation.columns

# Remove all answers that are not related to causal inflation narratives

df_task2_annotation["item_id"].nunique()
df_task2_annotation['label'].unique()

df_task2_annotation = df_task2_annotation[(df_task2_annotation["label"] == "Gründe der Inflation") | (df_task2_annotation["label"] == "kausales Inflationsnarrativ")]
df_task2_annotation["item_id"].nunique()


# select focus feature

focus_feature = "feature_four" 
df_sorted = df_task2_annotation.sort_values(["item_id", "annotator"]).reset_index(drop=True)
df_sorted[f"agreed_{focus_feature}"] = None
df_sorted[f"agreement_status"] = None  # new column for agreement status
df_sorted[f"annotator_20_{focus_feature}"] = None  # Annotation Annotator 20
df_sorted[f"annotator_21_{focus_feature}"] = None  # Annotation Annotator 21
df_sorted[f"resolution_{focus_feature}"] = None  # Manual resolution column

def fix_encoding(s):
    if not isinstance(s, str):
        return s
    return (s.replace('Ã–', 'Ö')
             .replace('Ã¤', 'ä')
             .replace('Ã¶', 'ö')
             .replace('Ã¼', 'ü')
             .replace('ÃŸ', 'ß')
             .replace('Ã„', 'Ä')
             .replace('Ãœ', 'Ü'))

def normalize_triple(triple):
    if isinstance(triple, list):
        triple = tuple(triple) # in tuple umwandeln, falls Liste
    def norm(x): # Normalisierung einer Zeichenkette
        if not isinstance(x, str):
            return x
        s = fix_encoding(x) 
        s = unicodedata.normalize('NFC', s)
        s = s.strip().lower()
        s = " ".join(s.split())
        return s
    return tuple(norm(x) for x in triple)

for item in df_sorted["item_id"].unique(): 
    idx = df_sorted[df_sorted["item_id"] == item].index # Indizes der Zeilen für dieses Item
    feature_sets = [] 
    annotator_values = {}  # Speichere die Originalwerte pro Annotator

    for i in idx:
        annotator = df_sorted.loc[i, "annotator"]
        val = df_sorted.loc[i, focus_feature] 
        # Leere oder ungültige Werte abfangen
        if not val or (isinstance(val, float) and pd.isna(val)):
            val = []
        elif isinstance(val, str):
            # Falls val ein Platzhalter wie "*" ist, überspringen
            if val.strip() == "*":
                val = []
            else:
                try:
                    val = ast.literal_eval(val)
                except Exception:
                    val = []
        
        # Speichere Originalwert für diesen Annotator
        val_str = "; ".join([str(x) for x in val]) if val else "*"
        annotator_values[annotator] = val_str
        
        normalized_set = set(normalize_triple(x) for x in val)
        feature_sets.append(normalized_set)

    # Schnittmenge aller nicht-leeren Mengen
    non_empty_sets = [s for s in feature_sets if s]
    if non_empty_sets:
        common_set = set.intersection(*non_empty_sets) 
    else:
        common_set = set()

    # Prüfe, ob alle Annotationen übereinstimmen (100% Agreement)
    # Wenn alle nicht-leeren Sets identisch sind
    if len(non_empty_sets) > 1:
        # Prüfe ob alle Sets gleich sind
        all_equal = all(s == non_empty_sets[0] for s in non_empty_sets)
        if all_equal:
            agreement_status = "AGREED"  # 100% Übereinstimmung
        else:
            agreement_status = "CONFLICT"  # Konflikt - manuelle Prüfung erforderlich
    elif len(non_empty_sets) == 1:
        agreement_status = "SINGLE"  # Nur eine Annotation vorhanden
    else:
        agreement_status = "EMPTY"  # Keine Annotationen

    # Ergebnis als String für CSV
    agreed_str = "; ".join([str(x) for x in common_set]) if common_set else "*"
    
    # Werte der einzelnen Annotatoren speichern
    ann_20_val = annotator_values.get(20, "*")
    ann_21_val = annotator_values.get(21, "*")
    
    for i in idx:
        df_sorted.loc[i, f"agreed_{focus_feature}"] = agreed_str
        df_sorted.loc[i, f"agreement_status"] = agreement_status
        df_sorted.loc[i, f"annotator_20_{focus_feature}"] = ann_20_val
        df_sorted.loc[i, f"annotator_21_{focus_feature}"] = ann_21_val
        # Bei CONFLICT bleibt resolution leer für manuelle Eingabe
        if agreement_status == "CONFLICT":
            df_sorted.loc[i, f"resolution_{focus_feature}"] = ""
        else:
            df_sorted.loc[i, f"resolution_{focus_feature}"] = agreed_str

df_sorted[["annotator", focus_feature, f"agreed_{focus_feature}"]].head(10)
df_sorted[["annotator", focus_feature, f"agreed_{focus_feature}"]].tail(10)
df_sorted['item_id'].unique()
# Zeige Konflikt-Statistiken
print("\n=== Agreement Status Statistik ===")
print(df_sorted.groupby("item_id")["agreement_status"].first().value_counts())
print(f"\nAnzahl Items mit Konflikten: {(df_sorted.groupby('item_id')['agreement_status'].first() == 'CONFLICT').sum()}")
print(f"Anzahl Items insgesamt: {df_sorted['item_id'].nunique()}")

# Erstelle Export-DataFrame
# Bei Konflikten: Beide Annotator-Zeilen behalten
# Bei Agreement/Single/Empty: Nur eine Zeile pro Item

export_rows = []

for item in df_sorted["item_id"].unique():
    item_rows = df_sorted[df_sorted["item_id"] == item]
    row = item_rows.iloc[0]  # Immer nur erste Zeile nehmen
    status = row["agreement_status"]
    
    if status == "CONFLICT":
        # Bei Konflikt: Beide Annotator-Spalten befüllen
        export_rows.append({
            "item_id": row["item_id"],
            "text": row["text"],
            f"annotator_20_{focus_feature}": row[f"annotator_20_{focus_feature}"],
            f"annotator_21_{focus_feature}": row[f"annotator_21_{focus_feature}"],
            f"agreed_{focus_feature}": row[f"agreed_{focus_feature}"],
            "agreement_status": row["agreement_status"],
            f"manual_resolution_{focus_feature}": ""  # Leere Spalte für manuelle Eingabe
        })
    else:
        # Bei Agreement/Single/Empty: Annotator-Spalten leer lassen
        export_rows.append({
            "item_id": row["item_id"],
            "text": row["text"],
            f"annotator_20_{focus_feature}": "",  # Leer bei Agreement
            f"annotator_21_{focus_feature}": "",  # Leer bei Agreement
            f"agreed_{focus_feature}": row[f"agreed_{focus_feature}"],
            "agreement_status": row["agreement_status"],
            f"manual_resolution_{focus_feature}": ""  # Leer
        })

df_export = pd.DataFrame(export_rows)

# Export
output_file = f"./annotation/analysis/survey/export/agreed_{focus_feature}.xlsx"
df_export.to_excel(output_file, index=False, engine='openpyxl')
print(f"\nExcel-Datei gespeichert: {output_file}")
print(f"  - Zeilen ohne Konflikt: {len(df_export[df_export['agreement_status'] != 'CONFLICT'])}")
print(f"  - Zeilen mit Konflikt: {len(df_export[df_export['agreement_status'] == 'CONFLICT'])}")

# Pickle speichern
df_sorted[["annotator", "item_id", "text", focus_feature, f"agreed_{focus_feature}", 
           f"annotator_20_{focus_feature}", f"annotator_21_{focus_feature}", 
           "agreement_status", f"resolution_{focus_feature}"]].to_pickle(
    f"./annotation/analysis/survey/export/agreed_{focus_feature}.pkl"
)