import pandas as pd
import ast
import unicodedata

# Daten laden
df_task2_annotation = pd.read_pickle(
    "./annotation/analysis/survey/export/task_2_annotation_survey.pkl"
)

focus_feature = "feature_six"
df_sorted = df_task2_annotation.sort_values(["item_id", "annotator"]).reset_index(drop=True)
df_sorted[f"agreed_{focus_feature}"] = None

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
        triple = tuple(triple)
    def norm(x):
        if not isinstance(x, str):
            return x
        s = fix_encoding(x)
        s = unicodedata.normalize('NFC', s)
        s = s.strip().lower()
        s = " ".join(s.split())
        return s
    return tuple(norm(x) for x in triple)

for item in df_sorted["item_id"].unique():
    idx = df_sorted[df_sorted["item_id"] == item].index
    feature_sets = []

    for i in idx:
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
        normalized_set = set(normalize_triple(x) for x in val)
        feature_sets.append(normalized_set)

    # Schnittmenge aller nicht-leeren Mengen
    non_empty_sets = [s for s in feature_sets if s]
    if non_empty_sets:
        common_set = set.intersection(*non_empty_sets)
    else:
        common_set = set()

    # Ergebnis als String für CSV
    agreed_str = "; ".join([str(x) for x in common_set]) if common_set else "*"
    for i in idx:
        df_sorted.loc[i, f"agreed_{focus_feature}"] = agreed_str

# Export
output_cols = ["annotator", "item_id", "text", focus_feature, f"agreed_{focus_feature}"]
df_sorted[output_cols].to_excel(f"./annotation/analysis/survey/export/agreed_{focus_feature}.xlsx", index=False, engine='openpyxl')
df_sorted[output_cols].to_pickle(f"./annotation/analysis/survey/export/agreed_{focus_feature}.pkl")