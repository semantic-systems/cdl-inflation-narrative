import pandas as pd
import ast
import unicodedata

# Versuche explizit encoding='utf-8' oder encoding='latin1'
df_task2_annotation = pd.read_csv(
    "./annotation/analysis/survey/export/task_2_annotation_survey.csv",
    encoding='latin1'
)

focus_feature = "feature_six"
df_sorted = df_task2_annotation.sort_values(["item_id", "annotator"]).reset_index(drop=True)
df_sorted[f"agreed_{focus_feature}"] = None

def fix_encoding(s):
    if not isinstance(s, str):
        return s
    # Ersetze häufige Encoding-Fehler
    return (s.replace('Ã–', 'Ö')
             .replace('Ã¤', 'ä')
             .replace('Ã¶', 'ö')
             .replace('Ã¼', 'ü')
             .replace('ÃŸ', 'ß')
             .replace('Ã„', 'Ä')
             .replace('Ãœ', 'Ü')
             .replace('Ã–', 'Ö'))

def normalize_triple(triple):
    if isinstance(triple, list):
        triple = tuple(triple)
    def norm(x):
        if not isinstance(x, str):
            return x
        s = fix_encoding(x)
        s = unicodedata.normalize('NFC', s)
        s = s.strip().lower()
        s = " ".join(s.split())  # entfernt doppelte/unsichtbare Whitespaces
        return s
    return tuple(norm(x) for x in triple)

for item in df_sorted["item_id"].unique():
    idx = df_sorted[df_sorted["item_id"] == item].index
    feature_sets = []

    for i in idx:
        val = df_sorted.loc[i, focus_feature]
        if not val or (isinstance(val, float) and pd.isna(val)):
            val = []
        elif isinstance(val, str):
            val = ast.literal_eval(val)
        normalized_set = set(normalize_triple(x) for x in val)
        feature_sets.append(normalized_set)

    for j, s in enumerate(feature_sets):
        print(f"Annotator {j+1} normalized set:")
        for t in s:
            print("  ", repr(t), [ord(c) for c in t[0]])

    non_empty_sets = [s for s in feature_sets if s]
    if non_empty_sets:
        common_set = set.intersection(*non_empty_sets)
    else:
        common_set = set()

    print("Common set:", repr(common_set))

    for i in idx:
        df_sorted.loc[i, f"agreed_{focus_feature}"] = common_set if common_set else "*"

output_cols = ["annotator", "item_id", "text", focus_feature, f"agreed_{focus_feature}"]
df_sorted[output_cols].to_csv(f"./export/agreed_{focus_feature}.csv", index=False)
print(f"Saved CSV with normalized agreed-overlap for {focus_feature}.")


df_sorted[df_sorted["agreed_feature_six"].notna()].tail(20)