import pandas as pd
import ast
import unicodedata
import json
import re
import os

# ==============================================================
# CONFIGURATION
# ==============================================================

LABEL_SET = [
    "staatsausgaben","geldpolitik","aufgestaute nachfrage","nachfrage (rest)",
    "lieferkettenprobleme","arbeitskräftemangel","löhne","hohe energiepreise","lebensmittelpreise",
    "wohnraum","angebot (rest)","pandemie","politisches missmanagement","krieg",
    "hohe staatsschulden","steuererhöhungen","preistreiberei","klimawandel","geopolitik",
    "migration","wechselkurse","ökonomische krise","inflation"
]

focus_feature = "feature_four"

# ==============================================================
# Load Data from Multiple Sources
# ==============================================================

print("="*70)
print("LOADING DATA")
print("="*70)

# 1. Load agreement data (existing workflow)
df_task2_annotation = pd.read_pickle(
    "./annotation/analysis/survey/export/task_2_annotation_survey.pkl"
)

df_task1_annotation = pd.read_csv(
    "./annotation/analysis/survey/export/survey_task_1_annotation.csv"
)

df_task1_annotation = df_task1_annotation[["inner_id", "label"]].rename(columns={"inner_id": "item_id"})
df_task2_annotation = pd.merge(df_task2_annotation, df_task1_annotation, on="item_id")

# Filter for causal narratives
df_task2_annotation = df_task2_annotation[
    (df_task2_annotation["label"] == "Gründe der Inflation") | 
    (df_task2_annotation["label"] == "kausales Inflationsnarrativ")
]

print(f"Loaded {df_task2_annotation['item_id'].nunique()} items from agreement data")

# 2. Load Label Studio span data (if available)
span_data_available = {}

for annotator_id in [20, 21]:
    span_file = f'./export/survey_annotations_project_{annotator_id}_dual.pkl'
    
    if os.path.exists(span_file):
        df_spans = pd.read_pickle(span_file)
        span_data_available[annotator_id] = df_spans
        print(f"✅ Loaded span data for Annotator {annotator_id}: {len(df_spans)} items")
    else:
        span_data_available[annotator_id] = None
        print(f"⚠️  No span data found for Annotator {annotator_id}")

# ==============================================================
# Helper Functions
# ==============================================================

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

def get_span_data_for_item(item_id, annotator_id):
    """
    Retrieves span data for a specific item and annotator
    Returns: (Relations_Spans JSON, Entities_Spans JSON) or (None, None)
    """
    if annotator_id not in span_data_available or span_data_available[annotator_id] is None:
        return None, None
    
    df_spans = span_data_available[annotator_id]
    
    # Match by Inner_ID or ID
    match = df_spans[
        (df_spans['Inner_ID'] == item_id) | 
        (df_spans['ID'] == item_id) |
        (df_spans['Inner_ID'] == str(item_id)) |
        (df_spans['ID'] == str(item_id))
    ]
    
    if len(match) > 0:
        row = match.iloc[0]
        return row.get('Relations_Spans'), row.get('Entities_Spans')
    
    return None, None

# ==============================================================
# DUAL EXTRACTION: Agreement + Spans
# ==============================================================

df_sorted = df_task2_annotation.sort_values(["item_id", "annotator"]).reset_index(drop=True)

# OLD COLUMNS (existing)
df_sorted[f"agreed_{focus_feature}"] = None
df_sorted[f"agreement_status"] = None
df_sorted[f"annotator_20_{focus_feature}"] = None
df_sorted[f"annotator_21_{focus_feature}"] = None
df_sorted[f"resolution_{focus_feature}"] = None

# NEW COLUMNS (span data)
df_sorted[f"annotator_20_{focus_feature}_spans"] = None
df_sorted[f"annotator_21_{focus_feature}_spans"] = None
df_sorted[f"annotator_20_entities_spans"] = None
df_sorted[f"annotator_21_entities_spans"] = None
df_sorted[f"agreed_{focus_feature}_spans"] = None  # Agreed span data (if possible)

print("\n" + "="*70)
print("PROCESSING AGREEMENTS AND SPANS")
print("="*70)

for item in df_sorted["item_id"].unique(): 
    idx = df_sorted[df_sorted["item_id"] == item].index
    feature_sets = [] 
    annotator_values = {}  # OLD: String values
    annotator_spans = {}   # NEW: Span data (JSON)
    annotator_entities = {}  # NEW: Entity spans (JSON)

    for i in idx:
        annotator = df_sorted.loc[i, "annotator"]
        val = df_sorted.loc[i, focus_feature]
        
        # ==============================================================
        # OLD METHOD: Parse string annotations
        # ==============================================================
        if not val or (isinstance(val, float) and pd.isna(val)):
            val = []
        elif isinstance(val, str):
            if val.strip() == "*":
                val = []
            else:
                try:
                    val = ast.literal_eval(val)
                except Exception:
                    val = []
        
        val_str = "; ".join([str(x) for x in val]) if val else "*"
        annotator_values[annotator] = val_str
        
        normalized_set = set(normalize_triple(x) for x in val)
        feature_sets.append(normalized_set)
        
        # ==============================================================
        # NEW METHOD: Get span data from Label Studio
        # ==============================================================
        relations_spans, entities_spans = get_span_data_for_item(item, annotator)
        
        annotator_spans[annotator] = relations_spans
        annotator_entities[annotator] = entities_spans

    # ==============================================================
    # Agreement Computation (OLD METHOD)
    # ==============================================================
    
    non_empty_sets = [s for s in feature_sets if s]
    if non_empty_sets:
        common_set = set.intersection(*non_empty_sets) 
    else:
        common_set = set()

    if len(non_empty_sets) > 1:
        all_equal = all(s == non_empty_sets[0] for s in non_empty_sets)
        if all_equal:
            agreement_status = "AGREED"
        else:
            agreement_status = "CONFLICT"
    elif len(non_empty_sets) == 1:
        agreement_status = "SINGLE"
    else:
        agreement_status = "EMPTY"

    agreed_str = "; ".join([str(x) for x in common_set]) if common_set else "*"
    
    # ==============================================================
    # Span Agreement (NEW METHOD)
    # ==============================================================
    
    # If both annotators have span data, try to find agreement
    if 20 in annotator_spans and 21 in annotator_spans:
        spans_20 = annotator_spans[20]
        spans_21 = annotator_spans[21]
        
        # Parse and compare (simplified: just check if both exist)
        if spans_20 and spans_21:
            try:
                relations_20 = json.loads(spans_20)
                relations_21 = json.loads(spans_21)
                
                # Find common relations (by source-target pairs)
                pairs_20 = set((r['source'].lower(), r['target'].lower()) for r in relations_20)
                pairs_21 = set((r['source'].lower(), r['target'].lower()) for r in relations_21)
                
                common_pairs = pairs_20 & pairs_21
                
                # Create agreed span data (use spans from annotator 20 for common pairs)
                agreed_spans = [r for r in relations_20 if (r['source'].lower(), r['target'].lower()) in common_pairs]
                
                agreed_spans_json = json.dumps(agreed_spans, ensure_ascii=False) if agreed_spans else None
            except:
                agreed_spans_json = None
        else:
            agreed_spans_json = None
    else:
        agreed_spans_json = None
    
    # ==============================================================
    # Store Results
    # ==============================================================
    
    ann_20_val = annotator_values.get(20, "*")
    ann_21_val = annotator_values.get(21, "*")
    ann_20_spans = annotator_spans.get(20)
    ann_21_spans = annotator_spans.get(21)
    ann_20_entities = annotator_entities.get(20)
    ann_21_entities = annotator_entities.get(21)
    
    for i in idx:
        # OLD COLUMNS
        df_sorted.loc[i, f"agreed_{focus_feature}"] = agreed_str
        df_sorted.loc[i, f"agreement_status"] = agreement_status
        df_sorted.loc[i, f"annotator_20_{focus_feature}"] = ann_20_val
        df_sorted.loc[i, f"annotator_21_{focus_feature}"] = ann_21_val
        
        if agreement_status == "CONFLICT":
            df_sorted.loc[i, f"resolution_{focus_feature}"] = ""
        else:
            df_sorted.loc[i, f"resolution_{focus_feature}"] = agreed_str
        
        # NEW COLUMNS (SPAN DATA)
        df_sorted.loc[i, f"annotator_20_{focus_feature}_spans"] = ann_20_spans
        df_sorted.loc[i, f"annotator_21_{focus_feature}_spans"] = ann_21_spans
        df_sorted.loc[i, f"annotator_20_entities_spans"] = ann_20_entities
        df_sorted.loc[i, f"annotator_21_entities_spans"] = ann_21_entities
        df_sorted.loc[i, f"agreed_{focus_feature}_spans"] = agreed_spans_json

# ==============================================================
# Statistics
# ==============================================================

print("\n" + "="*70)
print("AGREEMENT STATUS STATISTICS")
print("="*70)
print(df_sorted.groupby("item_id")["agreement_status"].first().value_counts())
print(f"\nItems with conflicts: {(df_sorted.groupby('item_id')['agreement_status'].first() == 'CONFLICT').sum()}")
print(f"Total items: {df_sorted['item_id'].nunique()}")

print("\n" + "="*70)
print("SPAN DATA AVAILABILITY")
print("="*70)
print(f"Items with Annotator 20 spans: {df_sorted.groupby('item_id')[f'annotator_20_{focus_feature}_spans'].first().notna().sum()}")
print(f"Items with Annotator 21 spans: {df_sorted.groupby('item_id')[f'annotator_21_{focus_feature}_spans'].first().notna().sum()}")
print(f"Items with agreed spans: {df_sorted.groupby('item_id')[f'agreed_{focus_feature}_spans'].first().notna().sum()}")

# ==============================================================
# Create Export DataFrame
# ==============================================================

export_rows = []

for item in df_sorted["item_id"].unique():
    item_rows = df_sorted[df_sorted["item_id"] == item]
    row = item_rows.iloc[0]
    status = row["agreement_status"]
    
    export_row = {
        "item_id": row["item_id"],
        "text": row["text"],
        f"annotator_20_{focus_feature}": row[f"annotator_20_{focus_feature}"] if status == "CONFLICT" else "",
        f"annotator_21_{focus_feature}": row[f"annotator_21_{focus_feature}"] if status == "CONFLICT" else "",
        f"agreed_{focus_feature}": row[f"agreed_{focus_feature}"],
        "agreement_status": row["agreement_status"],
        f"manual_resolution_{focus_feature}": ""
    }
    
    # NEW: Add span columns
    export_row[f"annotator_20_{focus_feature}_spans"] = row[f"annotator_20_{focus_feature}_spans"]
    export_row[f"annotator_21_{focus_feature}_spans"] = row[f"annotator_21_{focus_feature}_spans"]
    export_row[f"annotator_20_entities_spans"] = row[f"annotator_20_entities_spans"]
    export_row[f"annotator_21_entities_spans"] = row[f"annotator_21_entities_spans"]
    export_row[f"agreed_{focus_feature}_spans"] = row[f"agreed_{focus_feature}_spans"]
    
    export_rows.append(export_row)

df_export = pd.DataFrame(export_rows)

# ==============================================================
# Export
# ==============================================================

output_file = f"./annotation/analysis/survey/export/agreed_{focus_feature}_dual.xlsx"
output_pkl = f"./annotation/analysis/survey/export/agreed_{focus_feature}_dual.pkl"

df_export.to_excel(output_file, index=False, engine='openpyxl')
df_sorted[["annotator", "item_id", "text", focus_feature, 
           f"agreed_{focus_feature}", 
           f"annotator_20_{focus_feature}", 
           f"annotator_21_{focus_feature}", 
           "agreement_status", 
           f"resolution_{focus_feature}",
           f"annotator_20_{focus_feature}_spans",
           f"annotator_21_{focus_feature}_spans",
           f"agreed_{focus_feature}_spans"]].to_pickle(output_pkl)

print("\n" + "="*70)
print("SAVED FILES")
print("="*70)
print(f"✅ {output_file}")
print(f"✅ {output_pkl}")
print(f"\n  Rows without conflict: {len(df_export[df_export['agreement_status'] != 'CONFLICT'])}")
print(f"  Rows with conflict: {len(df_export[df_export['agreement_status'] == 'CONFLICT'])}")

# ==============================================================
# Create Compatible Format for oversampling.py
# ==============================================================

print("\n" + "="*70)
print("CREATING OVERSAMPLING-COMPATIBLE FORMAT")
print("="*70)

# This mimics the structure your oversampling.py expects
df_compatible = df_export.copy()

# Add manual_resolution logic (same as oversampling.py)
# For now, set to 1 (use agreed) for non-conflicts
df_compatible[f"manual_resolution_{focus_feature}"] = 1
df_compatible.loc[df_compatible["agreement_status"] == "CONFLICT", f"manual_resolution_{focus_feature}"] = ""

# Create feature_four column
conditions = [
    df_compatible[f"manual_resolution_{focus_feature}"] == 1,
    df_compatible[f"manual_resolution_{focus_feature}"] == 20,
    df_compatible[f"manual_resolution_{focus_feature}"] == 21
]
choices = [
    df_compatible[f"agreed_{focus_feature}"],
    df_compatible[f"annotator_20_{focus_feature}"],
    df_compatible[f"annotator_21_{focus_feature}"]
]

import numpy as np
df_compatible[focus_feature] = np.select(conditions, choices, default=np.nan)

# Rename to match oversampling.py expectations
df_compatible = df_compatible.rename(columns={
    focus_feature: "Annotation",
    "text": "Text"
})

# Parse annotations (OLD METHOD)
def parse_annotation(annotation_str):
    if pd.isna(annotation_str) or annotation_str == "" or annotation_str == "*":
        return []
    labels = re.findall(r"'([^']+)'", annotation_str)
    labels = [lbl.strip().lower() for lbl in labels if lbl.strip().lower() in LABEL_SET]
    return list(set(labels))

# Parse annotations from spans (NEW METHOD)
def parse_annotation_from_spans(spans_json):
    if pd.isna(spans_json):
        return []
    try:
        relations = json.loads(spans_json)
        labels = set()
        for rel in relations:
            source = rel['source'].lower().strip()
            target = rel['target'].lower().strip()
            if source in LABEL_SET:
                labels.add(source)
            if target in LABEL_SET:
                labels.add(target)
        return list(labels)
    except:
        return []

df_compatible["Annotation_Events"] = df_compatible["Annotation"].apply(parse_annotation)
df_compatible["Annotation_Events_Spans"] = df_compatible[f"agreed_{focus_feature}_spans"].apply(parse_annotation_from_spans)

# Combined events (union of both methods)
df_compatible["Annotation_Events_Combined"] = df_compatible.apply(
    lambda row: list(set(row["Annotation_Events"]) | set(row["Annotation_Events_Spans"])),
    axis=1
)

# Filter valid samples
df_compatible = df_compatible[
    (df_compatible['Annotation'].notna()) & 
    (df_compatible['Annotation'] != '*')
].copy()

print(f"Filtered to {len(df_compatible)} valid samples")

# Save
output_compatible = "./data/agreed_feature_four_expert_dual.xlsx"
df_compatible.to_excel(output_compatible, index=False)

print("\n" + "="*70)
print("✅ SAVED OVERSAMPLING-COMPATIBLE FILE")
print("="*70)
print(f"File: {output_compatible}")
print("\nThis file is a DROP-IN REPLACEMENT for:")
print("  './data/agreed_feature_four_expert.xlsx'")
print("\nNew columns available:")
print(f"  - agreed_{focus_feature}_spans: Agreed span data (JSON)")
print(f"  - annotator_20_{focus_feature}_spans: Annotator 20 spans (JSON)")
print(f"  - annotator_21_{focus_feature}_spans: Annotator 21 spans (JSON)")
print("  - Annotation_Events_Spans: Events from span data")
print("  - Annotation_Events_Combined: Union of old + span methods")

# ==============================================================
# Label Distribution Comparison
# ==============================================================

from collections import Counter

print("\n" + "="*70)
print("LABEL DISTRIBUTION COMPARISON")
print("="*70)

counts_old = Counter([lbl for labels in df_compatible["Annotation_Events"] for lbl in labels])
counts_new = Counter([lbl for labels in df_compatible["Annotation_Events_Spans"] for lbl in labels])
counts_combined = Counter([lbl for labels in df_compatible["Annotation_Events_Combined"] for lbl in labels])

comparison_df = pd.DataFrame({
    "Label": sorted(set(counts_old.keys()) | set(counts_new.keys())),
    "Old Method": [counts_old.get(lbl, 0) for lbl in sorted(set(counts_old.keys()) | set(counts_new.keys()))],
    "Span Method": [counts_new.get(lbl, 0) for lbl in sorted(set(counts_old.keys()) | set(counts_new.keys()))],
    "Combined": [counts_combined.get(lbl, 0) for lbl in sorted(set(counts_old.keys()) | set(counts_new.keys()))]
})

print(comparison_df.to_string(index=False))

print("\n" + "="*70)
print("✅ EXTRACTION COMPLETE")
print("="*70)