import pandas as pd
import json
import re
import os
import ast
import numpy as np

os.getcwd()

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

ANNOTATOR_ID = 21  # Change this for different annotators

# ==============================================================
# Load Label Studio JSON
# ==============================================================

with open(f'./export/survey_annotation_project_{ANNOTATOR_ID}.json', 'r') as f:
    data = json.load(f)
    
print(f"Loaded {len(data)} items from Label Studio")

# ==============================================================
# DUAL EXTRACTION: Old Format + Span Format
# ==============================================================

rows = []

for item in data:
    inner_id = item.get("inner_id")
    text = item.get("data", {}).get("text", "")
    ID = item.get("data", {}).get("id", "")

    first_stage = None
    second_stage_labels = []
    uncertainty = None
    
    # OLD FORMAT: Simple string relations
    relation_edges_old = []  # "Geldpolitik -> Inflation"
    
    # NEW FORMAT: Relations with spans
    relation_edges_spans = []  # Full span information
    entity_spans = []  # All entities with positions

    all_annots = item.get("annotations", [])

    for annot in all_annots:
        results = annot.get("result", [])
        
        # ==============================================================
        # STEP 1: Build entity map with span information
        # ==============================================================
        
        id_to_entity = {}  # Maps result-id -> {label, start, end, text}
        
        for res in results:
            if res.get("type") == "labels":
                rid = res.get("id")
                value = res.get("value", {})
                labels = value.get("labels", []) or []
                
                # Extract span positions (NEW)
                start = value.get("start")
                end = value.get("end")
                text_span = value.get("text")
                
                if rid and labels:
                    label_name = "+".join(labels)
                    
                    # Store entity with span info (NEW)
                    id_to_entity[rid] = {
                        "label": label_name,
                        "start": start,
                        "end": end,
                        "text": text_span
                    }
                    
                    # Add to entity_spans list (NEW)
                    if start is not None and end is not None:
                        entity_spans.append({
                            "entity_id": rid,
                            "label": label_name,
                            "start": start,
                            "end": end,
                            "text": text_span
                        })

        # ==============================================================
        # STEP 2: Extract relations (BOTH formats)
        # ==============================================================
        
        for res in results:
            if res.get("type") == "relation" or ("from_id" in res and "to_id" in res):
                from_id = res.get("from_id")
                to_id = res.get("to_id")
                
                if from_id in id_to_entity and to_id in id_to_entity:
                    source_entity = id_to_entity[from_id]
                    target_entity = id_to_entity[to_id]
                    
                    # OLD FORMAT: Simple string
                    relation_edges_old.append(
                        f"{source_entity['label']} -> {target_entity['label']}"
                    )
                    
                    # NEW FORMAT: With spans (only if positions available)
                    if all([
                        source_entity.get('start') is not None,
                        source_entity.get('end') is not None,
                        target_entity.get('start') is not None,
                        target_entity.get('end') is not None
                    ]):
                        relation_edges_spans.append({
                            "source": source_entity["label"],
                            "source_span": source_entity["text"],
                            "source_start": source_entity["start"],
                            "source_end": source_entity["end"],
                            "target": target_entity["label"],
                            "target_span": target_entity["text"],
                            "target_start": target_entity["start"],
                            "target_end": target_entity["end"]
                        })

        # ==============================================================
        # STEP 3: Extract choices and labels (existing logic)
        # ==============================================================
        
        for res in results:
            res_type = res.get("type")
            value = res.get("value", {})

            if res_type == "choices":
                choices = value.get("choices", [])
                if choices:
                    ch = choices[0]
                    m = re.search(r'([0-5])\b', ch)
                    if m:
                        uncertainty = int(m.group(1))
                        continue
                    if first_stage is None:
                        first_stage = ch

            elif res_type == "labels":
                labels = value.get("labels", [])
                second_stage_labels.extend(labels)

    # ==============================================================
    # FORMAT OUTPUTS
    # ==============================================================
    
    # Second stage (unchanged)
    second_stage = ", ".join(sorted(set(second_stage_labels))) if second_stage_labels else None
    
    # OLD FORMAT: Relations as string (for backward compatibility)
    relations_str_old = "; ".join(sorted(set(relation_edges_old))) if relation_edges_old else None
    
    # Clean old format (same as your original code)
    if relations_str_old:
        relations_str_old = (
            relations_str_old
            .replace(r'\s*->\s*', '-', regex=False)
            .replace('; ', ',')
            .strip()
        )
    
    # NEW FORMAT: Relations with spans (JSON)
    relations_with_spans_json = json.dumps(relation_edges_spans, ensure_ascii=False) if relation_edges_spans else None
    
    # NEW FORMAT: All entities with spans (JSON)
    entities_json = json.dumps(entity_spans, ensure_ascii=False) if entity_spans else None

    rows.append({
        "Inner_ID": inner_id,
        "ID": ID,
        "Text": text,
        "Annotation First Stage": first_stage,
        "Annotation Second Stage": second_stage,
        "Uncertainty Measure": uncertainty,
        
        # OLD FORMAT (for backward compatibility)
        "Relations": relations_str_old,  # e.g., "Geldpolitik-Inflation,Krieg-Inflation"
        
        # NEW FORMATS
        "Relations_Spans": relations_with_spans_json,  # Full span info (JSON)
        "Entities_Spans": entities_json  # All entities with spans (JSON)
    })

df = pd.DataFrame(rows)

# ==============================================================
# STATISTICS
# ==============================================================

print("\n" + "="*70)
print("EXTRACTION STATISTICS")
print("="*70)
print(f"Total items: {len(df)}")
print(f"Items with relations (old format): {df['Relations'].notna().sum()}")
print(f"Items with span data (new format): {df['Relations_Spans'].notna().sum()}")

print("\n" + "="*70)
print("EXAMPLE COMPARISON")
print("="*70)

sample_idx = df[df['Relations_Spans'].notna()].index[0] if df['Relations_Spans'].notna().any() else 0
sample = df.iloc[sample_idx]

print(f"\nText:\n{sample['Text'][:200]}...\n")
print(f"OLD FORMAT (Relations):\n{sample['Relations']}\n")

if pd.notna(sample['Relations_Spans']):
    print("NEW FORMAT (Relations_Spans):")
    relations_parsed = json.loads(sample['Relations_Spans'])
    for rel in relations_parsed[:3]:  # Show first 3
        print(f"  {rel['source']:20s} → {rel['target']:20s}")
        print(f"    Source: [{rel['source_start']:3d}, {rel['source_end']:3d}] '{rel['source_span']}'")
        print(f"    Target: [{rel['target_start']:3d}, {rel['target_end']:3d}] '{rel['target_span']}'")

# ==============================================================
# SAVE OUTPUTS
# ==============================================================

output_xlsx = f'./export/survey_annotations_project_{ANNOTATOR_ID}_dual.xlsx'
output_pkl = f'./export/survey_annotations_project_{ANNOTATOR_ID}_dual.pkl'

df.to_excel(output_xlsx, index=False)
df.to_pickle(output_pkl)

print("\n" + "="*70)
print("SAVED FILES")
print("="*70)
print(f"✅ {output_xlsx}")
print(f"✅ {output_pkl}")

# ==============================================================
# CREATE COMPATIBLE FORMAT FOR OVERSAMPLING.PY
# ==============================================================

print("\n" + "="*70)
print("CREATING COMPATIBLE FORMAT FOR OVERSAMPLING.PY")
print("="*70)

# Rename columns to match your existing workflow
df_compatible = df.rename(columns={
    "Inner_ID": "item_id",
    "Text": "text",
    "Relations": "Annotation"  # Use old format for compatibility
})

# Add span data as separate columns (NEW)
df_compatible["Annotation_Spans"] = df["Relations_Spans"]
df_compatible["Entity_Spans"] = df["Entities_Spans"]

# Parse annotations (OLD METHOD - for backward compatibility)
def parse_annotation(annotation_str):
    """OLD METHOD: Extract events from string format"""
    if pd.isna(annotation_str) or annotation_str == "" or annotation_str == "*":
        return []
    
    # Extract all labels from "Label1-Label2,Label3-Label4" format
    labels = re.findall(r'([a-zäöüß\s\(\)]+?)(?:-|,|$)', annotation_str.lower())
    labels = [lbl.strip() for lbl in labels if lbl.strip() and lbl.strip() in LABEL_SET]
    return list(set(labels))

# Parse annotations (NEW METHOD - from spans)
def parse_annotation_from_spans(spans_json):
    """NEW METHOD: Extract events from span data"""
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

# Apply BOTH methods
df_compatible["Annotation_Events"] = df_compatible["Annotation"].apply(parse_annotation)  # OLD
df_compatible["Annotation_Events_Spans"] = df_compatible["Annotation_Spans"].apply(parse_annotation_from_spans)  # NEW

# Merge both methods (union of events found by either method)
df_compatible["Annotation_Events_Combined"] = df_compatible.apply(
    lambda row: list(set(row["Annotation_Events"]) | set(row["Annotation_Events_Spans"])),
    axis=1
)

# Filter valid samples
df_compatible = df_compatible[
    (df_compatible['Annotation'].notna()) & 
    (df_compatible['Annotation'] != '*')
].copy()

print(f"Filtered to {len(df_compatible)} samples with valid annotations")

# Print label distribution (OLD vs NEW vs COMBINED)
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
    "New Method (Spans)": [counts_new.get(lbl, 0) for lbl in sorted(set(counts_old.keys()) | set(counts_new.keys()))],
    "Combined": [counts_combined.get(lbl, 0) for lbl in sorted(set(counts_old.keys()) | set(counts_new.keys()))]
})

print(comparison_df.to_string(index=False))

# Save compatible format
output_compatible = f'./data/agreed_feature_four_annotator_{ANNOTATOR_ID}_dual.xlsx'
df_compatible.to_excel(output_compatible, index=False)

print("\n" + "="*70)
print("✅ SAVED COMPATIBLE FORMAT")
print("="*70)
print(f"File: {output_compatible}")
print(f"Columns: {list(df_compatible.columns)}")
print("\nThis file can be used as drop-in replacement for:")
print("  './data/agreed_feature_four_expert.xlsx'")
print("\nNew columns available:")
print("  - Annotation_Spans: Full span information (JSON)")
print("  - Entity_Spans: All entities with positions (JSON)")
print("  - Annotation_Events_Spans: Events extracted from spans")
print("  - Annotation_Events_Combined: Union of old + new methods")

# ==============================================================
# HELPER FUNCTIONS FOR MTB INTEGRATION
# ==============================================================

print("\n" + "="*70)
print("EXAMPLE: USING SPANS FOR MTB")
print("="*70)

def replace_with_markers_from_spans(text, relation_span_dict):
    """
    MTB-ready: Replace events using exact span positions
    No fuzzy matching needed!
    """
    source_start = relation_span_dict['source_start']
    source_end = relation_span_dict['source_end']
    target_start = relation_span_dict['target_start']
    target_end = relation_span_dict['target_end']
    
    # Sort spans (replace from back to front)
    spans = sorted(
        [(source_start, source_end, '[E1]'), (target_start, target_end, '[E2]')],
        key=lambda x: x[0],
        reverse=True
    )
    
    result = text
    for start, end, marker in spans:
        result = result[:start] + marker + result[end:]
    
    return result

# Example usage
if df_compatible['Annotation_Spans'].notna().any():
    sample_row = df_compatible[df_compatible['Annotation_Spans'].notna()].iloc[0]
    
    print(f"Original Text:\n{sample_row['text'][:200]}...\n")
    
    relations = json.loads(sample_row['Annotation_Spans'])
    
    if relations:
        rel = relations[0]
        marked_text = replace_with_markers_from_spans(sample_row['text'], rel)
        
        print("MTB Format (with markers):")
        print(f"{marked_text[:200]}...\n")
        
        print("This can be used directly in:")
        print("  - Contrastive Pre-Training")
        print("  - Binary Classification")
        print("  - No fuzzy matching errors!")

print("\n" + "="*70)
print("✅ EXTRACTION COMPLETE")
print("="*70)