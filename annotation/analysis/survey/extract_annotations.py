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

ANNOTATOR_ID = 20  # Change this for different annotators

# ==============================================================
# Load Label Studio JSON
# ==============================================================

with open(f'./annotation/analysis/survey/export/survey_annotation_project_{ANNOTATOR_ID}.json', 'r') as f:
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
            .replace(' -> ', '-')
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

output_xlsx = f'./annotation/analysis/survey/export/survey_annotations_project_{ANNOTATOR_ID}_dual.xlsx'
output_pkl = f'./annotation/analysis/survey/export/survey_annotations_project_{ANNOTATOR_ID}_dual.pkl'

df.to_excel(output_xlsx, index=False)
df.to_pickle(output_pkl)

print("\n" + "="*70)
print("SAVED FILES")
print("="*70)
print(f"✅ {output_xlsx}")
print(f"✅ {output_pkl}")
