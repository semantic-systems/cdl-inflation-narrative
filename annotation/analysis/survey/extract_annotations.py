import pandas as pd
import json

for i in (20, 21):
    with open(f'./export/survey_annotation_project_{i}.json', 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data['annotations'])
    print(f"Project {i} has {len(df)} annotations")
    print(df.head())

with open(f'./export/survey_annotation_project_{20}.json', 'r') as f:
    data = json.load(f)
    
print(type(data))        # See what type it is
print(len(data))         # See how many items/keys it has
print(list(data)[:5])    # See the first few keys or list items


import json
import pandas as pd
import re

# Load your JSON file
with open(f'./export/survey_annotation_project_{20}.json', 'r') as f:
    data = json.load(f)




import json
import pandas as pd
import re

# assumes project_id is defined and data already loaded into `data`
# with open(f'./export/survey_annotation_project_{project_id}.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

rows = []
for item in data:
    inner_id = item.get("inner_id")
    text = item.get("data", {}).get("text", "")

    first_stage = None
    second_stage_labels = []
    uncertainty = None  # 0–5
    relation_edges = []  # will hold strings like "Geldpolitik -> Inflation"

    # Use only final annotations (as you had it)
    all_annots = item.get("annotations", [])

    for annot in all_annots:
        results = annot.get("result", [])

        # 1) First pass: build a map from result-id -> label name(s)
        # Some results have multiple labels; we'll join them with '+'.
        id_to_label = {}
        for res in results:
            if res.get("type") == "labels":
                rid = res.get("id")
                labels = res.get("value", {}).get("labels", []) or []
                if rid and labels:
                    # Join multiple labels to a single node name, e.g., "Geldpolitik+Inflation"
                    id_to_label[rid] = "+".join(labels)

        # 2) Second pass: collect relations using the map above
        for res in results:
            # Label Studio relation entries usually look like this:
            # {'from_id': '<rid1>', 'to_id': '<rid2>', 'type': 'relation', 'direction': 'right'}
            if res.get("type") == "relation" or ("from_id" in res and "to_id" in res):
                from_id = res.get("from_id")
                to_id = res.get("to_id")
                if from_id in id_to_label and to_id in id_to_label:
                    relation_edges.append(f"{id_to_label[from_id]} -> {id_to_label[to_id]}")

        # 3) Existing extraction (choices/labels/uncertainty)
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
                        continue  # don't treat as first-stage
                    if first_stage is None:
                        first_stage = ch

            elif res_type == "labels":
                labels = value.get("labels", [])
                second_stage_labels.extend(labels)

    second_stage = ", ".join(sorted(set(second_stage_labels))) if second_stage_labels else None
    relations_str = "; ".join(sorted(set(relation_edges))) if relation_edges else None

    rows.append({
        "Inner_ID": inner_id,
        "Text": text,
        "Annotation First Stage": first_stage,
        "Annotation Second Stage": second_stage,
        "Uncertainty Measure": uncertainty,
        "Relations": relations_str
    })

df = pd.DataFrame(rows)

# quick sanity checks
print(df[['Inner_ID','Relations']].head(10))
# df.to_csv("annotations_with_relations.csv", index=False)

df['Relations'] = (
    df['Relations']
      .str.replace(r'\s*->\s*', '-', regex=True)
      .str.replace(r'\s*;\s*', ',', regex=True)
      .str.strip() # remove leading/trailing whitespace
)
# ergibt z.B. "Angebot (Rest)-Inflation,Hohe Energiepreise-Inflation"


# Ensure no NaN in Relations column

df.to_excel('./export/survey_annotations_project_20.xlsx', index=False)



