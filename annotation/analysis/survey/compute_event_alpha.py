#!/usr/bin/env python3
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import krippendorff

def recompute_event_set(triples_label_form):
    # triples_label_form expected as list of triples: (subj_label, relation_label, obj_label)
    if not triples_label_form or triples_label_form == "*":
        return set()
    events = set()
    for triple in triples_label_form:
        try:
            events.add(triple[0])
            events.add(triple[2])
        except Exception:
            continue
    events.discard("Inflation")
    return events


def get_surface_form_from_id(result_id, results):
    return [result["value"]["text"] for result in results if result.get("id", None) == result_id][0]


def get_label_from_id(result_id, results):
    return [result["value"]["labels"][0] for result in results if result.get("id", None) == result_id][0]


def get_triples(results):
    relation_triples = [result for result in results if result.get("type") == "relation"]
    subjects = [relation_triple["from_id"] for relation_triple in relation_triples]
    objects = [relation_triple["to_id"] for relation_triple in relation_triples]
    relations = [relation_triple.get("labels", []) for relation_triple in relation_triples]
    triples = []
    for i in range(len(subjects)):
        triples.append((subjects[i], relations[i], objects[i]))

    # triples in label form
    triples_label_form = []
    for triple in triples:
        subj_label_form = get_label_from_id(triple[0], results)
        obj_label_form = get_label_from_id(triple[2], results)
        relation_label_form = triple[1]
        triples_label_form.append((subj_label_form, relation_label_form, obj_label_form))

    return triples, None, triples_label_form


def build_common_item_list(df, annotators):
    item_counts = df.groupby("item_id")["annotator"].nunique()
    common_items = item_counts[item_counts == len(annotators)].index.tolist()
    return sorted(common_items)


def main(save_path="./annotation/analysis/survey/export/event_alpha_krippendorff.json"):
    # Load label-studio export JSONs for project ids 20 and 21 from ./export/
    project_ids = [20, 21]
    project_annotations = []
    for pid in project_ids:
        path = Path(f"./annotation/analysis/survey/export/annotation_task2_project_{pid}.json")
        if not path.exists():
            print(f"Missing export file: {path}")
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as f:
            project_annotations.extend(json.load(f))

    # Build dataframe like in task_2_agreement.py
    inner_id = [ann.get("inner_id") for ann in project_annotations]
    text = [ann.get("data", {}).get("text") for ann in project_annotations]
    project_id_col = [ann.get("project") for ann in project_annotations]
    results = [ann.get("annotations", [None])[0].get("result") for ann in project_annotations]

    task2_annotation_dict = {"annotator": [], "item_id": [], "text": [], "triples_label_form": []}
    for i, result in enumerate(results):
        task2_annotation_dict["annotator"].append(project_id_col[i])
        task2_annotation_dict["item_id"].append(inner_id[i])
        task2_annotation_dict["text"].append(text[i])
        triples, _, triples_label_form = get_triples(result)
        task2_annotation_dict["triples_label_form"].append(triples_label_form)

    df = pd.DataFrame.from_dict(task2_annotation_dict)

    # Recompute event sets from triples_label_form to ignore previous event_alpha
    df = df.copy()
    df["_event_set"] = df["triples_label_form"].apply(recompute_event_set)

    annotators = sorted(df["annotator"].unique().tolist())
    item_ids = build_common_item_list(df, annotators)
    if not item_ids:
        print("No items annotated by all annotators found. Exiting.")
        sys.exit(1)

    # Collect all events
    all_events = sorted({e for s in df["_event_set"] for e in s if s})

    # count occurences of each event across all annotators and items
    event_counts = {event: 0 for event in all_events}
    print("Event counts across all annotations:")
    for event in all_events:
        count = sum(event in evset for evset in df["_event_set"])
        event_counts[event] = count
        print(f"  {event}: {count} occurrences")
    
    # remove all that occur less then 20 times to avoid unreliable alpha estimates
    all_events = [event for event in all_events if event_counts[event] >= 20]

    results = {"per_event": {}, "overall_stacked": None, "meta": {"n_annotators": len(annotators), "n_items": len(item_ids), "n_events": len(all_events)}}

    # Map (annotator, item) -> event set for quick lookup
    lookup = {}
    for ann in annotators:
        ann_df = df[df["annotator"] == ann].set_index("item_id")
        lookup[ann] = {}
        for item in item_ids:
            if item in ann_df.index:
                lookup[ann][item] = ann_df.at[item, "_event_set"]
            else:
                lookup[ann][item] = set()

    # Compute per-event alpha (binary 1/0, missing if item absent or marked as empty)
    mats = []
    for event in all_events:
        mat = np.full((len(annotators), len(item_ids)), np.nan)
        for i, ann in enumerate(annotators):
            for j, item in enumerate(item_ids):
                evset = lookup[ann].get(item, set())
                # treat empty set as explicit absence (0). If you prefer to mark as missing, change to np.nan
                mat[i, j] = 1.0 if event in evset else 0.0
        try:
            alpha = krippendorff.alpha(reliability_data=mat, level_of_measurement="nominal")
        except Exception as e:
            alpha = None
        results["per_event"][event] = {"alpha": alpha, "n_items": int(np.sum(~np.isnan(mat[0, :]))) }
        mats.append(mat)

    # Build overall stacked matrix by concatenating event matrices along columns
    if mats:
        stacked = np.concatenate(mats, axis=1)
        try:
            overall_alpha = krippendorff.alpha(reliability_data=stacked, level_of_measurement="nominal")
        except Exception:
            overall_alpha = None
        results["overall_stacked"] = overall_alpha


    print(f"Computed Krippendorff's alpha of {overall_alpha} for {len(all_events)} events across {len(item_ids)} items and {len(annotators)} annotators.")
    print("Per-event alphas:")
    for event, data in results["per_event"].items():
        print(f"  {event}: {data['alpha']} ({data['n_items']} items)")

    # Ensure export directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {save_path}")
    print(f"Events: {len(all_events)} | Annotators: {len(annotators)} | Items: {len(item_ids)}")


if __name__ == "__main__":
    # Linear execution: load exports for projects 20 and 21 and compute alphas
    main()
