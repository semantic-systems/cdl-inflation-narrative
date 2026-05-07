import json
from collections import Counter
from itertools import combinations
from pathlib import Path
import pandas as pd
import numpy as np
import krippendorff



def setup():
    if not Path("./export").exists():
        Path("./export").mkdir()


def get_task_2_annotation_json(project_id_list):
    project_annotations = []
    for project_id in project_id_list:
        path = Path(f'./export/annotation_task2_project_{project_id}.json')
        if not path.exists():
            raise FileNotFoundError(f"Local export not found: {path}")
        with open(path, "r") as f:
            project_annotations.extend(json.load(f))
    return project_annotations


def get_surface_form_from_id(result_id, results):
    return [result["value"]["text"] for result in results if result.get("id", None) == result_id][0]


def get_label_from_id(result_id, results):
    return [result["value"]["labels"][0] for result in results if result.get("id", None) == result_id][0]


def get_triples(results):
    relation_triples = [result for result in results if result["type"] == "relation"]
    subjects = [relation_triple["from_id"] for relation_triple in relation_triples]
    objects = [relation_triple["to_id"] for relation_triple in relation_triples]
    relations = [relation_triple.get("labels", []) for relation_triple in relation_triples]
    triples = []
    for i in range(len(subjects)):
        triples.append((subjects[i], relations[i], objects[i]))

    triples_surface_form = []
    for triple in triples:
        subj_surface_form = get_surface_form_from_id(triple[0], results)
        obj_surface_form = get_surface_form_from_id(triple[2], results)
        relation_surface_form = triple[1]
        triples_surface_form.append((subj_surface_form, relation_surface_form, obj_surface_form))

    triples_label_form = []
    event_remapping = {"Russia-Ukraine War": "War", "Energy Crisis": "Energy Prices", "House Costs": "Housing Costs"}
    for triple in triples:
        subj_label_form = get_label_from_id(triple[0], results)
        obj_label_form = get_label_from_id(triple[2], results)
        subj_label_form = event_remapping.get(subj_label_form, subj_label_form)
        obj_label_form = event_remapping.get(obj_label_form, obj_label_form)
        relation_label_form = triple[1]
        triples_label_form.append((subj_label_form, relation_label_form, obj_label_form))

    return triples, triples_surface_form, triples_label_form


def extract_subject_labels(triples_label_form):
    return {triple[0] for triple in triples_label_form if triple[0] != "Inflation"}


def extract_inflation_linked_subject_labels(triples_label_form):
    """All non-Inflation labels that have a directed path to Inflation in the DAG."""
    parent_map = {}
    for subject_label, _, object_label in triples_label_form: # subject label: directed parent of object label, object label: directed child of subject label
        parent_map.setdefault(object_label, set()).add(subject_label) # build parent map for DAG traversal

    inflation_linked_labels = set() # extract inflation linked labels
    stack = list(parent_map.get("Inflation", set())) # start from direct parents of Inflation
    while stack:
        label = stack.pop() # get next label to process
        if label == "Inflation" or label in inflation_linked_labels: 
            continue
        inflation_linked_labels.add(label)
        stack.extend(parent_map.get(label, set()))

    return inflation_linked_labels


def extract_object_labels(triples_label_form):
    return {triple[2] for triple in triples_label_form if triple[2] != "Inflation"}


def compute_label_alpha(df, annotator_list, label_side):
    all_item_ids = sorted(df["item_id"].unique())
    all_labels = sorted({label for labels in df[label_side] for label in labels})

    results = {}
    for label in all_labels:
        matrix = []
        for annotator in annotator_list:
            annotator_df = df[df["annotator"] == annotator].set_index("item_id")
            row = []
            for item_id in all_item_ids:
                if item_id not in annotator_df.index:
                    row.append(np.nan)
                else:
                    label_set = annotator_df.loc[item_id, label_side]
                    row.append(1 if label in label_set else 0)
            matrix.append(row)

        try:
            alpha = krippendorff.alpha(
                reliability_data=np.array(matrix, dtype=float),
                level_of_measurement="nominal",
            )
            results[label] = float(alpha) if np.isfinite(alpha) else None
        except Exception:
            results[label] = None

    return results


def compute_label_raw_agreement(df, annotator_list, label_side):
    all_item_ids = sorted(df["item_id"].unique())
    all_labels = sorted({label for labels in df[label_side] for label in labels})

    results = {}
    for label in all_labels:
        agree = 0
        total = 0
        for item_id in all_item_ids:
            votes = []
            for annotator in annotator_list:
                annotator_df = df[df["annotator"] == annotator].set_index("item_id")
                if item_id not in annotator_df.index:
                    continue
                label_set = annotator_df.loc[item_id, label_side]
                votes.append(1 if label in label_set else 0)
            if len(votes) == len(annotator_list): # Only consider items annotated by all annotators in the subset
                total += 1
                if len(set(votes)) == 1:
                    agree += 1
        results[label] = agree / total if total > 0 else None

    return results


def compute_overall_alpha(df, annotator_list, label_side):
    all_item_ids = sorted(df["item_id"].unique())
    all_labels = sorted({label for labels in df[label_side] for label in labels})

    full_matrix = []
    for annotator in annotator_list:
        annotator_df = df[df["annotator"] == annotator].set_index("item_id")
        row = []
        for label in all_labels:
            for item_id in all_item_ids:
                if item_id not in annotator_df.index:
                    row.append(np.nan)
                else:
                    label_set = annotator_df.loc[item_id, label_side]
                    row.append(1 if label in label_set else 0)
        full_matrix.append(row)

    try:
        alpha = krippendorff.alpha(
            reliability_data=np.array(full_matrix, dtype=float),
            level_of_measurement="nominal",
        )
        return float(alpha) if np.isfinite(alpha) else None
    except Exception:
        return None


if __name__ == "__main__":
    setup()

    project_id_list = [11, 12, 13, 14]
    project_annotations = get_task_2_annotation_json(project_id_list)

    rows = []
    for ann in project_annotations:
        _, _, triples_label_form = get_triples(ann["annotations"][0]["result"])
        rows.append({
            "annotator": ann["project"],
            "item_id": ann["data"]["inner_id"],
            "text": ann["data"]["text"],
            "triples_label_form": triples_label_form,
        })

    df = pd.DataFrame(rows)
    df["subject_labels"] = df["triples_label_form"].apply(extract_subject_labels)
    df["object_labels"] = df["triples_label_form"].apply(extract_object_labels)
    df["inflation_linked_subject_labels"] = df["triples_label_form"].apply(extract_inflation_linked_subject_labels)

    def mean_alpha(alpha_dict):
        values = [v for v in alpha_dict.values() if v is not None]
        return float(np.mean(values)) if values else None

    subsets = [project_id_list] + [list(c) for c in combinations(project_id_list, len(project_id_list) - 1)]

    alpha_store = {}
    for subset in subsets:
        key = "-".join(str(a) for a in subset)
        subj = compute_label_alpha(df, subset, "subject_labels")
        obj = compute_label_alpha(df, subset, "object_labels")
        subj_raw = compute_label_raw_agreement(df, subset, "subject_labels")
        obj_raw = compute_label_raw_agreement(df, subset, "object_labels")
        alpha_store[key] = {
            "subjects": subj,
            "subjects_mean": compute_overall_alpha(df, subset, "subject_labels"),
            "subjects_raw": subj_raw,
            "subjects_raw_mean": mean_alpha(subj_raw),
            "objects": obj,
            "objects_mean": compute_overall_alpha(df, subset, "object_labels"),
            "objects_raw": obj_raw,
            "objects_raw_mean": mean_alpha(obj_raw),
        }

    out_path = f"./export/alpha-{'-'.join([str(a) for a in project_id_list])}.json"
    with open(out_path, "w") as f:
        json.dump(alpha_store, f)
    print(f"Saved alpha scores to {out_path}")

    print("\nSubject agreement means:")
    for key, val in alpha_store.items():
        alpha_str = f"{val['subjects_mean']:.4f}" if val["subjects_mean"] is not None else "N/A"
        raw_str = f"{val['subjects_raw_mean']:.4f}" if val["subjects_raw_mean"] is not None else "N/A"
        print(f"  {key}: alpha={alpha_str}  raw={raw_str}")

    print("\nAll subject labels by alpha (per setting):")
    for key, val in alpha_store.items():
        scored = [(label, score) for label, score in val["subjects"].items() if score is not None]
        bottom5 = sorted(scored, key=lambda x: x[1])
        print(f"  {key}:")
        for label, score in bottom5:
            raw = val["subjects_raw"].get(label)
            raw_str = f"{raw:.4f}" if raw is not None else "N/A"
            print(f"    {label}: alpha={score:.4f}  raw={raw_str}")

    # --- Merged super-label analysis ---
    LABEL_GROUPS = {
        "Demand": {"Demand Shift", "Demand (residual)", "Pent-up Demand"},
        "Government": {"Government Debt", "Government Spending"},
        "Supply/Supply Chain": {"Supply Chain Issues", "Supply (residual)", "Transportation Costs"},
        "Labor": {"Labor Shortage", "Wages"},
        "Commodity": {"Energy Prices", "Food Prices", "Climate", "Trade Balance", "Exchange Rates"},
        "Pandemic": {"Pandemic"},
        "War": {"War"},
        "Monetary": {"Monetary Policy", "Inflation Expectations", "Base Effect"},
        "Input Costs": {"Housing Costs", "Medical Costs", "Education Costs"},
        "Taxation": {"Tax Increases"},
        "Market Power": {"Price-Gouging"},
        "Mismanagement": {"Mismanagement"},
    }

    def apply_label_groups(label_set):
        merged = set()
        for label in label_set:
            group = next((g for g, members in LABEL_GROUPS.items() if label in members), label)
            merged.add(group)
        return merged

    df["subject_labels_merged"] = df["subject_labels"].apply(apply_label_groups)

    print("\n--- Merged super-label subject alpha ---")
    merged_alpha_store = {}
    for subset in subsets:
        key = "-".join(str(a) for a in subset)
        subj = compute_label_alpha(df, subset, "subject_labels_merged")
        subj_raw = compute_label_raw_agreement(df, subset, "subject_labels_merged")
        merged_alpha_store[key] = {
            "subjects": subj,
            "subjects_mean": compute_overall_alpha(df, subset, "subject_labels_merged"),
            "subjects_raw": subj_raw,
            "subjects_raw_mean": mean_alpha(subj_raw),
        }

    print("\nMerged subject agreement means:")
    for key, val in merged_alpha_store.items():
        alpha_str = f"{val['subjects_mean']:.4f}" if val["subjects_mean"] is not None else "N/A"
        raw_str = f"{val['subjects_raw_mean']:.4f}" if val["subjects_raw_mean"] is not None else "N/A"
        print(f"  {key}: alpha={alpha_str}  raw={raw_str}")

    print("\nMerged subject agreement per label (per setting):")
    for key, val in merged_alpha_store.items():
        mean_str = f"{val['subjects_mean']:.4f}" if val["subjects_mean"] is not None else "N/A"
        raw_mean_str = f"{val['subjects_raw_mean']:.4f}" if val["subjects_raw_mean"] is not None else "N/A"
        print(f"  {key} (mean alpha: {mean_str}, mean raw: {raw_mean_str}):")
        for label, score in sorted(val["subjects"].items()):
            alpha_str = f"{score:.4f}" if score is not None else "N/A"
            raw = val["subjects_raw"].get(label)
            raw_str = f"{raw:.4f}" if raw is not None else "N/A"
            print(f"    {label}: alpha={alpha_str}  raw={raw_str}")

    # --- Inflation-linked label agreement ---
    print("\n--- Agreement: labels with a directed path to Inflation in the DAG ---")
    inflation_linked_alpha_store = {}
    for subset in subsets:
        key = "-".join(str(a) for a in subset)
        subj = compute_label_alpha(df, subset, "inflation_linked_subject_labels")
        subj_raw = compute_label_raw_agreement(df, subset, "inflation_linked_subject_labels")
        inflation_linked_alpha_store[key] = {
            "subjects": subj,
            "subjects_mean": compute_overall_alpha(df, subset, "inflation_linked_subject_labels"),
            "subjects_raw": subj_raw,
            "subjects_raw_mean": mean_alpha(subj_raw),
        }

    print("\nInflation-linked subject agreement means:")
    for key, val in inflation_linked_alpha_store.items():
        alpha_str = f"{val['subjects_mean']:.4f}" if val["subjects_mean"] is not None else "N/A"
        raw_str = f"{val['subjects_raw_mean']:.4f}" if val["subjects_raw_mean"] is not None else "N/A"
        print(f"  {key}: alpha={alpha_str}  raw={raw_str}")

    print("\nInflation-linked subject agreement per label (per setting):")
    for key, val in inflation_linked_alpha_store.items():
        mean_str = f"{val['subjects_mean']:.4f}" if val["subjects_mean"] is not None else "N/A"
        raw_mean_str = f"{val['subjects_raw_mean']:.4f}" if val["subjects_raw_mean"] is not None else "N/A"
        print(f"  {key} (mean alpha: {mean_str}, mean raw: {raw_mean_str}):")
        for label, score in sorted(val["subjects"].items()):
            alpha_str = f"{score:.4f}" if score is not None else "N/A"
            raw = val["subjects_raw"].get(label)
            raw_str = f"{raw:.4f}" if raw is not None else "N/A"
            print(f"    {label}: alpha={alpha_str}  raw={raw_str}")

    # Also apply merged super-labels to inflation-linked subjects
    df["inflation_linked_subject_labels_merged"] = df["inflation_linked_subject_labels"].apply(apply_label_groups)

    print("\nInflation-linked merged super-label agreement means:")
    inflation_linked_merged_alpha_store = {}
    for subset in subsets:
        key = "-".join(str(a) for a in subset)
        subj = compute_label_alpha(df, subset, "inflation_linked_subject_labels_merged")
        subj_raw = compute_label_raw_agreement(df, subset, "inflation_linked_subject_labels_merged")
        inflation_linked_merged_alpha_store[key] = {
            "subjects": subj,
            "subjects_mean": compute_overall_alpha(df, subset, "inflation_linked_subject_labels_merged"),
            "subjects_raw": subj_raw,
            "subjects_raw_mean": mean_alpha(subj_raw),
        }
        alpha_str = f"{inflation_linked_merged_alpha_store[key]['subjects_mean']:.4f}" if inflation_linked_merged_alpha_store[key]["subjects_mean"] is not None else "N/A"
        raw_str = f"{inflation_linked_merged_alpha_store[key]['subjects_raw_mean']:.4f}" if inflation_linked_merged_alpha_store[key]["subjects_raw_mean"] is not None else "N/A"
        print(f"  {key}: alpha={alpha_str}  raw={raw_str}")

    # --- Error analysis: co-occurring labels for focus labels ---
    FOCUS_LABELS = {"Mismanagement", "Demand Shift", "Inflation Expectations", "Transportation Costs", "Government Debt", "Pandemic",
                    "Labor Shortage", "Climate", "Base Effect"}


    print("\n--- Error analysis: what do others label when one annotator says X? ---")
    all_item_ids = sorted(df["item_id"].unique())

    for focus_label in sorted(FOCUS_LABELS):
        co_label_counts = {}
        for item_id in all_item_ids:
            item_df = df[df["item_id"] == item_id]
            users_with_label = item_df[item_df["subject_labels"].apply(lambda s: focus_label in s)]["annotator"].tolist()
            users_without_label = item_df[item_df["subject_labels"].apply(lambda s: focus_label not in s)]["annotator"].tolist()

            if not users_with_label or not users_without_label:
                continue  # skip if all agree or nobody has it

            for annotator in users_without_label:
                other_labels = item_df[item_df["annotator"] == annotator]["subject_labels"].values[0]
                for lbl in other_labels:
                    co_label_counts[lbl] = co_label_counts.get(lbl, 0) + 1

        print(f"\n  When one annotator says '{focus_label}', others instead used:")
        for lbl, count in sorted(co_label_counts.items(), key=lambda x: -x[1]):
            print(f"    {lbl}: {count}x")

    # --- Save annotated data as CSV ---
    pivot = df.pivot(index="item_id", columns="annotator", values="subject_labels")
    pivot.columns = [f"annotator_{col}" for col in pivot.columns]
    text_df = df.drop_duplicates("item_id").set_index("item_id")[["text"]]
    out_df = text_df.join(pivot).reset_index().rename(columns={"item_id": "doc_id"})
    for col in [c for c in out_df.columns if c.startswith("annotator_")]:
        out_df[col] = out_df[col].apply(lambda x: "|".join(sorted(x)) if isinstance(x, set) else "")

    # agreed_labels: labels selected by at least 3 annotators for each item
    all_annotators = project_id_list
    min_votes_for_agreement = 3

    def compute_agreed_labels(item_id, label_col):
        item_df = df[df["item_id"] == item_id]
        vote_counts = Counter()
        for annotator in all_annotators:
            row = item_df[item_df["annotator"] == annotator]
            if not row.empty:
                label_set = row.iloc[0][label_col]
                for label in label_set:
                    vote_counts[label] += 1
        if not vote_counts:
            return ""

        agreed = {label for label, count in vote_counts.items() if count >= min_votes_for_agreement}
        return "|".join(sorted(agreed))

    item_ids = out_df["doc_id"].tolist()
    out_df["agreed_labels"] = [compute_agreed_labels(i, "subject_labels") for i in item_ids]
    out_df["agreed_inflation_linked_labels"] = [compute_agreed_labels(i, "inflation_linked_subject_labels") for i in item_ids]

    csv_path = f"./export/annotations-{'-'.join([str(a) for a in project_id_list])}.csv"
    out_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nSaved annotation data to {csv_path}")

