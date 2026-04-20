
import json
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

    alpha_store = {
        "subjects": compute_label_alpha(df, project_id_list, "subject_labels"),
        "objects": compute_label_alpha(df, project_id_list, "object_labels"),
    }

    out_path = f"./export/alpha-{'-'.join([str(a) for a in project_id_list])}.json"
    with open(out_path, "w") as f:
        json.dump(alpha_store, f)
    print(f"Saved alpha scores to {out_path}")