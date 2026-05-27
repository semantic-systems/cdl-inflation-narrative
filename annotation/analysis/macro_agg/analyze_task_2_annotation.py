import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
import pandas as pd
import numpy as np
import krippendorff
from scipy.stats import chi2



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

    triples_label_form_raw = []
    for triple in triples:
        subj_label_form_raw = get_label_from_id(triple[0], results)
        obj_label_form_raw = get_label_from_id(triple[2], results)
        relation_label_form = triple[1]
        triples_label_form_raw.append((subj_label_form_raw, relation_label_form, obj_label_form_raw))

    triples_label_form = []
    event_remapping = {"Russia-Ukraine War": "War", "Energy Crisis": "Energy Prices", "House Costs": "Housing Costs"}
    for triple in triples:
        subj_label_form = get_label_from_id(triple[0], results)
        obj_label_form = get_label_from_id(triple[2], results)
        subj_label_form = event_remapping.get(subj_label_form, subj_label_form)
        obj_label_form = event_remapping.get(obj_label_form, obj_label_form)
        relation_label_form = triple[1]
        triples_label_form.append((subj_label_form, relation_label_form, obj_label_form))

    return triples, triples_surface_form, triples_label_form_raw, triples_label_form


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


def extract_direct_inflation_parents(triples_label_form):
    """Only direct parents of Inflation (one-hop), non-Inflation labels."""
    parent_map = {}
    for subject_label, _, object_label in triples_label_form:
        parent_map.setdefault(object_label, set()).add(subject_label)
    
    return parent_map.get("Inflation", set())


def extract_object_labels(triples_label_form):
    return {triple[2] for triple in triples_label_form if triple[2] != "Inflation"}


def print_label_distribution_descriptives(df, label_col, title):
    print(f"\n{title}")

    pooled_counts = Counter(
        label
        for label_set in df[label_col]
        for label in label_set
    )
    pooled_total = sum(pooled_counts.values())

    print(f"All annotators pooled (total label assignments: {pooled_total}):")
    for label, count in sorted(pooled_counts.items(), key=lambda x: (-x[1], x[0])):
        share = (count / pooled_total * 100.0) if pooled_total else 0.0
        print(f"  {label}: count={count}, share={share:.2f}%")

    print("By annotator:")
    for annotator in sorted(df["annotator"].unique()):
        annotator_df = df[df["annotator"] == annotator]
        annotator_counts = Counter(
            label
            for label_set in annotator_df[label_col]
            for label in label_set
        )
        annotator_total = sum(annotator_counts.values())
        print(f"  Annotator {annotator} (total label assignments: {annotator_total}):")
        for label, count in sorted(annotator_counts.items(), key=lambda x: (-x[1], x[0])):
            share = (count / annotator_total * 100.0) if annotator_total else 0.0
            print(f"    {label}: count={count}, share={share:.2f}%")


def chi_square_independence_stat(contingency):
    grand_total = contingency.sum()
    if grand_total == 0:
        return 0.0

    row_sums = contingency.sum(axis=1, keepdims=True)
    col_sums = contingency.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / grand_total
    valid = expected > 0

    chi2 = np.sum(((contingency - expected) ** 2)[valid] / expected[valid])
    return float(chi2)


def chi_square_sf(chi2_stat, dof):
    """Asymptotic right-tail p-value for chi-square using SciPy."""
    if dof <= 0:
        return None
    if chi2_stat <= 0:
        return 1.0
    return float(chi2.sf(chi2_stat, dof))


def build_assignment_contingency(df, label_col):
    label_assignments = []
    for _, row in df.iterrows():
        annotator = row["annotator"]
        for label in row[label_col]:
            label_assignments.append((annotator, label))

    if not label_assignments:
        return [], [], np.zeros((0, 0), dtype=float)

    annotators = sorted({annotator for annotator, _ in label_assignments})
    labels = sorted({label for _, label in label_assignments})
    annotator_to_idx = {annotator: i for i, annotator in enumerate(annotators)}
    label_to_idx = {label: j for j, label in enumerate(labels)}

    annotator_idx = np.array([annotator_to_idx[a] for a, _ in label_assignments], dtype=int)
    label_idx = np.array([label_to_idx[l] for _, l in label_assignments], dtype=int)

    contingency = np.zeros((len(annotators), len(labels)), dtype=float)
    np.add.at(contingency, (annotator_idx, label_idx), 1)
    return annotators, labels, contingency


def print_label_assignment_contingency_table(df, label_col, title, out_csv_path=None):
    annotators, labels, contingency = build_assignment_contingency(df, label_col)

    print(f"\n{title}")
    if contingency.size == 0:
        print("No label assignments available for contingency table.")
        return

    contingency_df = pd.DataFrame(contingency.astype(int), index=annotators, columns=labels)
    contingency_df["row_total"] = contingency_df.sum(axis=1)
    col_totals = contingency_df.sum(axis=0)
    contingency_df.loc["col_total"] = col_totals
    print(contingency_df.to_string())

    if out_csv_path is not None:
        contingency_df.to_csv(out_csv_path, encoding="utf-8")
        print(f"Saved contingency table to {out_csv_path}")


def print_label_distribution_difference_test(df, label_col, title):
    annotators, labels, contingency = build_assignment_contingency(df, label_col)
    if contingency.size == 0:
        print(f"\n{title}")
        print("No label assignments available for distribution-difference test.")
        return

    n_annotators = len(annotators)
    n_labels = len(labels)

    chi2_obs = chi_square_independence_stat(contingency)
    dof = (n_annotators - 1) * (n_labels - 1)
    p_value = chi_square_sf(chi2_obs, dof)

    print(f"\n{title}")
    print(
        "Asymptotic chi-square test (H0: annotator and label are independent): "
        f"chi2={chi2_obs:.4f}, dof={dof}, p={p_value:.6f}"
    )
    if p_value < 0.05:
        print("Result: reject H0 at 5% level; label distributions differ across annotators.")
    else:
        print("Result: fail to reject H0 at 5% level; no strong evidence of distribution differences.")


def print_single_label_chi_tests(df, label_col, title):
    print(f"\n{title}")

    annotators = sorted(df["annotator"].unique())
    labels = sorted({label for label_set in df[label_col] for label in label_set})
    if not labels:
        print("No labels available for single-label chi-square tests.")
        return

    annotator_to_idx = {annotator: i for i, annotator in enumerate(annotators)}
    obs_annotator_idx = np.array([annotator_to_idx[a] for a in df["annotator"]], dtype=int)
    n_annotators = len(annotators)
    dof = n_annotators - 1

    results = []
    for label in labels:
        present = np.array([1 if label in label_set else 0 for label_set in df[label_col]], dtype=float)
        totals = np.bincount(obs_annotator_idx, minlength=n_annotators).astype(float)
        present_counts = np.bincount(obs_annotator_idx, weights=present, minlength=n_annotators)
        absent_counts = totals - present_counts
        contingency = np.column_stack([present_counts, absent_counts])

        chi2_obs = chi_square_independence_stat(contingency)
        p_value = chi_square_sf(chi2_obs, dof)
        results.append((label, chi2_obs, p_value, present_counts, totals))

    for label, chi2_obs, p_value, present_counts, totals in sorted(results, key=lambda x: (x[2], -x[1], x[0])):
        shares = [
            f"{annotator}:{int(present_counts[i])}/{int(totals[i])} ({(present_counts[i] / totals[i] * 100.0) if totals[i] else 0.0:.1f}%)"
            for i, annotator in enumerate(annotators)
        ]
        print(
            f"  {label}: chi2={chi2_obs:.4f}, dof={dof}, p={p_value:.6f}, "
            f"per-annotator={'; '.join(shares)}"
        )


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


def compute_label_ac1(df, annotator_list, label_side):
    all_item_ids = sorted(df["item_id"].unique())
    all_labels = sorted({label for labels in df[label_side] for label in labels})

    results = {}
    m = len(annotator_list)
    if m < 2:
        return {label: None for label in all_labels}

    denom_pairs = m * (m - 1) / 2

    for label in all_labels:
        item_votes = []
        for item_id in all_item_ids:
            votes = []
            for annotator in annotator_list:
                annotator_df = df[df["annotator"] == annotator].set_index("item_id")
                if item_id not in annotator_df.index:
                    votes = []
                    break
                label_set = annotator_df.loc[item_id, label_side]
                votes.append(1 if label in label_set else 0)
            if len(votes) == m:
                item_votes.append(votes)

        if not item_votes:
            results[label] = None
            continue

        po_terms = []
        total_ones = 0
        total_ratings = 0
        for votes in item_votes:
            n1 = sum(votes)
            n0 = m - n1
            po_item = ((n1 * (n1 - 1) / 2) + (n0 * (n0 - 1) / 2)) / denom_pairs
            po_terms.append(po_item)
            total_ones += n1
            total_ratings += m

        po = float(np.mean(po_terms)) if po_terms else None
        if po is None:
            results[label] = None
            continue

        p1 = total_ones / total_ratings if total_ratings else 0.0
        p0 = 1.0 - p1
        pe = (p0 * (1.0 - p0)) + (p1 * (1.0 - p1))
        denom = 1.0 - pe
        if denom == 0:
            results[label] = None
            continue

        ac1 = (po - pe) / denom
        results[label] = float(ac1) if np.isfinite(ac1) else None

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


def compute_overall_simple_agreement(df, annotator_list, label_side):
    all_item_ids = sorted(df["item_id"].unique())
    all_labels = sorted({label for labels in df[label_side] for label in labels})

    agree = 0
    total = 0
    for label in all_labels:
        for item_id in all_item_ids:
            votes = []
            for annotator in annotator_list:
                annotator_df = df[df["annotator"] == annotator].set_index("item_id")
                if item_id not in annotator_df.index:
                    votes = []
                    break
                label_set = annotator_df.loc[item_id, label_side]
                votes.append(1 if label in label_set else 0)
            if len(votes) == len(annotator_list):
                total += 1
                if len(set(votes)) == 1:    
                    agree += 1

    return (agree / total) if total > 0 else None


def compute_overall_ac1(df, annotator_list, label_side):
    all_item_ids = sorted(df["item_id"].unique())
    all_labels = sorted({label for labels in df[label_side] for label in labels})

    m = len(annotator_list)
    if m < 2:
        return None

    denom_pairs = m * (m - 1) / 2
    po_terms = []
    total_ones = 0
    total_ratings = 0

    for label in all_labels:
        for item_id in all_item_ids:
            votes = []
            for annotator in annotator_list:
                annotator_df = df[df["annotator"] == annotator].set_index("item_id")
                if item_id not in annotator_df.index:
                    votes = []
                    break
                label_set = annotator_df.loc[item_id, label_side]
                votes.append(1 if label in label_set else 0)

            if len(votes) != m:
                continue

            n1 = sum(votes)
            n0 = m - n1
            po_item = ((n1 * (n1 - 1) / 2) + (n0 * (n0 - 1) / 2)) / denom_pairs
            po_terms.append(po_item)
            total_ones += n1
            total_ratings += m

    if not po_terms or total_ratings == 0:
        return None

    po = float(np.mean(po_terms))
    p1 = total_ones / total_ratings
    p0 = 1.0 - p1
    pe = (p0 * (1.0 - p0)) + (p1 * (1.0 - p1))
    denom = 1.0 - pe
    if denom == 0:
        return None

    ac1 = (po - pe) / denom
    return float(ac1) if np.isfinite(ac1) else None


if __name__ == "__main__":
    setup()

    project_id_list = [11, 12, 13, 14]
    project_annotations = get_task_2_annotation_json(project_id_list)

    rows = []
    for ann in project_annotations:
        _, _, triples_label_form_raw, triples_label_form = get_triples(ann["annotations"][0]["result"])
        rows.append({
            "annotator": ann["project"],
            "item_id": ann["data"]["inner_id"],
            "text": ann["data"]["text"],
            "triples_label_form_raw": triples_label_form_raw,
            "triples_label_form": triples_label_form,
        })

    df = pd.DataFrame(rows)

    EXCLUDED_LABELS = {"Base Effect", "Pandemic", "Trade Balance", "Mismanagement"}

    def exclude_labels(label_set):
        return {label for label in label_set if label not in EXCLUDED_LABELS}

    df["subject_labels_raw"] = df["triples_label_form_raw"].apply(extract_subject_labels).apply(exclude_labels)
    df["subject_labels"] = df["triples_label_form"].apply(extract_subject_labels).apply(exclude_labels)
    df["object_labels"] = df["triples_label_form"].apply(extract_object_labels).apply(exclude_labels)
    df["inflation_linked_subject_labels"] = df["triples_label_form"].apply(extract_inflation_linked_subject_labels).apply(exclude_labels)
    df["direct_inflation_parents"] = df["triples_label_form"].apply(extract_direct_inflation_parents).apply(exclude_labels)

    print_label_distribution_descriptives(
        df,
        "subject_labels_raw",
        "Raw subject-label distribution across all four annotators",
    )
    print_label_assignment_contingency_table(
        df,
        "subject_labels_raw",
        "Raw subject-label contingency table (annotator x label)",
        f"./export/contingency-raw-{'-'.join([str(a) for a in project_id_list])}.csv",
    )
    print_label_distribution_difference_test(
        df,
        "subject_labels_raw",
        "Raw subject-label distribution difference test across annotators",
    )
    print_single_label_chi_tests(
        df,
        "subject_labels_raw",
        "Raw single-label chi-square tests across annotators",
    )
    print_label_distribution_descriptives(
        df,
        "subject_labels",
        "Mapped subject-label distribution across all four annotators",
    )
    print_label_assignment_contingency_table(
        df,
        "subject_labels",
        "Mapped subject-label contingency table (annotator x label)",
        f"./export/contingency-mapped-{'-'.join([str(a) for a in project_id_list])}.csv",
    )
    print_label_distribution_difference_test(
        df,
        "subject_labels",
        "Mapped subject-label distribution difference test across annotators",
    )
    print_single_label_chi_tests(
        df,
        "subject_labels",
        "Mapped single-label chi-square tests across annotators",
    )

    subsets = [project_id_list] + [list(c) for c in combinations(project_id_list, len(project_id_list) - 1)]

    # --- Merged super-label analysis ---
    LABEL_GROUPS = {
        "Demand": {"Demand (residual)", "Pent-up Demand", "Demand Shift"},
        "Government": { "Government Debt", "Government Spending"},
        "Supply Chain": {"Supply Chain Issues", "Supply (residual)", "Transportation Costs"},
        "Labor": {"Labor Shortage", "Wages"},
        "Climate": {"Climate"},
        "War": {"War"},
        "Monetary": {"Monetary Policy", "Inflation Expectations", "Exchange Rates"},
        "Input Costs": {"Housing Costs", "Medical Costs", "Education Costs"},
        "Energy": {"Energy Prices"},
        "Food": {"Food Prices"},
        "Taxation": {"Tax Increases"},
        "Market Power": {"Price-Gouging"}
    }

    def apply_label_groups(label_set):
        merged = set()
        for label in label_set:
            group = next((g for g, members in LABEL_GROUPS.items() if label in members), label)
            merged.add(group)
        return merged

    df["subject_labels_merged"] = df["subject_labels"].apply(apply_label_groups)

    # Simple frequency count of each merged super-label across all annotations
    merged_label_counts = Counter(
        label
        for label_set in df["subject_labels_merged"]
        for label in label_set
    )
    print("\nMerged super-label counts:")
    for label, count in sorted(merged_label_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {label}: {count}")

    print("\n--- Merged super-label subject alpha ---")
    merged_alpha_store = {}
    for subset in subsets:
        key = "-".join(str(a) for a in subset)
        subj_alpha = compute_label_alpha(df, subset, "subject_labels_merged")
        subj_ac1 = compute_label_ac1(df, subset, "subject_labels_merged")
        subj_simple = compute_label_raw_agreement(df, subset, "subject_labels_merged")
        merged_alpha_store[key] = {
            "subjects_alpha": subj_alpha,
            "subjects_ac1": subj_ac1,
            "subjects_simple_agreement": subj_simple,
            "subjects_overall_alpha": compute_overall_alpha(df, subset, "subject_labels_merged"),
            "subjects_overall_ac1": compute_overall_ac1(df, subset, "subject_labels_merged"),
            "subjects_overall_simple_agreement": compute_overall_simple_agreement(df, subset, "subject_labels_merged"),
        }

    print("\nMerged subject overall agreement (across all categories):")
    for key, val in merged_alpha_store.items():
        alpha_str = f"{val['subjects_overall_alpha']:.4f}" if val["subjects_overall_alpha"] is not None else "N/A"
        ac1_str = f"{val['subjects_overall_ac1']:.4f}" if val["subjects_overall_ac1"] is not None else "N/A"
        simple_str = f"{val['subjects_overall_simple_agreement']:.4f}" if val["subjects_overall_simple_agreement"] is not None else "N/A"
        print(f"  {key}: alpha={alpha_str}  ac1={ac1_str}  simple={simple_str}")

    print("\nMerged subject agreement per label (per setting):")
    for key, val in merged_alpha_store.items():
        overall_str = f"{val['subjects_overall_alpha']:.4f}" if val["subjects_overall_alpha"] is not None else "N/A"
        overall_ac1_str = f"{val['subjects_overall_ac1']:.4f}" if val["subjects_overall_ac1"] is not None else "N/A"
        overall_simple_str = f"{val['subjects_overall_simple_agreement']:.4f}" if val["subjects_overall_simple_agreement"] is not None else "N/A"
        print(f"  {key} (overall alpha: {overall_str}, ac1: {overall_ac1_str}, simple: {overall_simple_str}):")
        for label in sorted(val["subjects_alpha"].keys()):
            alpha_score = val["subjects_alpha"].get(label)
            ac1_score = val["subjects_ac1"].get(label)
            simple_score = val["subjects_simple_agreement"].get(label)
            alpha_str = f"{alpha_score:.4f}" if alpha_score is not None else "N/A"
            ac1_str = f"{ac1_score:.4f}" if ac1_score is not None else "N/A"
            simple_str = f"{simple_score:.4f}" if simple_score is not None else "N/A"
            print(f"    {label}: alpha={alpha_str}  ac1={ac1_str}  simple={simple_str}")

    # Also apply merged super-labels to inflation-linked subjects
    df["inflation_linked_subject_labels_merged"] = df["inflation_linked_subject_labels"].apply(apply_label_groups)

    print("\nInflation-linked merged super-label agreement means:")
    inflation_linked_merged_alpha_store = {}
    for subset in subsets:
        key = "-".join(str(a) for a in subset)
        subj_alpha = compute_label_alpha(df, subset, "inflation_linked_subject_labels_merged")
        subj_ac1 = compute_label_ac1(df, subset, "inflation_linked_subject_labels_merged")
        subj_simple = compute_label_raw_agreement(df, subset, "inflation_linked_subject_labels_merged")
        inflation_linked_merged_alpha_store[key] = {
            "subjects_alpha": subj_alpha,
            "subjects_ac1": subj_ac1,
            "subjects_simple_agreement": subj_simple,
            "subjects_overall_alpha": compute_overall_alpha(df, subset, "inflation_linked_subject_labels_merged"),
            "subjects_overall_ac1": compute_overall_ac1(df, subset, "inflation_linked_subject_labels_merged"),
            "subjects_overall_simple_agreement": compute_overall_simple_agreement(df, subset, "inflation_linked_subject_labels_merged"),
        }
        alpha_str = f"{inflation_linked_merged_alpha_store[key]['subjects_overall_alpha']:.4f}" if inflation_linked_merged_alpha_store[key]["subjects_overall_alpha"] is not None else "N/A"
        ac1_str = f"{inflation_linked_merged_alpha_store[key]['subjects_overall_ac1']:.4f}" if inflation_linked_merged_alpha_store[key]["subjects_overall_ac1"] is not None else "N/A"
        simple_str = f"{inflation_linked_merged_alpha_store[key]['subjects_overall_simple_agreement']:.4f}" if inflation_linked_merged_alpha_store[key]["subjects_overall_simple_agreement"] is not None else "N/A"
        print(f"  {key}: alpha={alpha_str}  ac1={ac1_str}  simple={simple_str}")

    # --- One-hop analysis: direct parents of Inflation only ---
    df["direct_inflation_parents_merged"] = df["direct_inflation_parents"].apply(apply_label_groups)

    print("\n--- ONE-HOP: Direct parents of Inflation (merged super-labels) ---")
    direct_parents_merged_alpha_store = {}
    for subset in subsets:
        key = "-".join(str(a) for a in subset)
        subj_alpha = compute_label_alpha(df, subset, "direct_inflation_parents_merged")
        subj_ac1 = compute_label_ac1(df, subset, "direct_inflation_parents_merged")
        subj_simple = compute_label_raw_agreement(df, subset, "direct_inflation_parents_merged")
        direct_parents_merged_alpha_store[key] = {
            "subjects_alpha": subj_alpha,
            "subjects_ac1": subj_ac1,
            "subjects_simple_agreement": subj_simple,
            "subjects_overall_alpha": compute_overall_alpha(df, subset, "direct_inflation_parents_merged"),
            "subjects_overall_ac1": compute_overall_ac1(df, subset, "direct_inflation_parents_merged"),
            "subjects_overall_simple_agreement": compute_overall_simple_agreement(df, subset, "direct_inflation_parents_merged"),
        }
        alpha_str = f"{direct_parents_merged_alpha_store[key]['subjects_overall_alpha']:.4f}" if direct_parents_merged_alpha_store[key]["subjects_overall_alpha"] is not None else "N/A"
        ac1_str = f"{direct_parents_merged_alpha_store[key]['subjects_overall_ac1']:.4f}" if direct_parents_merged_alpha_store[key]["subjects_overall_ac1"] is not None else "N/A"
        simple_str = f"{direct_parents_merged_alpha_store[key]['subjects_overall_simple_agreement']:.4f}" if direct_parents_merged_alpha_store[key]["subjects_overall_simple_agreement"] is not None else "N/A"
        print(f"  {key}: alpha={alpha_str}  ac1={ac1_str}  simple={simple_str}")

    print("\nDirect parents agreement per label (per setting):")
    for key, val in direct_parents_merged_alpha_store.items():
        overall_str = f"{val['subjects_overall_alpha']:.4f}" if val["subjects_overall_alpha"] is not None else "N/A"
        overall_ac1_str = f"{val['subjects_overall_ac1']:.4f}" if val["subjects_overall_ac1"] is not None else "N/A"
        overall_simple_str = f"{val['subjects_overall_simple_agreement']:.4f}" if val["subjects_overall_simple_agreement"] is not None else "N/A"
        print(f"  {key} (overall alpha: {overall_str}, ac1: {overall_ac1_str}, simple: {overall_simple_str}):")
        for label in sorted(val["subjects_alpha"].keys()):
            alpha_score = val["subjects_alpha"].get(label)
            ac1_score = val["subjects_ac1"].get(label)
            simple_score = val["subjects_simple_agreement"].get(label)
            alpha_str = f"{alpha_score:.4f}" if alpha_score is not None else "N/A"
            ac1_str = f"{ac1_score:.4f}" if ac1_score is not None else "N/A"
            simple_str = f"{simple_score:.4f}" if simple_score is not None else "N/A"
            print(f"    {label}: alpha={alpha_str}  ac1={ac1_str}  simple={simple_str}")

    out_path = f"./export/alpha-super-{'-'.join([str(a) for a in project_id_list])}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "subject_labels_merged": merged_alpha_store,
                "inflation_linked_subject_labels_merged": inflation_linked_merged_alpha_store,
                "direct_inflation_parents_merged": direct_parents_merged_alpha_store,
            },
            f,
        )
    print(f"Saved super-category alpha scores to {out_path}")

    # --- Co-occurrence of all super-categories for annotators 11/12/13 ---
    target_annotators = {11, 12, 13}
    df_target = df[df["annotator"].isin(target_annotators)].copy()

    single_counts = Counter()
    pair_counts = Counter()

    for label_set in df_target["subject_labels_merged"]:
        labels = sorted(label_set)
        for label in labels:
            single_counts[label] += 1
        for left, right in combinations(labels, 2):
            pair_counts[(left, right)] += 1

    print("\n--- Super-category co-occurrence (annotators 11, 12, 13) ---")
    print("Single-label counts:")
    for label, count in sorted(single_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {label}: {count}")

    print("\nPair co-occurrence counts:")
    for (left, right), count in sorted(pair_counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
        print(f"  {left} + {right}: {count}")

    # --- Association strength for raw labels (annotators 11/12/13) ---
    raw_single_counts = Counter()
    raw_pair_counts = Counter()
    df_target_raw = df[df["annotator"].isin(target_annotators)].copy()
    num_observations = len(df_target_raw)

    for label_set in df_target_raw["subject_labels"]:
        labels = sorted(label_set)
        for label in labels:
            raw_single_counts[label] += 1
        for left, right in combinations(labels, 2):
            raw_pair_counts[(left, right)] += 1

    association_rows = []
    for (left, right), pair_count in raw_pair_counts.items():
        p_ab = pair_count / num_observations
        p_a = raw_single_counts[left] / num_observations
        p_b = raw_single_counts[right] / num_observations
        if p_a == 0 or p_b == 0:
            continue
        lift = p_ab / (p_a * p_b)
        pmi = math.log2(lift) if lift > 0 else None
        association_rows.append({
            "label_left": left,
            "label_right": right,
            "pair_count": pair_count,
            "lift": lift,
            "pmi": pmi,
        })

    assoc_df = pd.DataFrame(association_rows)
    if assoc_df.empty:
        print("\nNo raw-label association pairs found for annotators 11, 12, 13.")
    else:
        assoc_df = assoc_df.sort_values(["lift", "pair_count"], ascending=[False, False])
        print("\n--- Raw-label association strength (annotators 11, 12, 13) ---")
        print("Top pairs by lift (with PMI):")
        for _, row in assoc_df.head(30).iterrows():
            print(
                f"  {row['label_left']} + {row['label_right']}: "
                f"count={int(row['pair_count'])}, lift={row['lift']:.3f}, pmi={row['pmi']:.3f}"
            )

        assoc_out_path = "./export/raw-label-association-11-12-13.csv"
        assoc_df.to_csv(assoc_out_path, index=False, encoding="utf-8")
        print(f"Saved raw-label association strengths to {assoc_out_path}")

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

