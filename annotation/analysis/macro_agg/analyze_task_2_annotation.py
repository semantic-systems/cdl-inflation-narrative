import json
import importlib
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


def plot_label_frequency_histogram(df, label_col, title, out_html_path):
    out_pdf_path = Path(out_html_path).with_suffix(".pdf")
    try:
        px = importlib.import_module("plotly.express")
    except ModuleNotFoundError:
        px = None

    assignments = [
        {
            "annotator": str(row["annotator"]),
            "label": label,
        }
        for _, row in df.iterrows()
        for label in row[label_col]
    ]

    print(f"\n{title}")
    if not assignments:
        print("No label assignments available for histogram.")
        return

    plot_df = pd.DataFrame(assignments)
    plot_df["share"] = plot_df.groupby("annotator")["label"].transform("count")
    plot_df["share"] = 1.0 / plot_df["share"]
    plot_df = (
        plot_df.groupby(["label", "annotator"], as_index=False)["share"]
        .sum()
    )

    label_order = plot_df.groupby("label")["share"].sum().sort_values(ascending=False).index.tolist()
    annotator_order = sorted(plot_df["annotator"].unique())

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        plt = None

    if plt is not None:
        pivot = plot_df.pivot(index="label", columns="annotator", values="share").reindex(label_order).fillna(0.0)
        fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(label_order)), 6))
        x = np.arange(len(label_order))
        bar_width = 0.8 / max(1, len(annotator_order))

        for idx, annotator in enumerate(annotator_order):
            offsets = x - 0.4 + bar_width / 2 + idx * bar_width
            values = pivot[annotator].to_numpy() if annotator in pivot.columns else np.zeros(len(label_order))
            ax.bar(offsets, values * 100.0, width=bar_width, label=str(annotator))

        ax.set_xticks(x)
        ax.set_xticklabels(label_order, rotation=45, ha="right")
        ax.set_xlabel("Label")
        ax.set_ylabel("Share of label assignments (%)")
        ax.set_title(title)
        ax.legend(title="Annotator", frameon=False, ncol=min(4, len(annotator_order)))
        ax.set_ylim(0, max(100.0, (pivot.max().max() if not pivot.empty else 0.0) * 100.0 * 1.1))
        fig.tight_layout()
        fig.savefig(out_pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved histogram to {out_pdf_path}")
    else:
        print("Matplotlib is not installed; skipping PDF export.")

    if px is not None:
        fig = px.histogram(
            plot_df,
            x="label",
            y="share",
            color="annotator",
            barmode="group",
            histfunc="sum",
            category_orders={
                "label": label_order,
                "annotator": annotator_order,
            },
            labels={
                "label": "Label",
                "annotator": "Annotator",
                "share": "Share",
            },
            title=title,
        )
        fig.update_layout(
            xaxis_title="Label",
            yaxis_title="Share of label assignments (%)",
            legend_title_text="Annotator",
        )
        fig.update_xaxes(tickangle=45)
        fig.write_html(out_html_path)
        print(f"Saved histogram to {out_html_path}")
    else:
        print("Plotly is not installed; skipping HTML export.")


def plot_overall_alpha_by_subset(alpha_store, title, out_html_path):
    out_pdf_path = Path(out_html_path).with_suffix(".pdf")

    rows = []
    for subset_key, metrics in alpha_store.items():
        alpha_value = metrics.get("subjects_overall_alpha")
        if alpha_value is not None:
            rows.append({"subset": subset_key, "alpha": alpha_value})

    print(f"\n{title}")
    if not rows:
        print("No overall alpha values available for plotting.")
        return

    def _subset_sort_key(subset_key):
        parts = subset_key.split("-")
        return (-len(parts), subset_key)

    plot_df = pd.DataFrame(rows)
    subset_order = sorted(plot_df["subset"].tolist(), key=_subset_sort_key)
    plot_df["subset"] = pd.Categorical(plot_df["subset"], categories=subset_order, ordered=True)
    plot_df = plot_df.sort_values("subset")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        plt = None

    if plt is not None:
        fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(subset_order)), 5))
        x = np.arange(len(plot_df))
        y = plot_df["alpha"].to_numpy(dtype=float)
        ax.bar(x, y, width=0.6)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["subset"].astype(str).tolist(), rotation=35, ha="right")
        ax.set_xlabel("Annotator subset")
        ax.set_ylabel("Krippendorff alpha")
        ax.set_title(title)
        ymin = min(-0.1, float(np.min(y)) - 0.05)
        ymax = max(1.0, float(np.max(y)) + 0.05)
        ax.set_ylim(ymin, ymax)
        fig.tight_layout()
        fig.savefig(out_pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved alpha plot to {out_pdf_path}")
    else:
        print("Matplotlib is not installed; skipping PDF alpha plot export.")

    try:
        px = importlib.import_module("plotly.express")
    except ModuleNotFoundError:
        px = None

    if px is not None:
        fig = px.bar(
            plot_df,
            x="subset",
            y="alpha",
            category_orders={"subset": subset_order},
            labels={"subset": "Annotator subset", "alpha": "Krippendorff alpha"},
            title=title,
        )
        fig.update_layout(yaxis_title="Krippendorff alpha")
        fig.write_html(out_html_path)
        print(f"Saved alpha plot to {out_html_path}")
    else:
        print("Plotly is not installed; skipping HTML alpha plot export.")




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


def compute_label_free_marginal_kappa(df, annotator_list, label_side):
    """Per-label Randolph free-marginal multirater kappa using statsmodels.

    Returns a dict mapping label -> kappa (or None if not computable).
    """
    try:
        from statsmodels.stats.inter_rater import fleiss_kappa
    except Exception:
        return {label: None for label in sorted({label for labels in df[label_side] for label in labels})}

    all_item_ids = sorted(df["item_id"].unique())
    all_labels = sorted({label for labels in df[label_side] for label in labels})

    m = len(annotator_list)
    results = {}
    if m < 2:
        return {label: None for label in all_labels}

    for label in all_labels:
        rows = []
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
            present = int(sum(votes))
            absent = m - present
            rows.append([present, absent])

        if not rows:
            results[label] = None
            continue

        table = np.array(rows, dtype=int)
        try:
            k = fleiss_kappa(table, method="randolph")
            results[label] = float(k) if np.isfinite(k) else None
        except Exception:
            results[label] = None

    return results


def compute_overall_free_marginal_kappa(df, annotator_list, label_side):
    """Attempt an overall free-marginal kappa by treating each label as a separate binary category
    and stacking item-label pairs as "subjects". Returns a single kappa float or None.
    """
    try:
        from statsmodels.stats.inter_rater import fleiss_kappa
    except Exception:
        return None

    all_item_ids = sorted(df["item_id"].unique())
    all_labels = sorted({label for labels in df[label_side] for label in labels})
    m = len(annotator_list)
    if m < 2 or not all_labels:
        return None

    rows = []
    # For each (item, label) pair, count how many annotators selected the label (present) and absent
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
            present = int(sum(votes))
            absent = m - present
            rows.append([present, absent])

    if not rows:
        return None

    table = np.array(rows, dtype=int)
    try:
        k = fleiss_kappa(table, method="randolph")
        return float(k) if np.isfinite(k) else None
    except Exception:
        return None


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

    EXCLUDED_LABELS = {"Base Effect", "Pandemic", "Trade Balance", "Mismanagement", "Inflation Expectations"}

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
    plot_label_frequency_histogram(
        df,
        "subject_labels_raw",
        "Raw subject-label frequency histogram across annotators",
        f"./export/subject-label-frequency-raw-{'-'.join([str(a) for a in project_id_list])}.html",
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
    plot_label_frequency_histogram(
        df,
        "subject_labels",
        "Mapped subject-label frequency histogram across annotators",
        f"./export/subject-label-frequency-mapped-{'-'.join([str(a) for a in project_id_list])}.html",
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
        "Monetary": {"Monetary Policy"}, # exchange rates
        #"Inflation Expectations": {"Inflation Expectations"},
        "Exchange Rates": {"Exchange Rates"},
        "Input Costs": {"Housing Costs", "Medical Costs", "Education Costs"},
        "Energy": {"Energy Prices"},
        "Food": {"Food Prices"},
        "Taxation": {"Tax Increases"},
        "Market Power": {"Price-Gouging"}
    }
    
    LABEL_GROUPS = {
    "Fiscal Policy & Government Action": [
        "Government Debt", "Government Spending", "Tax Increases"
    ],
    "Private Demand Shocks": [
        "Pent-up Demand", "Demand Shift", "Demand (residual)"
    ],
    "Exogenous Supply & Commodity Shocks": [
        "Supply Chain Issues", "Supply (residual)", "Transportation Costs",
        "Energy Prices", "Food Prices"
    ],
    "External / Geopolitical Shocks": [
        "War", "Climate", "Exchange Rates"
    ],
    "Labor Market Dynamics": [
        "Labor Shortage", "Wages"
    ],
    "Micro-Frictions": [
        "Housing Costs", "Medical Costs", "Education Costs"
    ],
    
    "Market Power": [
        "Price-Gouging"
    ],
    
    "Monetary Policy": [
        "Monetary Policy"
    ]
}

    # --- Reduced Macroeconomic Super-label Analysis ---
    REDUCED_LABEL_GROUPS = {
        "Aggregate Demand": {
            "Demand (residual)", "Pent-up Demand", "Demand Shift",
            "Government Debt", "Government Spending"
        },
        "Tax Increases": {
            "Tax Increases"
        },
        "Exogenous Supply Shocks": {
            "Supply Chain Issues", "Supply (residual)", "Transportation Costs",
            "Energy Prices", "Food Prices", 
        },
        "War": {
            "War"
        },
        "Climate": {
            "Climate"
        },
        "Labor Market Dynamics": {
            "Labor Shortage", "Wages"
        },
        "Structural Costs & Frictions": {
            "Housing Costs", "Medical Costs",
            "Education Costs"
        },
        
        "Market Power": {
            "Price-Gouging"
        },
        "Monetary Policy": {
            "Monetary Policy"
        }, 
        
        "Exchange Rates": {
            "Exchange Rates"
        },
    }

    def build_label_to_group_map(label_groups):
        label_to_group = {}
        for group_name, members in label_groups.items():
            for member in members:
                label_to_group[member] = group_name
        return label_to_group

    def apply_group_map(label_set, label_to_group):
        return {label_to_group.get(label, label) for label in label_set}

    label_to_super_group = build_label_to_group_map(LABEL_GROUPS)
    label_to_reduced_group = build_label_to_group_map(REDUCED_LABEL_GROUPS)

    def apply_label_groups(label_set):
        return apply_group_map(label_set, label_to_super_group)

    def apply_reduced_label_groups(label_set):
        return apply_group_map(label_set, label_to_reduced_group)

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

    df["subject_labels_reduced_merged"] = df["subject_labels"].apply(apply_reduced_label_groups)
    reduced_label_counts = Counter(
        label
        for label_set in df["subject_labels_reduced_merged"]
        for label in label_set
    )
    print("\nReduced merged super-label counts:")
    for label, count in sorted(reduced_label_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {label}: {count}")

    # IAA is computed only for inflation-linked merged super-labels.
    df["inflation_linked_subject_labels_merged"] = df["inflation_linked_subject_labels"].apply(apply_label_groups)
    df["inflation_linked_subject_labels_reduced_merged"] = df["inflation_linked_subject_labels"].apply(apply_reduced_label_groups)

    print("\n--- Inflation-linked merged super-label IAA ---")
    inflation_linked_merged_alpha_store = {}
    for subset in subsets:
        key = "-".join(str(a) for a in subset)
        subj_alpha = compute_label_alpha(df, subset, "inflation_linked_subject_labels_merged")
        subj_ac1 = compute_label_ac1(df, subset, "inflation_linked_subject_labels_merged")
        subj_simple = compute_label_raw_agreement(df, subset, "inflation_linked_subject_labels_merged")
        subj_fm_kappa = compute_label_free_marginal_kappa(df, subset, "inflation_linked_subject_labels_merged")
        subj_overall_fm = compute_overall_free_marginal_kappa(df, subset, "inflation_linked_subject_labels_merged")
        inflation_linked_merged_alpha_store[key] = {
            "subjects_alpha": subj_alpha,
            "subjects_ac1": subj_ac1,
            "subjects_simple_agreement": subj_simple,
            "subjects_free_marginal_kappa": subj_fm_kappa,
            "subjects_overall_alpha": compute_overall_alpha(df, subset, "inflation_linked_subject_labels_merged"),
            "subjects_overall_ac1": compute_overall_ac1(df, subset, "inflation_linked_subject_labels_merged"),
            "subjects_overall_simple_agreement": compute_overall_simple_agreement(df, subset, "inflation_linked_subject_labels_merged"),
            "subjects_overall_free_marginal_kappa": subj_overall_fm,
        }
        alpha_str = f"{inflation_linked_merged_alpha_store[key]['subjects_overall_alpha']:.4f}" if inflation_linked_merged_alpha_store[key]["subjects_overall_alpha"] is not None else "N/A"
        ac1_str = f"{inflation_linked_merged_alpha_store[key]['subjects_overall_ac1']:.4f}" if inflation_linked_merged_alpha_store[key]["subjects_overall_ac1"] is not None else "N/A"
        simple_str = f"{inflation_linked_merged_alpha_store[key]['subjects_overall_simple_agreement']:.4f}" if inflation_linked_merged_alpha_store[key]["subjects_overall_simple_agreement"] is not None else "N/A"
        fm_str = f"{inflation_linked_merged_alpha_store[key]['subjects_overall_free_marginal_kappa']:.4f}" if inflation_linked_merged_alpha_store[key]["subjects_overall_free_marginal_kappa"] is not None else "N/A"
        print(f"  {key}: alpha={alpha_str}  ac1={ac1_str}  simple={simple_str}  fm_kappa={fm_str}")

    print("\nInflation-linked merged super-label single-label alpha:")
    for key, val in inflation_linked_merged_alpha_store.items():
        print(f"  {key}:")
        for label in sorted(val["subjects_alpha"].keys()):
            alpha_score = val["subjects_alpha"].get(label)
            alpha_str = f"{alpha_score:.4f}" if alpha_score is not None else "N/A"
            fm_score = None
            if val.get("subjects_free_marginal_kappa"):
                fm_score = val["subjects_free_marginal_kappa"].get(label)
            fm_str = f"{fm_score:.4f}" if fm_score is not None else "N/A"
            print(f"    {label}: alpha={alpha_str}  fm_kappa={fm_str}")

    print("\n--- Inflation-linked reduced merged super-label IAA ---")
    inflation_linked_reduced_merged_alpha_store = {}
    for subset in subsets:
        key = "-".join(str(a) for a in subset)
        subj_alpha = compute_label_alpha(df, subset, "inflation_linked_subject_labels_reduced_merged")
        subj_ac1 = compute_label_ac1(df, subset, "inflation_linked_subject_labels_reduced_merged")
        subj_simple = compute_label_raw_agreement(df, subset, "inflation_linked_subject_labels_reduced_merged")
        subj_fm_kappa = compute_label_free_marginal_kappa(df, subset, "inflation_linked_subject_labels_reduced_merged")
        subj_overall_fm = compute_overall_free_marginal_kappa(df, subset, "inflation_linked_subject_labels_reduced_merged")
        inflation_linked_reduced_merged_alpha_store[key] = {
            "subjects_alpha": subj_alpha,
            "subjects_ac1": subj_ac1,
            "subjects_simple_agreement": subj_simple,
            "subjects_free_marginal_kappa": subj_fm_kappa,
            "subjects_overall_alpha": compute_overall_alpha(df, subset, "inflation_linked_subject_labels_reduced_merged"),
            "subjects_overall_ac1": compute_overall_ac1(df, subset, "inflation_linked_subject_labels_reduced_merged"),
            "subjects_overall_simple_agreement": compute_overall_simple_agreement(df, subset, "inflation_linked_subject_labels_reduced_merged"),
            "subjects_overall_free_marginal_kappa": subj_overall_fm,
        }
        alpha_str = f"{inflation_linked_reduced_merged_alpha_store[key]['subjects_overall_alpha']:.4f}" if inflation_linked_reduced_merged_alpha_store[key]["subjects_overall_alpha"] is not None else "N/A"
        ac1_str = f"{inflation_linked_reduced_merged_alpha_store[key]['subjects_overall_ac1']:.4f}" if inflation_linked_reduced_merged_alpha_store[key]["subjects_overall_ac1"] is not None else "N/A"
        simple_str = f"{inflation_linked_reduced_merged_alpha_store[key]['subjects_overall_simple_agreement']:.4f}" if inflation_linked_reduced_merged_alpha_store[key]["subjects_overall_simple_agreement"] is not None else "N/A"
        fm_str = f"{inflation_linked_reduced_merged_alpha_store[key]['subjects_overall_free_marginal_kappa']:.4f}" if inflation_linked_reduced_merged_alpha_store[key]["subjects_overall_free_marginal_kappa"] is not None else "N/A"
        print(f"  {key}: alpha={alpha_str}  ac1={ac1_str}  simple={simple_str}  fm_kappa={fm_str}")

    print("\nInflation-linked reduced merged super-label single-label alpha:")
    for key, val in inflation_linked_reduced_merged_alpha_store.items():
        print(f"  {key}:")
        for label in sorted(val["subjects_alpha"].keys()):
            alpha_score = val["subjects_alpha"].get(label)
            alpha_str = f"{alpha_score:.4f}" if alpha_score is not None else "N/A"
            fm_score = None
            if val.get("subjects_free_marginal_kappa"):
                fm_score = val["subjects_free_marginal_kappa"].get(label)
            fm_str = f"{fm_score:.4f}" if fm_score is not None else "N/A"
            print(f"    {label}: alpha={alpha_str}  fm_kappa={fm_str}")

    inflation_linked_single_alpha_store = {}
    for subset in subsets:
        key = "-".join(str(a) for a in subset)
        inflation_linked_single_alpha_store[key] = compute_label_alpha(df, subset, "inflation_linked_subject_labels")

    inflation_linked_single_kappa_store = {}
    for subset in subsets:
        key = "-".join(str(a) for a in subset)
        inflation_linked_single_kappa_store[key] = compute_label_free_marginal_kappa(df, subset, "inflation_linked_subject_labels")

    print("\nInflation-linked single-label alpha (non-merged labels):")
    for key, label_alpha in inflation_linked_single_alpha_store.items():
        print(f"  {key}:")
        for label in sorted(label_alpha.keys()):
            alpha_score = label_alpha.get(label)
            alpha_str = f"{alpha_score:.4f}" if alpha_score is not None else "N/A"
            kappa_score = None
            if inflation_linked_single_kappa_store.get(key):
                kappa_score = inflation_linked_single_kappa_store[key].get(label)
            kappa_str = f"{kappa_score:.4f}" if kappa_score is not None else "N/A"
            print(f"    {label}: alpha={alpha_str}  fm_kappa={kappa_str}")

    out_path = f"./export/alpha-super-{'-'.join([str(a) for a in project_id_list])}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "inflation_linked_subject_labels_merged": inflation_linked_merged_alpha_store,
                "inflation_linked_subject_labels_reduced_merged": inflation_linked_reduced_merged_alpha_store,
                    "inflation_linked_subject_labels_single_alpha": inflation_linked_single_alpha_store,
                    "inflation_linked_subject_labels_single_kappa": inflation_linked_single_kappa_store,
                "label_groups": {k: sorted(v) for k, v in LABEL_GROUPS.items()},
                "reduced_label_groups": {k: sorted(v) for k, v in REDUCED_LABEL_GROUPS.items()},
            },
            f,
        )
    print(f"Saved super-category alpha scores to {out_path}")

    # --- Save annotated data as CSV ---
    pivot = df.pivot(index="item_id", columns="annotator", values="subject_labels")
    pivot.columns = [f"annotator_{col}" for col in pivot.columns]
    text_df = df.drop_duplicates("item_id").set_index("item_id")[["text"]]
    out_df = text_df.join(pivot).reset_index().rename(columns={"item_id": "doc_id"})
    for col in [c for c in out_df.columns if c.startswith("annotator_")]:
        out_df[col] = out_df[col].apply(lambda x: "|".join(sorted(x)) if isinstance(x, set) else "")

    # agreed_labels: labels selected by at least 3 annotators for each item
    all_annotators = [11, 12, 13]
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

    # Final compact export: text + agreed annotations for three (super-)label variants.
    category_out_df = out_df[["doc_id", "text"]].copy()
    category_out_df["annotations_single_labels"] = [
        compute_agreed_labels(i, "inflation_linked_subject_labels") for i in item_ids
    ]
    category_out_df["annotations_super_labels"] = [
        compute_agreed_labels(i, "inflation_linked_subject_labels_merged") for i in item_ids
    ]
    category_out_df["annotations_reduced_super_labels"] = [
        compute_agreed_labels(i, "inflation_linked_subject_labels_reduced_merged") for i in item_ids
    ]

    category_csv_path = f"./export/annotations-three-category-superlabels-{'-'.join([str(a) for a in project_id_list])}.csv"
    category_out_df.to_csv(category_csv_path, index=False, encoding="utf-8")
    print(f"Saved three-category super-label annotation data to {category_csv_path}")




