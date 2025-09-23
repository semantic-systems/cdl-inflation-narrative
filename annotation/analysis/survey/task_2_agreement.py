import argparse
import json
from pathlib import Path
from itertools import chain
import pandas as pd
import requests
import numpy as np
import networkx as nx
from krippendorff_graph import (compute_alpha, graph_edit_distance, graph_overlap_metric,
                                nominal_metric, node_overlap_metric, compute_distance_matrix)


def setup(): # Create export directory if it does not exist
    if not Path("./export").exists():
        Path("./export").mkdir()


def export_project_to_json(project_id, write_to_dist=True):
    url = f"{LABEL_STUDIO_URL}api/projects/{project_id}/export"
    headers = {"Authorization": f"Token  {API_KEY}"}
    response = requests.get(url, headers=headers)
    export = response.json()
    if write_to_dist:
        with open(f'./export/annotation_task2_project_{project_id}.json', 'w') as f:
            json.dump(export, f)
    return export

def get_task_2_annotation_json(project_id_list, redownload=False):
    project_annotations = []
    for project_id in project_id_list:
        cache_path = Path(f'./export/annotation_task2_project_{project_id}.json')
        if redownload or not cache_path.exists():
            project_annotations.extend(export_project_to_json(project_id))
        else:
            with open(cache_path, "r") as f:
                project_annotations.extend(json.load(f))
    return project_annotations


def get_triples(results):
    relation_triples = [result for result in results if result["type"] == "relation"]
    subjects = [relation_triple["from_id"] for relation_triple in relation_triples]
    objects = [relation_triple["to_id"] for relation_triple in relation_triples]
    relations = [relation_triple.get("labels", []) for relation_triple in relation_triples]
    triples = []
    for i in range(len(subjects)):
        triples.append((subjects[i], relations[i], objects[i]))

    # get triples in surface form
    triples_surface_form = []
    for triple in triples:
        subj_surface_form = get_surface_form_from_id(triple[0], results)
        obj_surface_form = get_surface_form_from_id(triple[2], results)
        relation_surface_form = triple[1]
        triples_surface_form.append((subj_surface_form, relation_surface_form, obj_surface_form))

    # get triples in label form
    triples_label_form = []
    #event_remapping = {"Russia-Ukraine War": "War", "Energy Crisis": "Energy Prices", "House Costs": 'Housing Costs'}
    for triple in triples:
        subj_label_form = get_label_from_id(triple[0], results)
        obj_label_form = get_label_from_id(triple[2], results)
        #subj_label_form = event_remapping.get(subj_label_form, subj_label_form)
        #obj_label_form = event_remapping.get(obj_label_form, obj_label_form)
        relation_label_form = triple[1]
        triples_label_form.append((subj_label_form, relation_label_form, obj_label_form))

    return triples, triples_surface_form, triples_label_form


def get_surface_form_from_id(result_id, results):
    return [result["value"]["text"] for result in results if result.get("id", None) == result_id][0]


def get_label_from_id(result_id, results):
    return [result["value"]["labels"][0] for result in results if result.get("id", None) == result_id][0]


def get_feature_one(row):
    # constituent event: factors that have a direct relation to inflation
    constituent_event = {triple[0] for triple in row["triples_label_form"] if triple[2] == "Inflation"}
    if not constituent_event:
        constituent_event = "*"
    return constituent_event


def get_feature_two(row):
    # constituent and supplementary events: all factors marked in the annotation
    all_events = {triple[0] for triple in row["triples_label_form"]}|{triple[2] for triple in row["triples_label_form"]}
    if "Inflation" in all_events:
        all_events.remove("Inflation")
    if not all_events:
        all_events = "*"
    return all_events


def get_feature_three(row):
    # relation: direction of influence on inflation
    feature = []
    for triple in row["triples_label_form"]:
        if triple and triple[2] == "Inflation":
            feature.append(triple[1])
    feature = list(set(chain(*feature)))
    if len(feature) == 0:
        narrative_feature = "*"
    elif len(feature) == 1:
        narrative_feature = {feature[0]}
    else:
        narrative_feature = {"Increases", "Decreases"}
    return narrative_feature


def get_feature_four(row):
    # graph: all marked events and relations, remove triples with empty relation
    raw_triples = row["triples_label_form"]
    if not raw_triples:
        graph = "*"
    else:
        graph = replace_empty_relation(raw_triples)
    return graph


def get_feature_five(row, event_category):
    # graph: all marked events and relations, remove triples with empty relation
    raw_triples = row["triples_label_form"]
    if not raw_triples:
        graph = "*"
    else:
        graph = replace_empty_relation(raw_triples)
    high_level_event_graph = []
    for g in graph:
        if g != "*":
            sub_category = low_level_event_to_high_level_event_map(g[0], event_category)
            obj_category = low_level_event_to_high_level_event_map(g[2], event_category)
            high_level_event_graph.append((sub_category, g[1], obj_category))
    high_level_event_graph = set(high_level_event_graph) if high_level_event_graph else "*"
    return high_level_event_graph


def get_feature_six(row):
    # graph: all events that go to inflation (in graph)
    raw_triples = row["triples_label_form"]
    if not raw_triples:
        graph = "*"
    else:
        graph = replace_empty_relation(raw_triples)
    high_level_event_graph = []
    for g in graph:
        if g != "*":
            sub_category = g[0]
            obj_category = g[2]
            if obj_category == "Inflation":
                high_level_event_graph.append((sub_category, g[1], obj_category))
    high_level_event_graph = set(high_level_event_graph) if high_level_event_graph else "*"
    return high_level_event_graph


def get_feature_seven(row, event_category):
    # graph: all events that go to inflation (in graph)
    raw_triples = row["triples_label_form"]
    if not raw_triples:
        graph = "*"
    else:
        graph = replace_empty_relation(raw_triples)
    high_level_event_graph = []
    for g in graph:
        if g != "*":
            sub_category = low_level_event_to_high_level_event_map(g[0], event_category)
            obj_category = low_level_event_to_high_level_event_map(g[2], event_category)
            high_level_event_graph.append((sub_category, g[1], obj_category))
            if obj_category == "Inflation":
                high_level_event_graph.append((sub_category, g[1], obj_category))
    high_level_event_graph = set(high_level_event_graph) if high_level_event_graph else "*"
    return high_level_event_graph


def low_level_event_to_high_level_event_map(event: str, event_category: dict):
    reverse_event_category = {v: key for key, value in event_category.items() for v in value}
    return reverse_event_category.get(event, event)


def to_tuple(obj):
    if isinstance(obj, list):
        return tuple(to_tuple(x) for x in obj)
    return obj

def replace_empty_relation(triples: list[tuple]):
    new_triples = []
    for triple in triples:
        # Alle Elemente rekursiv in Tupel umwandeln, falls nötig
        new_triples.append((
            to_tuple(triple[0]),
            to_tuple(triple[1]),
            to_tuple(triple[2])
        ))
    new_triples = set(new_triples)
    return new_triples


def get_distance_metric_map():
    distance_metric_map = {"lenient": [node_overlap_metric, graph_overlap_metric],
                           "strict": [nominal_metric, graph_edit_distance]}
    return distance_metric_map


def compute_iaa(df, project_id_list,
                feature_column="feature_one", empty_graph_indicator="*", annotator_list=None,
                distance_metric=node_overlap_metric, metric_type="lenient", graph_type=nx.Graph,
                forced=False):
    print(feature_column)
    data = [df[df["annotator"] == annotator_id][feature_column].to_list() for annotator_id in project_id_list]
    save_path = f"./export/{metric_type}_distance_matrix_{feature_column}_{'_'.join([str(annotator) for annotator in annotator_list])}.npy"

    if not forced and Path(save_path).exists():
        distance_matrix = np.load(save_path)
    else:
        distance_matrix = compute_distance_matrix(df_task2_annotation, feature_column=feature_column,
                                                  graph_distance_metric=distance_metric,
                                                  empty_graph_indicator=empty_graph_indicator, save_path=save_path,
                                                  graph_type=graph_type, timeout=60)

    alpha = compute_alpha(data, distance_matrix=distance_matrix, missing_items=empty_graph_indicator)
    print(f"{metric_type} distance metric: {alpha:.4f}")
    return alpha

if __name__ == "__main__":
    setup()
    # Create an ArgumentParser for project_list, and forced args
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project_list', nargs="+", type=int)
    parser.add_argument('-f', '--forced', action='store_true', default=False)
    parser.add_argument('--redownload', action='store_true', default=False)
    args = parser.parse_args()

    LABEL_STUDIO_URL = 'https://annotation.hitec.skynet.coypu.org/'
    API_KEY = '87023e8a5f12dee9263581bc4543806f80051133'
    if args.project_list is None:
        raise ValueError("You must provide at least one project ID using --project_list or -p.")
    project_id_list = args.project_list
    annotator_list = [[project_id] for project_id in project_id_list]

    feature_cols = ["feature_one", "feature_two", "feature_three", "feature_four", "feature_five", "feature_six", "feature_seven"]
    empty_graph_indicator = "*"
    alpha_store = {feature: {"lenient": None, "strict": None} for feature in feature_cols}

    # crawl project
    project_annotations = get_task_2_annotation_json(project_id_list)
    inner_id = [project_annotations[i]["inner_id"] for i in range(len(project_annotations))]
    text = [project_annotations[i]["data"]["text"] for i in range(len(project_annotations))]
    project_id = [project_annotations[i]["project"] for i in range(len(project_annotations))]
    results = [project_annotations[i]["annotations"][0]["result"] for i in range(len(project_annotations))]

    task2_annotation_dict = {"annotator": [], "item_id": [], "text": [], "triples": [], "triples_surface_form": [],
                             "triples_label_form": []}

    for i, result in enumerate(results):
        task2_annotation_dict["annotator"].append(project_id[i])
        task2_annotation_dict["item_id"].append(inner_id[i])
        task2_annotation_dict["text"].append(text[i])
        triples, triples_surface_form, triples_label_form = get_triples(result)
        task2_annotation_dict["triples"].append(triples)
        task2_annotation_dict["triples_surface_form"].append(triples_surface_form)
        task2_annotation_dict["triples_label_form"].append(triples_label_form)

    df_task2_annotation = pd.DataFrame.from_dict(task2_annotation_dict)
    df_task2_annotation.to_csv("./export/task_2_annotation_raw.csv", index=False)
    
    # remove documents with empty triples
    df_task2_annotation = df_task2_annotation[df_task2_annotation["triples"].apply(lambda x: bool(x) and x != "*")].reset_index(drop=True) #bool(x) and x != "*" filters out empty graphs

    # take only those that have been fully annotated by all annotators
    item_id_counts = df_task2_annotation.groupby("item_id")["annotator"].nunique()
    item_ids_with_multiple_annotators = item_id_counts[item_id_counts > 1].index
    df_task2_annotation = df_task2_annotation[df_task2_annotation["item_id"].isin(item_ids_with_multiple_annotators)].reset_index(drop=True)




    # create features
    event_category = {"Inflation": ["Inflation"],
                      "Nachfrage": ["Staatsausgaben", "Geldpolitik", "Aufgestaute Nachfrage", "Nachfrageverschiebung",
                                 "Nachfrage (Rest)"],
                      "Angebot": ["Lieferkettenprobleme", "Arbeitskräftemangel", "Lebensmittelpreise", "Hohe Energiepreise", "Angebot (Rest)",
                                 'Löhne', "Wohnraum"],
                      "Andere": ["Pandemie", "Politisches Missmanagement", "Inflationserwartungen", "Basiseffekt",
                                        "Hohe Staatsschulden", "Steuererhöhungen", "Preistreiberei", "Klimawandel",
                                        "Krieg", "Geopolitik", "Migration", 'Zölle', 'Ökonomische Krise']}

    df_task2_annotation["feature_one"] = df_task2_annotation.apply(get_feature_one, axis=1)
    df_task2_annotation["feature_two"] = df_task2_annotation.apply(get_feature_two, axis=1)
    #df_task2_annotation["feature_three"] = df_task2_annotation.apply(get_feature_three, axis=1)
    df_task2_annotation["feature_four"] = df_task2_annotation.apply(get_feature_four, axis=1)
    df_task2_annotation["feature_five"] = df_task2_annotation.apply(get_feature_five, event_category=event_category, axis=1)
    df_task2_annotation["feature_six"] = df_task2_annotation.apply(get_feature_six, axis=1)
    df_task2_annotation["feature_seven"] = df_task2_annotation.apply(get_feature_seven, event_category=event_category, axis=1)

    df_task2_annotation.to_csv("./export/task_2_annotation_survey.csv", index=False)
    

    # configurations for IAA computing
    configurations = {"feature_one": {"graph_type": nx.Graph, "graph_distance_metric": {"lenient": node_overlap_metric, "strict": nominal_metric}},
                      "feature_two": {"graph_type": nx.Graph, "graph_distance_metric": {"lenient": node_overlap_metric, "strict": nominal_metric}},
                      #"feature_three": {"graph_type": nx.Graph, "graph_distance_metric": {"lenient": node_overlap_metric, "strict": nominal_metric}},
                      "feature_four": {"graph_type": nx.DiGraph, "graph_distance_metric": {"lenient": graph_overlap_metric, "strict": graph_edit_distance}},
                      "feature_five": {"graph_type": nx.MultiDiGraph, "graph_distance_metric": {"lenient": graph_overlap_metric, "strict": graph_edit_distance}},
                      "feature_six": {"graph_type": nx.DiGraph, "graph_distance_metric": {"lenient": graph_overlap_metric, "strict": graph_edit_distance}},
                      "feature_seven": {"graph_type": nx.MultiDiGraph, "graph_distance_metric": {"lenient": graph_overlap_metric, "strict": graph_edit_distance}}}

    forced = args.forced
    for feature_column, configs in configurations.items():
        graph_type = configs["graph_type"]
        for metric_type, metric in configs["graph_distance_metric"].items():
            alpha = compute_iaa(df=df_task2_annotation, project_id_list=project_id_list,
                                feature_column=feature_column, annotator_list=annotator_list,
                                empty_graph_indicator=empty_graph_indicator,
                                distance_metric=metric, metric_type=metric_type,
                                graph_type=graph_type, forced=forced)
            alpha_store[feature_column][metric_type] = alpha

    with open(f"./export/alpha-{'-'.join([str(annotator) for annotator in annotator_list])}.json", "w") as f:
        json.dump(alpha_store, f)



from collections import defaultdict

# Features und Metriken, die du betrachten willst
features = ["feature_four", "feature_six"]
metrics = ["lenient", "strict"]

# Hilfsfunktion zum Vergleich (Agreement) für ein Feature und eine Metrik
def check_agreement(df, feature, metric):
    # Gruppiere nach item_id und sammle die Werte aller Annotatoren
    agreement_item_ids = []
    for item_id, group in df.groupby("item_id"):
        values = group[feature].tolist()
        # Prüfe auf vollständige Übereinstimmung (alle Werte gleich)
        if all(v == values[0] for v in values):
            agreement_item_ids.append(item_id)
    return agreement_item_ids

# Ergebnisse speichern
agreement_results = defaultdict(dict)

for feature in features:
    for metric in metrics:
        # Hier ggf. die richtige Spalte/Metrik auswählen, falls du verschiedene Spalten für lenient/strict hast
        # In deinem Fall sind die Features gleich, die Metrik wirkt sich auf die spätere Auswertung aus
        agreement_item_ids = check_agreement(df_task2_annotation, feature, metric)
        # Extrahiere die Zeilen mit Agreement
        df_agreement = df_task2_annotation[df_task2_annotation["item_id"].isin(agreement_item_ids)]
        # Speichern oder exportieren
        agreement_results[feature][metric] = df_agreement
        df_agreement.to_csv(f"./export/agreement_{feature}_{metric}.csv", index=False)

