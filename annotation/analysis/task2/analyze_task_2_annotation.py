import json
from pathlib import Path
from itertools import chain
import pandas as pd
import requests
import numpy as np
import networkx as nx
from typing import Union, Optional, Callable
from krippendorrf_graph import (compute_alpha, graph_edit_distance, graph_overlap_metric,
                                nominal_metric, node_overlap_metric, compute_distance_matrix)


def export_project_to_json(project_id, write_to_dist=True):
    url = f"{LABEL_STUDIO_URL}api/projects/{project_id}/export"
    headers = {"Authorization": f"Token  {API_KEY}"}
    response = requests.get(url, headers=headers)
    export = response.json()
    if write_to_dist:
        with open(f'./export/annotation_task2_project_{project_id}.json', 'w') as f:
            json.dump(export, f)
    return export


def get_task_2_annotation_json(project_id_list):
    project_annotations = []
    for project_id in project_id_list:
        if Path(f'./export/annotation_task2_project_{project_id}.json').exists():
            with open(f'./export/annotation_task2_project_{project_id}.json', "r") as f:
                project_annotations.extend(json.load(f))
        else:
            project_annotations.extend(export_project_to_json(project_id))
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
    for triple in triples:
        subj_label_form = get_label_from_id(triple[0], results)
        obj_label_form = get_label_from_id(triple[2], results)
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
    all_events = {triple[0] for triple in row["triples_label_form"]}|{triple[0] for triple in row["triples_label_form"]}
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

def low_level_event_to_high_level_event_map(event: str, event_category):
    reverse_event_category = {v: key for key, value in event_category.items() for v in value}
    return reverse_event_category.get(event, "Miscellaneous")


def replace_empty_relation(triples: list[tuple]):
    new_triples = []
    for triple in triples:
        if not triple[1]:
            continue
        else:
            new_triples.append((triple[0], triple[1][0], triple[2]))
    new_triples = set(new_triples) if new_triples else "*"
    return new_triples


if __name__ == "__main__":
    LABEL_STUDIO_URL = 'https://annotation.hitec.skynet.coypu.org/'
    API_KEY = '87023e8a5f12dee9263581bc4543806f80051133'
    project_id_list = [11, 12, 13, 14]

    project_annotations = get_task_2_annotation_json(project_id_list)
    inner_id = [project_annotations[i]["inner_id"] for i in range(len(project_annotations))]
    project_id = [project_annotations[i]["project"] for i in range(len(project_annotations))]
    results = [project_annotations[i]["annotations"][0]["result"] for i in range(len(project_annotations))]

    task2_annotation_dict = {"annotator": [], "item_id": [], "triples": [], "triples_surface_form": [],
                             "triples_label_form": []}

    for i, result in enumerate(results):
        task2_annotation_dict["annotator"].append(project_id[i])
        task2_annotation_dict["item_id"].append(inner_id[i])
        triples, triples_surface_form, triples_label_form = get_triples(result)
        task2_annotation_dict["triples"].append(triples)
        task2_annotation_dict["triples_surface_form"].append(triples_surface_form)
        task2_annotation_dict["triples_label_form"].append(triples_label_form)

    df_task2_annotation = pd.DataFrame.from_dict(task2_annotation_dict)

    event_category = {"Inflation": ["Inflation"],
                      "Demand": ["Government Spending", "Monetary Policy", "Pent-up Demand", "Demand Shift",
                                 "Demand (residual)"],
                      "Supply": ["Supply Chain Issues", "Labor Shortage", "Energy Crisis", "Supply (residual)", "Wages"],
                      "Miscellaneous": ["Pandemic", "Mismanagement", "Russia-Ukraine War", "Inflation Expectations",
                                        "Base Effect", "Government Debt", "Tax Increases", "Price-Gouging",
                                        "Trade Balance", "Exchange Rates", "Medical Costs", "Food Prices",
                                        "Energy Prices", "War", "Transportation Costs", "Education Costs",
                                        "House Costs", 'Housing Costs', 'Food Prices', 'Transportation Costs',
                                        'Climate', 'Education Costs', 'War', 'Energy Prices']}

    df_task2_annotation["feature_one"] = df_task2_annotation.apply(get_feature_one, axis=1)
    df_task2_annotation["feature_two"] = df_task2_annotation.apply(get_feature_two, axis=1)
    df_task2_annotation["feature_three"] = df_task2_annotation.apply(get_feature_three, axis=1)
    df_task2_annotation["feature_four"] = df_task2_annotation.apply(get_feature_four, axis=1)
    df_task2_annotation["feature_five"] = df_task2_annotation.apply(get_feature_five, event_category=event_category,
                                                                    axis=1)
    df_task2_annotation.to_csv("./export/task_2_annotation.csv", index=False)


    # Feature 1
    data = [
        df_task2_annotation[df_task2_annotation["annotator"] == 11].feature_one.to_list(),
        df_task2_annotation[df_task2_annotation["annotator"] == 12].feature_one.to_list(),
        df_task2_annotation[df_task2_annotation["annotator"] == 13].feature_one.to_list(),
        df_task2_annotation[df_task2_annotation["annotator"] == 14].feature_one.to_list()
    ]
    empty_graph_indicator = "*"  # indicator for missing values
    feature_column = "feature_one"
    save_path = "./lenient_distance_matrix_feature_one.npy"
    graph_distance_metric = node_overlap_metric
    forced = True

    if not Path(save_path).exists() or forced:
        distance_matrix = compute_distance_matrix(df_task2_annotation, feature_column=feature_column,
                                                  graph_distance_metric=graph_distance_metric,
                                                  empty_graph_indicator=empty_graph_indicator, save_path=save_path,
                                                  graph_type=nx.Graph)
    else:
        distance_matrix = np.load(save_path)

    print("Lenient node metric: %.3f" % compute_alpha(data, distance_matrix=distance_matrix,
                                                      missing_items=empty_graph_indicator))

    save_path = "./strict_distance_matrix_feature_one.npy"
    graph_distance_metric = nominal_metric
    forced = True

    if not Path(save_path).exists() or forced:
        distance_matrix = compute_distance_matrix(df_task2_annotation, feature_column=feature_column,
                                                  graph_distance_metric=graph_distance_metric,
                                                  empty_graph_indicator=empty_graph_indicator, save_path=save_path,
                                                  graph_type=nx.Graph)
    else:
        distance_matrix = np.load(save_path)

    print("Strict node metric: %.3f" % compute_alpha(data, distance_matrix=distance_matrix,
                                                     missing_items=empty_graph_indicator))

    # feature 2
    data = [
        df_task2_annotation[df_task2_annotation["annotator"] == 11].feature_two.to_list(),
        df_task2_annotation[df_task2_annotation["annotator"] == 12].feature_two.to_list(),
        df_task2_annotation[df_task2_annotation["annotator"] == 13].feature_two.to_list(),
        df_task2_annotation[df_task2_annotation["annotator"] == 14].feature_two.to_list()
    ]
    empty_graph_indicator = "*"  # indicator for missing values
    feature_column = "feature_two"
    save_path = "./lenient_distance_matrix_feature_two.npy"
    graph_distance_metric = node_overlap_metric
    forced = True

    if not Path(save_path).exists() or forced:
        distance_matrix = compute_distance_matrix(df_task2_annotation, feature_column=feature_column,
                                                  graph_distance_metric=graph_distance_metric,
                                                  empty_graph_indicator=empty_graph_indicator, save_path=save_path,
                                                  graph_type=nx.Graph)
    else:
        distance_matrix = np.load(save_path)

    print("Lenient node metric: %.3f" % compute_alpha(data, distance_matrix=distance_matrix,
                                                      missing_items=empty_graph_indicator))

    save_path = "./strict_distance_matrix_feature_two.npy"
    graph_distance_metric = nominal_metric
    forced = True

    if not Path(save_path).exists() or forced:
        distance_matrix = compute_distance_matrix(df_task2_annotation, feature_column=feature_column,
                                                  graph_distance_metric=graph_distance_metric,
                                                  empty_graph_indicator=empty_graph_indicator, save_path=save_path,
                                                  graph_type=nx.Graph)
    else:
        distance_matrix = np.load(save_path)

    print("Strict node metric: %.3f" % compute_alpha(data, distance_matrix=distance_matrix,
                                                     missing_items=empty_graph_indicator))

    # feature 3
    data = [
        df_task2_annotation[df_task2_annotation["annotator"] == 11].feature_three.to_list(),
        df_task2_annotation[df_task2_annotation["annotator"] == 12].feature_three.to_list(),
        df_task2_annotation[df_task2_annotation["annotator"] == 13].feature_three.to_list(),
        df_task2_annotation[df_task2_annotation["annotator"] == 14].feature_three.to_list()
    ]
    empty_graph_indicator = "*"  # indicator for missing values
    feature_column = "feature_three"
    save_path = "./lenient_distance_matrix_feature_three.npy"
    graph_distance_metric = node_overlap_metric
    forced = True

    if not Path(save_path).exists() or forced:
        distance_matrix = compute_distance_matrix(df_task2_annotation, feature_column=feature_column,
                                                  graph_distance_metric=graph_distance_metric,
                                                  empty_graph_indicator=empty_graph_indicator, save_path=save_path,
                                                  graph_type=nx.Graph)
    else:
        distance_matrix = np.load(save_path)

    print("Lenient node metric: %.3f" % compute_alpha(data, distance_matrix=distance_matrix,
                                                      missing_items=empty_graph_indicator))

    save_path = "./strict_distance_matrix_feature_three.npy"
    graph_distance_metric = nominal_metric
    forced = True

    if not Path(save_path).exists() or forced:
        distance_matrix = compute_distance_matrix(df_task2_annotation, feature_column=feature_column,
                                                  graph_distance_metric=graph_distance_metric,
                                                  empty_graph_indicator=empty_graph_indicator, save_path=save_path,
                                                  graph_type=nx.Graph)
    else:
        distance_matrix = np.load(save_path)

    print("Strict node metric: %.3f" % compute_alpha(data, distance_matrix=distance_matrix,
                                                     missing_items=empty_graph_indicator))

    # feature 4
    data = [
        df_task2_annotation[df_task2_annotation["annotator"] == 11].feature_four.to_list(),
        df_task2_annotation[df_task2_annotation["annotator"] == 12].feature_four.to_list(),
        df_task2_annotation[df_task2_annotation["annotator"] == 13].feature_four.to_list(),
        df_task2_annotation[df_task2_annotation["annotator"] == 14].feature_four.to_list()
    ]
    empty_graph_indicator = "*"  # indicator for missing values
    feature_column = "feature_four"
    save_path = "./lenient_distance_matrix_feature_four.npy"
    graph_distance_metric = graph_overlap_metric
    forced = True

    if not Path(save_path).exists() or forced:
        distance_matrix = compute_distance_matrix(df_task2_annotation, feature_column=feature_column,
                                                  graph_distance_metric=graph_distance_metric,
                                                  empty_graph_indicator=empty_graph_indicator, save_path=save_path,
                                                  graph_type=nx.DiGraph)
    else:
        distance_matrix = np.load(save_path)

    print("Lenient node metric: %.3f" % compute_alpha(data, distance_matrix=distance_matrix,
                                                      missing_items=empty_graph_indicator))

    save_path = "./strict_distance_matrix_feature_four.npy"
    graph_distance_metric = graph_edit_distance
    forced = True

    if not Path(save_path).exists() or forced:
        distance_matrix = compute_distance_matrix(df_task2_annotation, feature_column=feature_column,
                                                  graph_distance_metric=graph_distance_metric,
                                                  empty_graph_indicator=empty_graph_indicator, save_path=save_path,
                                                  graph_type=nx.DiGraph)
    else:
        distance_matrix = np.load(save_path)

    print("Strict node metric: %.3f" % compute_alpha(data, distance_matrix=distance_matrix,
                                                     missing_items=empty_graph_indicator))

    # feature 5
    data = [
        df_task2_annotation[df_task2_annotation["annotator"] == 11].feature_five.to_list(),
        df_task2_annotation[df_task2_annotation["annotator"] == 12].feature_five.to_list(),
        df_task2_annotation[df_task2_annotation["annotator"] == 13].feature_five.to_list(),
        df_task2_annotation[df_task2_annotation["annotator"] == 14].feature_five.to_list()
    ]
    empty_graph_indicator = "*"  # indicator for missing values
    feature_column = "feature_five"
    save_path = "./lenient_distance_matrix_feature_five.npy"
    graph_distance_metric = graph_overlap_metric
    forced = True

    if not Path(save_path).exists() or forced:
        distance_matrix = compute_distance_matrix(df_task2_annotation, feature_column=feature_column,
                                                  graph_distance_metric=graph_distance_metric,
                                                  empty_graph_indicator=empty_graph_indicator, save_path=save_path,
                                                  graph_type=nx.MultiDiGraph)
    else:
        distance_matrix = np.load(save_path)

    print("Lenient node metric: %.3f" % compute_alpha(data, distance_matrix=distance_matrix,
                                                      missing_items=empty_graph_indicator))

    save_path = "./strict_distance_matrix_feature_five.npy"
    graph_distance_metric = graph_edit_distance
    forced = True

    if not Path(save_path).exists() or forced:
        distance_matrix = compute_distance_matrix(df_task2_annotation, feature_column=feature_column,
                                                  graph_distance_metric=graph_distance_metric,
                                                  empty_graph_indicator=empty_graph_indicator, save_path=save_path,
                                                  graph_type=nx.MultiDiGraph)
    else:
        distance_matrix = np.load(save_path)

    print("Strict node metric: %.3f" % compute_alpha(data, distance_matrix=distance_matrix,
                                                     missing_items=empty_graph_indicator))
