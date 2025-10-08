import os
import json
from pathlib import Path
import pandas as pd
import requests
import numpy as np
from label_studio_sdk.client import LabelStudio
from sklearn.model_selection import train_test_split
import krippendorff
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from label_studio_sdk import Client

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"

ls = Client(url="https://annotation.hitec.skynet.coypu.org/", api_key="87023e8a5f12dee9263581bc4543806f80051133")
# Get all projects
projects = ls.get_projects()
for project in projects:
    print(f"Project ID: {project.id}, Title: {project.title}")


import json

# Datei öffnen und laden
#with open('./export/survey_annotation_project_20.json', 'r', encoding='utf-8') as f:
#    data = json.load(f)

# Jetzt ist data ein Python-Objekt (z.B. dict oder list)
# print(data)

seed = 118
# Set random seed for reproducibility
LABEL_STUDIO_URL = 'https://annotation.hitec.skynet.coypu.org/'
API_KEY = '87023e8a5f12dee9263581bc4543806f80051133'
client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)
project_id_list = [20,21]
# assuming `client` is an SDK client:

number_documents = client.projects.get(id=20).finished_task_number + int(
            client.projects.get(id=20).queue_total)
label2id_map = {"Keine sinnvolle Antwort": 0, "Gründe der Inflation": 1, "kausales Inflationsnarrative": 2}
training_data_inner_id = None
training_data_label = None



def is_annotation_done(project_id): # check if annotation is done
    num_unfinished = client.projects.get(id=project_id).queue_total
    is_done = num_unfinished == 0
    return is_done, num_unfinished

is_annotation_done(20) # still to annotate
is_annotation_done(21)


with open("./export/survey_annotation_status.txt", "w") as file:
    for project_id in project_id_list:
        is_done, num_unfinished = is_annotation_done(project_id)
        file.write(f"annotator {project_id}: {num_unfinished} document(s) to annotate\n")
        print(f"annotator {project_id}: {num_unfinished} document(s) to annotate.")

# set up directory structure
for folder in ["./logs/", "./export/"]:
    path = Path(folder)
    if not path.exists():  # Check existence
        path.mkdir(parents=True)

# define function to export to json and create dataframe
def export_project_to_json(project_id, write_to_dist=True):
    url = f"{LABEL_STUDIO_URL}api/projects/{project_id}/export"
    headers = {"Authorization": f"Token  {API_KEY}"}
    response = requests.get(url, headers=headers)
    export = response.json()
    if write_to_dist:
        with open(f'./export/survey_annotation_project_{project_id}.json', 'w') as f:
            json.dump(export, f)
    return export

# define function to extract annotation from document
def extract_annotation_from_doc(annotations_json, annotator_id):
    inner_id_list = []
    label_list = []

    for document in annotations_json:
        inner_id = document.get("inner_id", None) # Get inner_id from the document
        found_label = False # Append inner_id to the list
        for annotation in document["annotations"]:
            for anno in annotation["result"]:
                if anno.get("from_name", None) == "document type":
                    result = anno["value"]["choices"][0]
                    inner_id_list.append(inner_id)
                    label_list.append(result)
                    found_label = True
                    break
            if found_label:
                break
    print(f"inner_id_list {len(inner_id_list)}, annotator_{annotator_id}: {len(label_list)}")
    df = pd.DataFrame.from_dict({"inner_id": inner_id_list, f"label": label_list})
    df = df.sort_values(by='inner_id', ascending=True)
    return df

def get_task_1_annotation(annotator_id, filler="MISSING"):
    annotations_json = export_project_to_json(annotator_id, True)
    df = pd.read_csv("./export/task_1_annotation_survey.csv")
    # Extract Labels from JSON
    annot_df = extract_annotation_from_doc(annotations_json, annotator_id)
    # Inner ID as Index
    annot_series = annot_df.set_index("inner_id")["label"]
    # Fill missing values with filler
    df[f"annotator_{annotator_id}"] = annot_series.reindex(df["inner_id"], fill_value=filler).values
    df.to_csv("./export/task_1_annotation_survey.csv", index=False, encoding='utf-8')
    df.to_pickle("./export/task_1_annotation_survey.pkl")
    print(f"Updated annotation from user {annotator_id}, result saved in ./export/task_1_annotation_survey.csv")
    task_1_annotation = df.set_index('inner_id')[f"annotator_{annotator_id}"].to_dict()
    return task_1_annotation

# get project title from project id
def get_project_title_from_id(project_id): # get project title from project id
    print(client.projects.get(id=project_id).title)
    return client.projects.get(id=project_id).title
        
def instantiate(pull_from_label_studio=True): # export project to json and create dataframe

    response = export_project_to_json(project_id=21, write_to_dist=True)
    inner_id = [document["inner_id"] for document in response]
    text = [document["data"]["text"] for document in response]
    if Path("./export/task_1_annotation_survey.csv").exists():
        df = pd.read_csv("./export/task_1_annotation_survey.csv")
    else:
        df = pd.DataFrame.from_dict({"inner_id": inner_id, "text": text})
        df = df.sort_values(by='inner_id', ascending=True)
        df.to_csv("./export/task_1_annotation_survey.csv", index=False, encoding='utf-8')
        df.to_pickle("./export/task_1_annotation_survey.pkl")
        
    if pull_from_label_studio:
        for annotator_id in project_id_list:
                get_task_1_annotation(annotator_id)
                print(get_project_title_from_id(annotator_id))
    return df
        
data = instantiate(pull_from_label_studio=True)
data.head()
len(data)

def get_data_by_inner_id(inner_id, json_data):
    for data in json_data:
        if data.get("inner_id") == inner_id:
            return data


def get_majority_vote(df):
    col_names = [f"annotator_{i}" for i in project_id_list]
    df_annotations = df[col_names]
    # Find rows where all annotators agree
    agreement_rows = df_annotations.eq(df_annotations.iloc[:, 0], axis=0).all(axis=1)
        
    # Find rows where not all annotators agree 
    disagreement_rows = df_annotations.eq(df_annotations.iloc[:, 0], axis=0).all(axis=1) == False

    # Get indices of agreeing/not agreeing rows
    agreeing_indices = df_annotations.index[agreement_rows].tolist()
    disagreement_indices = df_annotations.index[disagreement_rows].tolist()

    agreeing_df = df.loc[agreeing_indices]
    disgreeing_df = df.loc[disagreement_indices]
    print(f"agreeing indices: {len(agreeing_indices)}, disagreeing indices: {len(disagreement_indices)}")
    agreeing_df.to_csv("./export/survey_all_agreeing_answers.csv", index=False)
    disgreeing_df.to_csv("./export/survey_all_disagreeing_answers.csv", index=False)

    majority_labels = []
    has_winner = 0

    for i, row in enumerate(df[col_names].values):
        counter = Counter(row)
        most_common = counter.most_common()

        # Get highest frequency count
        max_count = most_common[0][1]
        top_labels = [label for label, count in most_common if count == max_count]

        # Handle tie-breaks
        if len(top_labels) == 1:
            majority_labels.append(top_labels[0])  # Clear winner
            has_winner += 1
        else:
            majority_labels.append("Gründe der Inflation")  # no winner, pick inflation-related
    print(f"has winner ratio: {has_winner/len(df)}\nall agreeing count: {len(agreeing_indices)}")
    return majority_labels

def compute_task_1_agreement(annotation_list_of_dict, metric: str = "krippendorff"):
    if metric == "krippendorff":
        return krippendorff.alpha(reliability_data=annotation_list_of_dict)
    else:
        raise NotImplemented
    
def compute_agreement(annotator_list=None):
    if annotator_list is None:
        annotator_list = project_id_list
    df = pd.read_csv("./export/task_1_annotation_survey.csv")
    for annotator_id in annotator_list:
        # get only overlapped annotations
        df = df[df[f"annotator_{annotator_id}"].notnull()]
        df = df[df[f"annotator_{annotator_id}"] != "MISSING"]
    annotations: list[list] = [df[f"annotator_{annotator_id}"].tolist() for annotator_id in annotator_list]
    annotations_numeric = [[label2id_map.get(label, -1) for label in annotation] for annotation in annotations]
    labels = get_majority_vote(df)
    df["label"] = labels
    df.to_csv("./export/survey_task_1_annotation.csv", index=False)
    irr = compute_task_1_agreement(annotations_numeric, metric="krippendorff")
    
    label_remapping = {"Gründe der Inflation": "Inflationsnarrative", "kausales Inflationsnarrativ": "Inflationsnarrative"}
    df["label"] = df["label"].replace(label_remapping)
    
    remapped_annotations = [[label_remapping.get(label, label) for label in annotator_labels]
    for annotator_labels in annotations]
    
    remapped_annotations_numeric = [
    [label2id_map.get(label, -1) for label in annotator_labels]
    for annotator_labels in remapped_annotations]

    irr_2 = compute_task_1_agreement(remapped_annotations_numeric, metric="krippendorff")
    
    print(irr)
    print(irr_2)
    return irr, irr_2

agreement = compute_agreement(annotator_list=project_id_list)



#if __name__ == "__main__":
#    seeds = [11, 22, 33, 44]
#    for seed in seeds:
#        random.seed(seed)
#        np.random.seed(seed)
#        inflation_narrative = InflationNarrative(pull_from_label_studio=True, seed=seed)
#        inflation_narrative.compute_agreement([20, 21])
        #inflation_narrative.create_training_data_from_annotation()
        #inflation_narrative.train_sequence_classifier()


