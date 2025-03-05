import json
from pathlib import Path

import pandas as pd
import requests
import numpy as np
from label_studio_sdk.client import LabelStudio
from sklearn.model_selection import train_test_split
import krippendorff
from collections import Counter
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset
import evaluate


class InflationNarrative(object):
    def __init__(self, pull_from_label_studio=True):
        self.LABEL_STUDIO_URL = 'https://annotation.hitec.skynet.coypu.org/'
        self.API_KEY = '87023e8a5f12dee9263581bc4543806f80051133'
        self.client = LabelStudio(base_url=self.LABEL_STUDIO_URL, api_key=self.API_KEY)
        self.project_id_list = [5, 7, 8]
        self.number_documents = self.client.projects.get(id=5).finished_task_number + int(
            self.client.projects.get(id=5).queue_total)
        self.label2id_map = {"inflation-cause-dominant": 0, "inflation-related": 1, "non-inflation-related": 2}
        self.training_data_inner_id = None
        self.training_data_label = None
        self.df = self.instantiate(pull_from_label_studio=pull_from_label_studio)

    def is_annotation_done(self, project_id):
        num_unfinished = self.client.projects.get(id=project_id).queue_total
        is_done = num_unfinished == 0
        return is_done, num_unfinished

    def check_annotation_status(self):
        with open("./export/annotation_status.txt", "w") as file:
            for project_id in self.project_id_list:
                is_done, num_unfinished = self.is_annotation_done(project_id)
                file.write(f"annotator {project_id}: {num_unfinished} document(s) to annotate\n")
                print(f"annotator {project_id}: {num_unfinished} document(s) to annotate.")

    def setup_directory(self):
        for folder in ["./logs/", "./exports/"]:
            path = Path(folder)
            if not path.exists():  # Check existence
                path.mkdir(parents=True)

    def instantiate(self, pull_from_label_studio=True):
        """
        Instantiation step of the class.
        1) check unfinished documents for each annotator
        2) create df for storing annotated result, and write to disk
        """
        self.setup_directory()
        self.check_annotation_status()
        response = self.export_project_to_json(project_id=1, write_to_dist=True)
        inner_id = [document["inner_id"] for document in response]
        text = [document["data"]["text"] for document in response]
        if Path("./export/task_1_annotation.csv").exists():
            df = pd.read_csv("./export/task_1_annotation.csv")
        else:
            df = pd.DataFrame.from_dict({"inner_id": inner_id, "text": text})
            df = df.sort_values(by='inner_id', ascending=True)
            df.to_csv("./export/task_1_annotation.csv", index=False)
        if pull_from_label_studio:
            for annotator_id in self.project_id_list:
                self.get_task_1_annotation(annotator_id)
                print(self.get_project_title_from_id(annotator_id))
        return df

    def get_project_title_from_id(self, project_id):
        return self.client.projects.get(id=project_id).title

    def export_project_to_json(self, project_id, write_to_dist=True):
        url = f"{self.LABEL_STUDIO_URL}api/projects/{project_id}/export"
        headers = {"Authorization": f"Token  {self.API_KEY}"}
        response = requests.get(url, headers=headers)
        export = response.json()
        if write_to_dist:
            with open(f'./export/annotation_project_{project_id}.json', 'w') as f:
                json.dump(export, f)
        return export

    @staticmethod
    def get_data_by_inner_id(inner_id, json_data):
        for data in json_data:
            if data.get("inner_id") == inner_id:
                return data

    def extract_annotation_from_doc(self, annotations_json, annotator_id):
        inner_id_list = []
        label_list = []

        for document in annotations_json:
            inner_id = document.get("inner_id", None)
            inner_id_list.append(inner_id)
            for annotation in document["annotations"]:
                for anno in annotation["result"]:
                    if anno.get("from_name", None) == "document type":
                        result = anno["value"]["choices"][0]
                        label_list.append(result)
        print(f"inner_id_list {len(inner_id_list)}, annotator_{annotator_id}: {len(label_list)}")
        df = pd.DataFrame.from_dict({"inner_id": inner_id_list, f"label": label_list})
        df = df.sort_values(by='inner_id', ascending=True)
        return df

    def get_task_1_annotation(self, annotator_id):
        annotations_json = self.export_project_to_json(annotator_id, True)
        df = pd.read_csv("./export/task_1_annotation.csv")
        df[f"annotator_{annotator_id}"] = self.extract_annotation_from_doc(annotations_json, annotator_id)["label"]
        df.to_csv("./export/task_1_annotation.csv", index=False)
        print(f"Updated annotation from user {annotator_id}, result saved in ./export/task_1_annotation.csv")
        task_1_annotation = df.set_index('inner_id')[f"annotator_{annotator_id}"].to_dict()
        return task_1_annotation

    def compute_agreement(self, annotator_list=None):
        if annotator_list is None:
            annotator_list = self.project_id_list
        df = pd.read_csv("./export/task_1_annotation.csv")
        for annotator_id in annotator_list:
            # get only overlapped annotations
            df = df[df[f"annotator_{annotator_id}"].notnull()]
        annotations: list[list] = [df[f"annotator_{annotator_id}"].tolist() for annotator_id in annotator_list]
        annotations_numeric = [[self.label2id_map.get(label, -1) for label in annotation] for annotation in annotations]
        df["label"] = self.get_majority_vote(df)
        df.to_csv("./export/task_1_annotation.csv", index=False)
        irr = self.compute_task_1_agreement(annotations_numeric, metric="krippendorff")
        print(irr)
        return irr

    @staticmethod
    def get_text_from_inner_id(data, inner_id):
        for document in data:
            tmp_inner_id = document.get("inner_id", None)
            if int(tmp_inner_id) == inner_id:
                text = document["data"]["text"]
                return text

    def get_text_from_indices(self, indices):
        with open("./export/annotation_project_5.json", "r") as f:
            docs = json.load(f)
        text = [self.get_text_from_inner_id(docs, inner_id) for inner_id in indices]
        return text

    @staticmethod
    def compute_task_1_agreement(annotation_list_of_dict, metric: str = "krippendorff"):
        if metric == "krippendorff":
            return krippendorff.alpha(reliability_data=annotation_list_of_dict)
        else:
            raise NotImplemented

    @staticmethod
    def create_training_data_from_annotation(save_to_disk=True):
        df = pd.read_csv("./export/task_1_annotation.csv", index_col=0)

        df_train, df_valid = train_test_split(df, test_size=0.3)
        df_valid, df_test = train_test_split(df_valid, test_size=0.66)
        if save_to_disk:
            df_train.to_csv("./export/task_1_train.csv")
            df_valid.to_csv("./export/task_1_valid.csv")
            df_test.to_csv("./export/task_1_test.csv")
        return df_train, df_valid, df_test

    def get_majority_vote(self, df):
        col_names = [f"annotator_{i}" for i in self.project_id_list]
        majority_labels = []

        for row in df[col_names].values:
            counter = Counter(row)
            most_common = counter.most_common()

            # Get highest frequency count
            max_count = most_common[0][1]
            top_labels = [label for label, count in most_common if count == max_count]

            # Handle tie-breaks
            if len(top_labels) == 1:
                majority_labels.append(top_labels[0])  # Clear winner
            else:
                majority_labels.append("inflation-related")  # no winner, pick inflation-related

        return majority_labels


    def train_sequence_classifier(self):
        model_names = ["distilbert/distilbert-base-uncased", "ProsusAI/finbert",
                       "microsoft/deberta-v3-base", "FacebookAI/roberta-base"]
        train = pd.read_csv("./export/task_1_train.csv")
        valid = pd.read_csv("./export/task_1_valid.csv")
        test = pd.read_csv("./export/task_1_test.csv")
        train['label'] = train['label'].replace(self.label2id_map)
        valid['label'] = valid['label'].replace(self.label2id_map)
        test['label'] = test['label'].replace(self.label2id_map)

        train = Dataset.from_pandas(train)
        valid = Dataset.from_pandas(valid)
        test = Dataset.from_pandas(test)
        id2label_map = {value: key for key, value in self.label2id_map.items()}

        def preprocess_function(examples):
            return tokenizer(examples["text"], truncation=True)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels, average="weighted")

        for model_name in model_names:
            name = model_name.split('/')[-1]
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=3, id2label=id2label_map, label2id=self.label2id_map)
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            tokenized_train = train.map(preprocess_function, batched=True)
            tokenized_valid = valid.map(preprocess_function, batched=True)
            tokenized_test = test.map(preprocess_function, batched=True)

            # Training Arguments
            training_args = TrainingArguments(
                output_dir=f"./results/{name}",
                eval_strategy="epoch",
                save_strategy="epoch",
                per_device_train_batch_size=64,
                per_device_eval_batch_size=64,
                num_train_epochs=20,
                weight_decay=0.01,
                logging_dir=f"./logs/{name}",
                eval_steps=500,
            )
            # Setup evaluation
            metric = evaluate.load("f1")

            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_valid,
                processing_class=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )

            # Train Model
            trainer.train()

            # Test model
            model.eval()
            predictions = trainer.predict(tokenized_test)
            print(predictions.metrics)
            with open(f"./logs/{name}.txt", "w") as file:
                file.write(f"{name} - F1: {predictions.metrics['test_f1']}\n")


if __name__ == "__main__":
    inflation_narrative = InflationNarrative(pull_from_label_studio=False)
    #inflation_narrative.compute_agreement([5, 7, 8, 9])
    #inflation_narrative.compute_agreement([5, 7])
    #inflation_narrative.compute_agreement([5, 8])
    #inflation_narrative.compute_agreement([7, 8])
    inflation_narrative.compute_agreement([5, 7, 8])

    inflation_narrative.create_training_data_from_annotation()
    inflation_narrative.train_sequence_classifier()
    #inflation_narrative.compute_agreement([5, 7, 9])
    #inflation_narrative.compute_agreement([5, 8, 9])
    #inflation_narrative.compute_agreement([7, 8, 9])
    #inflation_narrative.compute_agreement([5, 7])
    #inflation_narrative.compute_agreement([5, 8])
    #inflation_narrative.compute_agreement([5, 9])
    #inflation_narrative.compute_agreement([7, 8])
    #inflation_narrative.compute_agreement([7, 9])
    #inflation_narrative.compute_agreement([8, 9])




