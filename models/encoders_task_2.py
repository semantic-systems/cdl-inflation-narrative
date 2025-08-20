from abc import abstractmethod
import ast
from collections import Counter
from itertools import chain
from pathlib import Path
import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
import evaluate
import sys
sys.path.append('./')
from models.base_models import CustomTrainerWeightedCELoss


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"


class Classification(object):
    def __init__(self, csv_path, split_ratio: list[float], seed=11):
        self.setup_directory()
        self.task_name = "classification"
        self.save_path = f"./data/annotated/task_2_{self.task_name}.csv"
        self.seed = seed
        self.project_id_list = [11, 12, 13, 14]
        self.df = pd.read_csv(csv_path)
        self.split_ratio = split_ratio
        self.df_aggregated = None
        self.df_train = None
        self.df_test = None
        self.label_distribution = None
        self.label_weights = None

    def setup_directory(self):
        for folder in ["./logs/", "./export/"]:
            path = Path(folder)
            if not path.exists():  # Check existence
                path.mkdir(parents=True)

    @property
    def feature_col(self):
        return None

    @property
    def label2id_map(self):
        return {}

    @abstractmethod
    def aggregate(self, df, forced=False, save_path=None):
        raise NotImplementedError

    def split(self, df, split_ratio, forced=False, save_path="./data/annotated/"):
        train_path = Path(save_path, f"task_2_{self.task_name}_train.csv")
        test_path = Path(save_path, f"task_2_{self.task_name}_test.csv")
        if not forced and train_path.exists() and test_path.exists():
            return pd.read_csv(train_path, index_col=False), pd.read_csv(test_path, index_col=False)
        else:
            assert sum(split_ratio) == 1.0, "Split ratios must sum to 1.0"

            df = shuffle(df, random_state=self.seed).reset_index(drop=True)
            df_train, df_test = train_test_split(df, test_size=split_ratio[-1], random_state=self.seed, stratify=df[['aggregated_label']])

            df_train.to_csv(train_path, index_label=False)
            df_test.to_csv(test_path, index_label=False)
            return df_train, df_test

    def get_weights_inverse_num_of_samples(self, label_counts):
        #label_counts = list(self.label_distribution.values())
        no_of_classes = len(label_counts)
        weights_for_samples = [1 / label_counts[label_name] for label_name, label_index in self.label2id_map.items()]
        return [weights_for_sample / sum(weights_for_samples) * no_of_classes for weights_for_sample in weights_for_samples]

    def train(self):
        model_names = {"distilbert/distilbert-base-uncased": 64,
                       #"ProsusAI/finbert": 64,
                       "FacebookAI/roberta-base": 64,
                       "google-bert/bert-base-uncased": 64,
                       "worldbank/econberta-fs": 8,
                       "worldbank/econberta": 4,
                       #"MAPAi/InflaBERT": 8, 
                       }
        train = pd.read_csv(f"./data/annotated/task_2_{self.task_name}_train.csv")
        test = pd.read_csv(f"./data/annotated/task_2_{self.task_name}_test.csv")
        train['label'] = train['aggregated_label'].replace(self.label2id_map)
        test['label'] = test['aggregated_label'].replace(self.label2id_map)

        train = Dataset.from_pandas(train)
        test = Dataset.from_pandas(test)
        id2label_map = {value: key for key, value in self.label2id_map.items()}

        def preprocess_function(examples):
            return tokenizer(examples["text"], truncation=True, padding=True)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            f1_weighted = f1_score(labels, predictions, average="weighted")
            f1_macro = f1_score(labels, predictions, average="macro")
            f1_micro = f1_score(labels, predictions, average="micro")
            return {'f1_weighted': f1_weighted, "f1_macro": f1_macro, "f1_micro": f1_micro}
        
        for model_name, batch_size in model_names.items():
            name = f"{model_name.split('/')[-1]}-{self.task_name}"
            print(name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=self.num_labels, id2label=id2label_map, label2id=self.label2id_map)
            model.resize_token_embeddings(len(tokenizer))
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            tokenized_train = train.map(preprocess_function, batched=True)
            tokenized_test = test.map(preprocess_function, batched=True)

            # Training Arguments
            training_args = TrainingArguments(
                output_dir=f"./results/{name}",
                eval_strategy="epoch",
                save_strategy="best",
                learning_rate=2e-5,
                load_best_model_at_end=True,
                metric_for_best_model="f1_macro",
                per_device_train_batch_size=batch_size,
                overwrite_output_dir=True,
                num_train_epochs=20,
                weight_decay=0.01,
                logging_dir=f"./logs/{name}"
            )

            # Trainer
            trainer = CustomTrainerWeightedCELoss(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_test,
                processing_class=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                label_weights=self.label_weights)

            # Train Model
            trainer.train()
            model.save_pretrained(f"./logs/{name}/model", from_pt=True)

            # Test model
            predictions = trainer.predict(tokenized_test)

            # Extract predicted labels
            predicted_logits = predictions.predictions
            predicted_labels = np.argmax(predicted_logits, axis=-1)

            # Convert label indices back to label names
            decoded_predictions = [id2label_map[label] for label in predicted_labels]
            df_pred = pd.read_csv(f"./data/annotated/task_2_{self.task_name}_test.csv")
            df_pred["prediction"] = decoded_predictions

            df_pred.to_csv(f"./logs/{name}/prediction_seed_{self.seed}.csv", index=False)
            target_names = list(self.label2id_map.keys())
            report = classification_report(df_pred["aggregated_label"], df_pred["prediction"], target_names=target_names)
            print(model_name)
            print(report)
            with open(f"./logs/{name}/test_metric_seed_{self.seed}.txt", "w") as file:
                file.write(report)

            del model
            del trainer

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def get_majority_vote(self, df, no_winner_label="Mixed"):
        col_names = df.columns

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
                majority_labels.append(no_winner_label)  # no winner, pick inflation-related
        print(f"has winner ratio: {has_winner / len(df)}\n")
        return majority_labels


class DirectionClassification(Classification):
    def __init__(self, csv_path, split_ratio: list[float], forced=False, seed=11):
        super().__init__(csv_path, split_ratio, seed)
        self.task_name = "direction_classification"
        self.df_aggregated = self.aggregate(self.df, forced=forced, save_path=self.save_path)
        self.df_train, self.df_test = self.split(self.df_aggregated, split_ratio, forced=forced, save_path="./data/annotated/")
        self.num_labels = len(self.label2id_map)
        self.label_distribution = dict(Counter(self.df_aggregated.aggregated_label))
        print(f"label distribution: {self.label_distribution}")
        self.label_weights = self.get_weights_inverse_num_of_samples(self.label_distribution)
        print(f"label weights: {self.label_weights}")

    @property
    def feature_col(self):
        return "feature_three"

    @property
    def label2id_map(self):
        #return {"Decreases": 0, "Increases": 1, "Mixed": 2}
        return {"Increases": 0, "Mixed": 1}

    def aggregate(self, df, forced=False, save_path="./data/annotated/task_2_direction_classification.csv"):
        if not forced and Path(save_path).exists():
            return pd.read_csv(save_path, index_col=False)
        else:
            df[self.feature_col] = df[self.feature_col].str.replace('*', '{}', regex=False)
            df[self.feature_col] = df[self.feature_col].apply(ast.literal_eval)
            df_aggregated = df.pivot(index='item_id', columns='annotator', values=self.feature_col)
            df_aggregated.columns = [f'annotation_{col}' for col in df_aggregated.columns]
            for col in df_aggregated.columns:
                df_aggregated[col] = df_aggregated[col].apply(self.convert_label)
            df_aggregated["text"] = df.text
            df_aggregated = df_aggregated.dropna()
            df_aggregated["aggregated_label"] = self.get_majority_vote(df_aggregated, no_winner_label="Mixed")
            df_aggregated.to_csv(save_path, index_label=False)
            return df_aggregated

    @staticmethod
    def convert_label(row):
        row = list(row)
        if len(row) == 0:
            return None
        elif len(row) == 1 and row[0] == "Increases":
            return row[0]
        else:
            return "Mixed"
        #elif len(row) == 2:
        #    return "Mixed"
        #else:
        #    raise ValueError


class CoreStoryClassification(Classification):
    def __init__(self, csv_path, split_ratio: list[float], forced=False, seed=11):
        super().__init__(csv_path, split_ratio, seed)
        self.task_name = "core_story_classification"
        self.df_aggregated = self.aggregate(self.df, forced=forced, save_path=self.save_path)
        self.df_train, self.df_test = self.split(self.df_aggregated, split_ratio, forced=forced, save_path="./data/annotated/")
        self.label_distribution = dict(Counter(list(chain(*self.df_aggregated.aggregated_label))))
        print(f"label distribution: {self.label_distribution}")
        self.num_labels = len(self.label2id_map)

    @property
    def feature_col(self):
        return "feature_six"

    @property
    def label2id_map(self):
        return {label: i for i, label in enumerate(self.get_unique_labels(self.df_aggregated))}

    def get_unique_labels(self, df):
        return list(set(chain(*[value for value in df["aggregated_label"].values])))

    def aggregate(self, df, forced=False, save_path="./data/annotated/task_2_direction_classification.csv"):
        if not forced and Path(save_path).exists():
            return pd.read_csv(save_path, index_col=False)
        else:
            df[self.feature_col] = df[self.feature_col].str.replace('*', '{}', regex=False)
            df[self.feature_col] = df[self.feature_col].apply(ast.literal_eval)
            df_aggregated = df.pivot(index='item_id', columns='annotator', values=self.feature_col)
            df_aggregated.columns = [f'annotation_{col}' for col in df_aggregated.columns]
            df_aggregated["text"] = df.text
            df_aggregated = df_aggregated.dropna()
            df_aggregated["aggregated_label"] = self.get_majority_vote(df_aggregated)
            df_aggregated = df_aggregated.dropna()
            df_aggregated.to_csv(save_path, index_label=False)
            return df_aggregated

    def get_majority_vote(self, df, no_winner_label=None):
        col_names = [col for col in df.columns if col.startswith("annotation")]

        majority_labels = []
        has_winner = 0

        for i, row in enumerate(df[col_names].values):
            row = list(chain(*row))
            counter = Counter(row)
            most_common = [count for count in counter.most_common() if count[1] > 1]
            if not most_common:
                majority_labels.append(None)
            else:
                # Get highest frequency count
                max_count = most_common[0][1]
                top_labels = [label for label, count in most_common if count == max_count]
                # Handle tie-breaks
                if len(top_labels) == 1:
                    majority_labels.append([top_labels[0]])  # Clear winner
                    has_winner += 1
                else:
                    majority_labels.append(top_labels)  # no winner
        print(f"has winner ratio: {has_winner / len(df)}\n")
        return majority_labels

    def train(self):
        model_names = {#"distilbert/distilbert-base-uncased": 64,
                       #"ProsusAI/finbert": 64,
                       #"FacebookAI/roberta-base": 64,
                       #"google-bert/bert-base-uncased": 64,
                       #"worldbank/econberta-fs": 8,
                       "worldbank/econberta": 4,
                       "MAPAi/InflaBERT": 8, }
        train = pd.read_csv(f"./data/annotated/task_2_{self.task_name}_train.csv")
        test = pd.read_csv(f"./data/annotated/task_2_{self.task_name}_test.csv")
        train["aggregated_label"] = train["aggregated_label"].apply(ast.literal_eval)
        test["aggregated_label"] = test["aggregated_label"].apply(ast.literal_eval)
        train['label'] = [[self.label2id_map[label] for label in labels] for labels in train['aggregated_label']]
        test['label'] = [[self.label2id_map[label] for label in labels] for labels in test['aggregated_label']]

        train = Dataset.from_pandas(train)
        test = Dataset.from_pandas(test)
        label2id_map = {str(key): value for key, value in self.label2id_map.items()}
        id2label_map = {value: key for key, value in label2id_map.items()}

        def preprocess_function(examples):
            enc = tokenizer(examples['text'], truncation=True, padding='max_length')
            enc['labels'] = to_multi_hot(examples['label'], len(label2id_map))
            return enc

        def compute_metrics(eval_pred):
            f1 = evaluate.load("f1")
            logits, labels = eval_pred
            probs = 1 / (1 + np.exp(-logits))  # sigmoid
            preds = (probs > 0.5).astype(int)
            return f1.compute(predictions=preds, references=labels, average="weighted")

        def to_multi_hot(label_indices, num_classes):
            multi_hot_batch = []
            for indices in label_indices:
                vec = [float(0)] * num_classes
                for i in indices:
                    vec[i] = float(1)
                multi_hot_batch.append(vec)
            return multi_hot_batch

        for model_name, batch_size in model_names.items():
            name = f"{model_name.split('/')[-1]}-{self.task_name}"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=self.num_labels, id2label=id2label_map, label2id=label2id_map,
                problem_type="multi_label_classification")
            model.resize_token_embeddings(len(tokenizer))
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            tokenized_train = train.map(preprocess_function, batched=True)
            tokenized_test = test.map(preprocess_function, batched=True)

            # Training Arguments
            training_args = TrainingArguments(
                output_dir=f"./results/{name}",
                eval_strategy="epoch",
                save_strategy="best",
                learning_rate=2e-5,
                load_best_model_at_end=True,
                metric_for_best_model="f1_macro",                per_device_train_batch_size=batch_size,
                num_train_epochs=20,
                weight_decay=0.01,
                logging_dir=f"./logs/{name}"
            )

            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_test,
                processing_class=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )

            # Train Model
            trainer.train()
            model.save_pretrained(f"./logs/{name}/model", from_pt=True)

            # Test model
            predictions = trainer.predict(tokenized_test)

            # Extract predicted labels
            predicted_logits = predictions.predictions
            predicted_labels = np.argmax(predicted_logits, axis=-1)

            # Convert label indices back to label names
            decoded_predictions = [id2label_map[label] for label in predicted_labels]
            df_pred = pd.read_csv(f"./data/annotated/task_2_{self.task_name}_test.csv")
            df_pred["prediction"] = decoded_predictions

            df_pred.to_csv(f"./logs/{name}/prediction_seed_{self.seed}.csv", index=False)
            target_names = list(label2id_map.keys())
            report = classification_report(df_pred["aggregated_label"], df_pred["prediction"], target_names=target_names)
            print(model_name)
            print(report)
            with open(f"./logs/{name}/test_metric_seed_{self.seed}.txt", "w") as file:
                file.write(report)

            del model
            del trainer

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    csv_path = './data/annotated/task_2_annotation.csv'
    split_ratio = [0.7, 0.3]
    seed = 11
    forced = True
    classifier = DirectionClassification(csv_path, split_ratio, forced, seed)
    classifier.train()