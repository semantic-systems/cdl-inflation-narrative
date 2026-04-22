import pandas as pd
from itertools import chain
from collections import Counter
from pathlib import Path
import ast
from sklearn.preprocessing import MultiLabelBinarizer
import csv
import random
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
from datasets import Dataset, DatasetDict
from scipy.stats import entropy


# load data given a feature and output designated csv for multi-label classification
class PrepareData:
    def __init__(self, input_csv_path, output_csv_path, feature_column, label_columns, task_name):
        self.input_csv_path = input_csv_path
        self.output_csv_path = output_csv_path
        self.task_name = task_name
        self.feature_column = feature_column
        self.label_columns = label_columns
        self.labels_in_string = None
        self.df = self.load_data()

    def load_data(self):
        df = pd.read_csv(self.input_csv_path)
        df = df[[self.feature_column, self.label_columns, "annotator"]]
        df[self.label_columns] = df[self.label_columns].str.replace('*', '{}', regex=False)
        df[self.label_columns] = df[self.label_columns].apply(ast.literal_eval)
        return df
    
    def remap_labels(self, event_list: list[str]):
        label_mapping = {
            "Essential Services Costs": ["Medical Costs", "Educational Costs", "Housing Costs"],
            "Demand Rebound": ["Demand Shift", "Pent-up Demand"],
            "Labor Market": ["Labor Shortage", "Wages"],
            "Supply Chains and Logistics": ["Supply Chain Issues", "Transportation Costs"]
        }
        reverse_event_category = {v: key for key, value in label_mapping.items() for v in value}
        remapped_events = list({reverse_event_category.get(event, event) for event in event_list})
        return remapped_events
    
    def preprocess(self):
        df = self.load_data()
        #df[self.label_columns] = df[self.label_columns].apply(self.remap_labels)
        df = self.get_overlap_labels(df)
        
        mlb = MultiLabelBinarizer()
        binarized = mlb.fit_transform(df["overlap_label"])
        labels_in_string = [label for label in mlb.classes_]
        self.labels_in_string = labels_in_string
        # Create DataFrame with binarized columns
        binarized_df = pd.DataFrame(binarized, columns=labels_in_string)

        # Combine with original grouped data
        df_multi_hot = pd.concat([df.reset_index(drop=True), binarized_df], axis=1)
        self.df_preprocessed = df_multi_hot
        df_multi_hot = df_multi_hot.drop(columns=["overlap_label"])
        df_multi_hot.to_csv(self.output_csv_path, index=False)
        print(f"Preprocessed data saved to {self.output_csv_path}")
        

    def get_unique_labels(self, df):
        return list(set(chain(*[value for value in df["overlap_label"].values])))
    
    @property
    def label2id_map(self):
        return {str(label): i for i, label in enumerate(self.get_unique_labels(self.df_preprocessed))}
    
    def get_pivot_df(self, df):
        pivoted = df.pivot(index='text', columns='annotator', values=self.label_columns).reset_index()
        pivoted.columns.name = None  # Remove the 'annotator' name from columns
        annotation_cols = [col for col in pivoted.columns if col != 'text']
        col_mapping = {col: f'annotation_{col}' for col in annotation_cols}
        pivoted.rename(columns=col_mapping, inplace=True)
        return pivoted

    def get_overlap_labels(self, df):
        df = self.get_pivot_df(df)
        col_names = [col for col in df.columns if col.startswith("annotation")]

        majority_labels = []
        has_winner = 0
        no_winner = 0
        multiple_winner = 0
        winner_type = []

        for i, row in enumerate(df[col_names].values):
            row = list(chain(*list(row)))
            counter = Counter(row)
            most_common = [count for count in counter.most_common() if count[1] > 1]
            if not most_common:
                majority_labels.append(None)
                no_winner += 1
                winner_type.append("no_winner")
            else:
                # Get highest frequency count
                max_count = most_common[0][1]
                top_labels = [label for label, count in most_common if count >= 2]# == max_count]
                #print(f"most_common {most_common} - Top labels: {top_labels} with count {max_count}")
                # Handle tie-breaks
                if len(top_labels) == 1:
                    majority_labels.append([top_labels[0]])  # Clear winner
                    has_winner += 1
                else:
                    majority_labels.append(top_labels)  # tie winners
                    multiple_winner += 1
                    winner_type.append("multiple_winner")
        df["winner_type"] = winner_type
        df["overlap_label"] = majority_labels
        df["overlap_label"] = df["overlap_label"].apply(lambda x: x if isinstance(x, list) else [])
        print(f"Has winner ratio: {has_winner / len(df)} ({has_winner})")
        print(f"Multiple winner ratio: {multiple_winner / len(df)} ({multiple_winner})")
        print(f"No winner ratio: {no_winner / len(df)} ({no_winner})")
        df.to_csv(f"data/preprocessed/task_2_{self.task_name}_overlap_labels_w_winner_type.csv", index=False, sep=',')
        df = df.drop(columns=col_names)
        return df
    
    def stratify_split(self, test_size=0.2):
        
        # load data
        with open(self.output_csv_path, newline='') as csvfile:
            data = list(csv.reader(csvfile, delimiter=','))
            data.pop(0)
        random.shuffle(data)

        # reshape
        text, labels = list(zip(*[(f'{row[0].strip()}', row[1:]) for row in data]))
        labels = np.array(labels, dtype=int)

        # create label weights
        label_weights = 1 - labels.sum(axis=0) / labels.sum()
        if not Path(f"data/preprocessed/dataset_{self.task_name}.hf").exists():
            # stratified train test split for multilabel ds
            row_ids = np.arange(len(labels))
            train_idx, y_train, val_idx, y_val = iterative_train_test_split(row_ids[:,np.newaxis], labels, test_size = test_size)
            x_train = [text[i] for i in train_idx.flatten()]
            x_val = [text[i] for i in val_idx.flatten()]

            # create hf dataset
            dataset = DatasetDict({
                'train': Dataset.from_dict({'text': x_train, 'labels': y_train}),
                'val': Dataset.from_dict({'text': x_val, 'labels': y_val})
            })
            dataset.save_to_disk(f"data/preprocessed/dataset_{self.task_name}.hf")
        else:
            print(f"Loading existing dataset from data/preprocessed/dataset_{self.task_name}.hf")
            dataset = DatasetDict.load_from_disk(f"data/preprocessed/dataset_{self.task_name}.hf")
        print(f"Train size: {len(dataset['train'])}, Val size: {len(dataset['val'])}")
        print("Label weights:", label_weights)
        return dataset, label_weights, labels


class PrepareDataTriples:
    """
    Prepare data for causal triple extraction task.
    Triples are represented as strings in format: (Event_A, Relation, Event_B)
    """
    def __init__(self, input_csv_path, output_csv_path, feature_column, label_column, task_name):
        self.input_csv_path = input_csv_path
        self.output_csv_path = output_csv_path
        self.task_name = task_name
        self.feature_column = feature_column
        self.label_column = label_column
        self.df = None
        self.df_preprocessed = None
        self.all_events = self._get_all_events()
        self.all_relations = ["Increases", "Decreases"]
        
    def _get_all_events(self):
        """Define all possible events for triple extraction."""
        return [
            # DEMAND EVENTS
            "Government Spending",
            "Monetary Policy",
            "Pent-up Demand",
            "Demand Shift",
            "Demand (Residuals)",
            # SUPPLY EVENTS
            "Supply Chain Issues",
            "Transportation Costs",
            "Labor Shortage",
            "Wages",
            "Energy Prices",
            "Food Prices",
            "Housing Costs",
            "Supply (Residual)",
            # MISCELLANEOUS EVENTS
            "Pandemic",
            "Politics",
            "War",
            "Inflation Expectations",
            "Base Effect",
            "Government Debt",
            "Tax Increases",
            "Trade Balance",
            "Exchange Rates",
            "Medical Costs",
            "Education Costs",
            "Climate Crisis",
            "Price-Gouging",
            # SPECIAL EVENTS
            "Inflation"
        ]

    def load_data(self):
        """Load and parse CSV data with triple annotations."""
        df = pd.read_csv(self.input_csv_path)
        df = df[[self.feature_column, self.label_column, "annotator"]]
        
        # Parse triples from string representation of sets
        # Expected format: "{('Event_A', 'Relation', 'Event_B'), ('Event_C', 'Relation', 'Event_D')}"
        def parse_triple_string(triple_str):
            if pd.isna(triple_str) or triple_str == "" or triple_str == "set()" or triple_str == "{}" or triple_str == "*" :
                return []
            
            try:
                # Use ast.literal_eval to safely parse the string
                parsed = ast.literal_eval(triple_str)
                
                # Handle both set and list formats
                if isinstance(parsed, set):
                    return list(parsed)
                elif isinstance(parsed, list):
                    return parsed
                else:
                    return []
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing triple string: {triple_str[:100]}")
                print(f"Error: {e}")
                return []
        
        df[self.label_column] = df[self.label_column].apply(parse_triple_string)
        
        return df
    
    def normalize_triple(self, triple):
        """
        Normalize a triple to standard format.
        Input can be dict or tuple: {"event_a": ..., "relation": ..., "event_b": ...} or (event_a, relation, event_b)
        Output: (event_a, relation, event_b) tuple
        """
        if isinstance(triple, dict):
            return (
                triple.get("event_a", "").strip(),
                triple.get("relation", "").strip(),
                triple.get("event_b", "").strip()
            )
        elif isinstance(triple, (list, tuple)) and len(triple) == 3:
            print(triple)
            return (triple[0].strip(), triple[1].strip(), triple[2].strip())
        else:
            return None
    
    def validate_triple(self, triple):
        """
        Validate that a triple has valid events and relation.
        Returns True if valid, False otherwise.
        """
        if not triple or len(triple) != 3:
            return False
        
        event_a, relation, event_b = triple
        
        # Check if events are in the valid event list
        if event_a not in self.all_events or event_b not in self.all_events:
            return False
        
        # Check if relation is valid
        if relation not in self.all_relations:
            return False
        
        return True
    
    def preprocess(self):
        """Preprocess data for triple extraction."""
        self.df = self.load_data()
        
        # Get overlap triples across annotators
        df = self.get_overlap_triples(self.df)
        
        # Store preprocessed data
        self.df_preprocessed = df
        
        # Save to CSV with string representation of triples
        df_to_save = df.copy()
        df_to_save["all_agreed_triples"] = df_to_save["all_agreed_triples"].apply(str)
        df_to_save.to_csv(self.output_csv_path, index=False)
        
        print(f"Preprocessed data saved to {self.output_csv_path}")
        print(f"Total samples: {len(df)}")
        print(f"Average length of all agreed triples per sample: {df['all_agreed_triples'].apply(len).mean():.2f}")
    
    def get_pivot_df(self, df):
        """Pivot DataFrame to have one row per text with multiple annotator columns."""
        pivoted = df.pivot(index='text', columns='annotator', values=self.label_column).reset_index()
        pivoted.columns.name = None
        
        annotation_cols = [col for col in pivoted.columns if col != 'text']
        col_mapping = {col: f'annotation_{col}' for col in annotation_cols}
        pivoted.rename(columns=col_mapping, inplace=True)
        
        return pivoted
    
    def get_overlap_triples(self, df):
        """
        Get overlapping triples across annotators.
        A triple is included if at least 2 annotators agree on it.
        """
        df = self.get_pivot_df(df)
        col_names = [col for col in df.columns if col.startswith("annotation")]
        
        overlap_triples_list = []
        has_overlap = 0
        no_overlap = 0
        multiple_overlap = 0
        winner_type = []
        n_overlap_triples = []
        triple_entropy = []
        n_all_agreed_triples = []

        for i, row in enumerate(df[col_names].values):
            # Flatten all triples from all annotators
            all_triples = []
            for annotator_triples in row:
                if isinstance(annotator_triples, list):
                    for triple in annotator_triples:
                        normalized = self.normalize_triple(triple)
                        if normalized and self.validate_triple(normalized):
                            all_triples.append(normalized)

            # Count occurrences of each triple
            triple_counter = Counter(all_triples)
            # Get triples that appear at least twice (agreement from 2+ annotators)
            all_agreed_triples = [triple for triple, count in triple_counter.items() if count == 4]
            n_all_agreed_triples.append(len(all_agreed_triples))

            overlap_counts = [count for triple, count in triple_counter.items()] or [0]
            n_overlap_triples.append(max(overlap_counts))
            triple_entropy.append(entropy(overlap_counts, base=2) if len(overlap_counts) > 1 else 0)

            if not all_agreed_triples:
                overlap_triples_list.append([])
                no_overlap += 1
                winner_type.append("no_winner")
            elif len(all_agreed_triples) == 1:
                overlap_triples_list.append(all_agreed_triples)
                has_overlap += 1
                winner_type.append("single_winner")
            else:
                overlap_triples_list.append(all_agreed_triples)
                multiple_overlap += 1
                winner_type.append("multiple_winner")
        
        df["all_agreed_triples"] = overlap_triples_list
        df["n_all_agreed_triples"] = n_all_agreed_triples
        df["winner_type"] = winner_type
        df["n_overlap_triples"] = n_overlap_triples
        df["triple_entropy"] = triple_entropy
        print(f"Single overlap ratio: {has_overlap / len(df):.3f} ({has_overlap})")
        print(f"Multiple overlap ratio: {multiple_overlap / len(df):.3f} ({multiple_overlap})")
        print(f"No overlap ratio: {no_overlap / len(df):.3f} ({no_overlap})")
        print(f"average no. overlap triples: {np.mean(n_overlap_triples):.2f}")
        
        # Save detailed version
        df.to_csv(
            f"data/preprocessed/task_triples_{self.task_name}_overlap_w_winner_type_w_n_winners.csv",
            index=False
        )
        
        # Drop annotation columns for final dataset
        df = df.drop(columns=col_names)
        return df
    
    def triple_to_string(self, triple):
        """Convert triple tuple to string representation."""
        if not triple:
            return ""
        return f"({triple[0]}, {triple[1]}, {triple[2]})"
    
    def string_to_triple(self, triple_str):
        """Convert string representation back to triple tuple."""
        if not triple_str or triple_str == "":
            return None
        
        # Remove parentheses and split by comma
        triple_str = triple_str.strip("()")
        parts = [part.strip() for part in triple_str.split(",")]
        
        if len(parts) == 3:
            return tuple(parts)
        return None
    
    def stratify_split(self, test_size=0.2, random_seed=42):
        """
        Create train/validation split.
        For triple extraction, we use random split since stratification is complex.
        """
        dataset_path = f"data/preprocessed/dataset_{self.task_name}.hf"
        
        if not Path(dataset_path).exists():
            # Load preprocessed data
            df = pd.read_csv(self.output_csv_path)
            
            # Parse triples back from string
            df["overlap_triples"] = df["overlap_triples"].apply(ast.literal_eval)
            
            # Prepare data
            texts = df[self.feature_column].tolist()
            triples = df["overlap_triples"].tolist()
            
            # Random shuffle with seed
            random.seed(random_seed)
            indices = list(range(len(texts)))
            random.shuffle(indices)
            
            # Split indices
            split_idx = int(len(indices) * (1 - test_size))
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            
            # Create train/val datasets
            x_train = [texts[i] for i in train_indices]
            y_train = [triples[i] for i in train_indices]
            x_val = [texts[i] for i in val_indices]
            y_val = [triples[i] for i in val_indices]
            
            # Create HuggingFace dataset
            dataset = DatasetDict({
                'train': Dataset.from_dict({
                    'text': x_train,
                    'triples': y_train
                }),
                'val': Dataset.from_dict({
                    'text': x_val,
                    'triples': y_val
                })
            })
            
            dataset.save_to_disk(dataset_path)
            print(f"Dataset saved to {dataset_path}")
        else:
            print(f"Loading existing dataset from {dataset_path}")
            dataset = DatasetDict.load_from_disk(dataset_path)
        
        print(f"Train size: {len(dataset['train'])}, Val size: {len(dataset['val'])}")
        
        # Calculate statistics
        train_triple_counts = [len(t) for t in dataset['train']['triples']]
        val_triple_counts = [len(t) for t in dataset['val']['triples']]
        
        print(f"Train - Avg triples per sample: {np.mean(train_triple_counts):.2f}")
        print(f"Val - Avg triples per sample: {np.mean(val_triple_counts):.2f}")
        
        return dataset


if __name__ == "__main__":
    feature_col = "text"
    label_col = "feature_six"  # Column containing triple annotations
    task_name = "causal_triple_extraction"
    input_csv = "data/annotated/task_2_annotation.csv"
    output_csv = f"data/preprocessed/task_triples_{task_name}_{label_col}.csv"
    
    preparer = PrepareDataTriples(input_csv, output_csv, feature_col, label_col, task_name)
    preparer.preprocess()
    #dataset = preparer.stratify_split(test_size=0.2)
    
    #print("\nSample from training set:")
    #print(f"Text: {dataset['train'][0]['text'][:100]}...")
    #print(f"Triples: {dataset['train'][0]['triples']}")


#if __name__ == "__main__":
#    feature_col = "text"
#    label_cols = "feature_one"
#    task_name = "adjacent_event_classification"
#    input_csv = "data/annotated/task_2_annotation.csv"
#    output_csv = f"data/preprocessed/task_2_{task_name}.csv"
#    preparer = PrepareData(input_csv, output_csv, feature_col, label_cols, task_name)
#    preparer.preprocess()
#    print(preparer.label2id_map)
    #ds, label_weights, labels_in_binary, labels_in_string = preparer.stratify_split(test_size=0.2)
