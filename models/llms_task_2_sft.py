import os
import sys
import re
sys.path.append('./')
import torch
import functools
import json
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import numpy as np
from models.prepare_data import PrepareData, PrepareDataTriples
from transformers import EarlyStoppingCallback
from sklearn.metrics import (
    f1_score, 
    hamming_loss, 
    accuracy_score, 
    precision_score, 
    recall_score,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"



class GenerativeMultilabelClassifier:
    def __init__(self, task_name, label_col, seed=11, prompt_file="./models/prompts/prompt_events.txt"):
        self.setup_directory()
        self.task_name = task_name
        self.seed = seed
        self.label_col = label_col
        
        # Load custom prompt template
        self.prompt_template = self.load_prompt_template(prompt_file)
        
        # Load your preprocessed data
        feature_col = "text"
        input_csv = "data/annotated/task_2_annotation.csv"
        output_csv = f"data/preprocessed/task_2_{self.task_name}.csv"
        self.data = PrepareData(input_csv, output_csv, feature_col, label_col, task_name)
        self.data.preprocess()
        
        self.label2id_map = self.data.label2id_map
        self.id2label_map = {v: k for k, v in self.label2id_map.items()}
        
        # Get dataset with stratified split
        self.dataset, self.label_weights, self.labels = self.data.stratify_split(test_size=0.2)
        
        print(f"Label mapping: {self.label2id_map}")
        print(f"Train size: {len(self.dataset['train'])}, Val size: {len(self.dataset['val'])}")

    def load_prompt_template(self, prompt_file):
        """
        Load prompt template from file.
        """
        with open(prompt_file, 'r') as f:
            return f.read().strip()
    

    def setup_directory(self):
        for folder in ["./logs/", "./export/", "./checkpoints/llm_generative/"]:
            path = Path(folder)
            if not path.exists():
                path.mkdir(parents=True)

    def create_instruction_prompt(self, text, labels=None, is_training=True):
        """
        Create instruction-based prompt for event classification using JSON format.
        """
        # Start with the base instruction from prompt template
        prompt = self.prompt_template + "\n\n"
        
        # Add the text to analyze
        prompt += f"Text: {text}\n\n"
        
        if is_training and labels is not None:
            # Convert binary labels to event categories
            active_labels = labels #[self.id2label_map[i] for i, val in enumerate(labels) if val == 1]
            
            # Create JSON output
            if len(active_labels) == 0:
                json_output = '{"events": []}'
            else:
                events = [{"category": label} for label in active_labels]
                json_output = json.dumps({"events": events}, ensure_ascii=False)
            
            prompt += f"Output: {json_output}"
        else:
            prompt += "Output:"
        
        return prompt

    def parse_generated_labels(self, generated_text):
        """
        Parse generated JSON text to extract event categories.
        Handles various output formats and malformed JSON.
        """
        try:
            # Extract text after "Output:" if present
            if "Output:" in generated_text:
                json_text = generated_text.split("Output:")[-1].strip()
            else:
                json_text = generated_text.strip()
            
            # Remove markdown fences if present
            json_text = json_text.replace("```json", "").replace("```", "").strip()
            
            # Try to parse JSON
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError:
                # Try to fix common JSON errors
                # Remove trailing commas
                json_text = json_text.replace(",]", "]").replace(",}", "}")
                # Try parsing again
                try:
                    data = json.loads(json_text)
                except:
                    # If still fails, try to extract events manually
                    return self.extract_events_from_malformed_json(json_text)
            
            # Extract categories from events
            if isinstance(data, dict) and "events" in data:
                events = data["events"]
                if isinstance(events, list):
                    categories = []
                    for event in events:
                        if isinstance(event, dict) and "category" in event:
                            categories.append(event["category"])
                    return categories
            
            return []
            
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            print(f"Generated text: {generated_text[:200]}")
            return []
    
    def extract_events_from_malformed_json(self, text):
        """
        Extract event categories from malformed JSON using pattern matching.
        """
        categories = []
        
        # Try to find patterns like "category": "Event Name"
        pattern = r'"category"\s*:\s*"([^"]+)"'
        matches = re.findall(pattern, text)
        
        if matches:
            categories = matches
        
        return categories

    def text_to_binary_vector(self, label_text):
        """
        Convert label text to binary vector.
        """
        binary_vec = [0] * len(self.label2id_map)
        
        parsed_labels = self.parse_generated_labels(label_text)
        
        for label in parsed_labels:
            if label in self.label2id_map:
                idx = self.label2id_map[label]
                binary_vec[idx] = 1
        
        return binary_vec

    def compute_metrics_for_generative(self, eval_pred, tokenizer):
        """
        Compute multilabel classification metrics from generated text.
        This function handles the evaluation for generative models.
        """
        predictions, labels = eval_pred
        
        # Decode predictions (token ids to text)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Handle padding and special tokens
        decoded_preds = []
        for pred in predictions:
            # Remove padding tokens
            pred = pred[pred != -100]
            decoded_text = tokenizer.decode(pred, skip_special_tokens=True)
            decoded_preds.append(decoded_text)
        
        # Decode labels (ground truth)
        decoded_labels = []
        for label in labels:
            # Remove padding tokens
            label = label[label != -100]
            decoded_text = tokenizer.decode(label, skip_special_tokens=True)
            decoded_labels.append(decoded_text)
        
        # Convert to binary vectors
        pred_binary = np.array([self.text_to_binary_vector(text) for text in decoded_preds])
        true_binary = np.array([self.text_to_binary_vector(text) for text in decoded_labels])
        
        # Calculate metrics
        metrics = self.calculate_multilabel_metrics(true_binary, pred_binary)
        
        return metrics

    def calculate_multilabel_metrics(self, y_true, y_pred):
        """
        Calculate comprehensive multilabel classification metrics.
        """
        print(f"shape y_true: {y_true.shape}, shape y_pred: {y_pred.shape}")
        print(f"Sample y_true: {y_true[0]}, Sample y_pred: {y_pred[0]}")
        # Exact match ratio (all labels must match)
        exact_match = accuracy_score(y_true, y_pred)
        
        # Hamming loss (fraction of wrong labels)
        hamming = hamming_loss(y_true, y_pred)
        
        # F1 scores
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_samples = f1_score(y_true, y_pred, average='samples', zero_division=0)
        
        # Precision and Recall
        precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Per-label F1 scores
        per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics = {
            'exact_match_ratio': exact_match,
            'hamming_loss': hamming,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_samples': f1_samples,
            'precision_micro': precision_micro,
            'precision_macro': precision_macro,
            'recall_micro': recall_micro,
            'recall_macro': recall_macro,
        }
        
        # Add per-label F1 scores
        for i, label_name in enumerate(self.id2label_map.values()):
            metrics[f'f1_{label_name}'] = per_label_f1[i] if i < len(per_label_f1) else 0.0
        
        return metrics
    
    def multi_hot_labels_to_strings(self, multi_hot_label):
        return [label for bit, label in zip(multi_hot_label, self.data.labels_in_string) if bit == 1]

    def evaluate_model_on_dataset(self, model, tokenizer, dataset, dataset_name="Validation"):
        """
        Perform comprehensive evaluation on a dataset using generation.
        """
        print(f"\n{'='*80}")
        print(f"Evaluating on {dataset_name} Set")
        print(f"{'='*80}\n")
        
        model.eval()
        all_predictions = []
        all_true_labels = []
        
        with torch.no_grad():
            for i, example in enumerate(dataset):
                # Get text and true labels
                text = example['text']
                true_labels = example['labels']
                
                # Create prompt
                prompt = self.create_instruction_prompt(text, labels=None, is_training=False)
                
                # Tokenize and generate
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,  # Increased for JSON output
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                # Decode prediction
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Convert to binary vectors
                pred_binary = self.text_to_binary_vector(generated_text)
                all_predictions.append(pred_binary)
                all_true_labels.append(true_labels)
                
                # Print some examples
                if i < 3:
                    pred_labels = self.parse_generated_labels(generated_text)
                    true_label_names = [self.id2label_map[j] for j, val in enumerate(true_labels) if val == 1]
                    print(f"Example {i+1}:")
                    print(f"  Text: {text[:100]}...")
                    print(f"  True labels: {true_label_names}")
                    print(f"  Predicted labels: {pred_labels}")
                    print()
        
        # Convert to numpy arrays
        y_true = np.array(all_true_labels)
        y_pred = np.array(all_predictions)
        
        # Calculate metrics
        metrics = self.calculate_multilabel_metrics(y_true, y_pred)
        
        # Print detailed results
        print(f"\n{dataset_name} Set Metrics:")
        print("-" * 80)
        print(f"Exact Match Ratio: {metrics['exact_match_ratio']:.4f}")
        print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
        print(f"F1 Micro: {metrics['f1_micro']:.4f}")
        print(f"F1 Macro: {metrics['f1_macro']:.4f}")
        print(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
        print(f"F1 Samples: {metrics['f1_samples']:.4f}")
        print(f"Precision Micro: {metrics['precision_micro']:.4f}")
        print(f"Precision Macro: {metrics['precision_macro']:.4f}")
        print(f"Recall Micro: {metrics['recall_micro']:.4f}")
        print(f"Recall Macro: {metrics['recall_macro']:.4f}")
        print("\nPer-Label F1 Scores:")
        print("-" * 80)
        for label_name in self.id2label_map.values():
            f1_key = f'f1_{label_name}'
            if f1_key in metrics:
                print(f"  {label_name}: {metrics[f1_key]:.4f}")
        print("-" * 80)
        
        return metrics
    
    def tokenize_function(self, examples, tokenizer, max_length=8000):
        """
        Tokenize examples with instruction format.
        For training, we include both input and output.
        Note: Using larger max_length (8000) to accommodate the detailed prompt.
        """
        prompts = []
        for i in range(len(examples['text'])):
            text = examples['text'][i]
            labels = self.multi_hot_labels_to_strings(examples['labels'][i] if 'labels' in examples else None)
            prompt = self.create_instruction_prompt(text, labels, is_training=True)
            prompts.append(prompt)
        
       # Tokenize - MUST use padding=True for batch processing
        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding='max_length',  # Pad to max_length to ensure consistent tensor shapes
            return_tensors=None
        )
        
        # For causal LM, labels are the same as input_ids
        # Replace padding tokens in labels with -100 (ignored by loss function)
        labels = []
        for input_ids in tokenized["input_ids"]:
            label_ids = [
                token_id if token_id != tokenizer.pad_token_id else -100 
                for token_id in input_ids
            ]
            labels.append(label_ids)
        
        tokenized["labels"] = labels
        return tokenized

    def preprocess_logits_for_metrics(self, logits, labels):
        # logits: [batch, seq_len, vocab_size]
        # return token IDs: [batch, seq_len]
        return logits.argmax(dim=-1)

    def train_llm_sft(self):
        """
        Train LLM using generative approach with proper LoRA configuration.
        """
        # Model configurations optimized for your 48GB GPU
        model_configs = {
            # Use smaller models for better results with limited data
            "meta-llama/Llama-3.2-1B": {
                "batch_size": 2,
                "gradient_accumulation_steps": 8,
                "max_length": 8000
            },
            "meta-llama/Llama-3.2-3B": {
                "batch_size": 2,
                "gradient_accumulation_steps": 8,
                "max_length": 8000  # Increased for longer prompt
            },
            "mistralai/Mistral-7B-v0.3": {
                "batch_size": 2,
                "gradient_accumulation_steps": 8,
                "max_length": 8000
            }
        }
        
        for model_name, config in model_configs.items():
            print(f"\n{'='*80}")
            print(f"Training model: {model_name}")
            print(f"Batch size: {config['batch_size']}, Gradient accumulation: {config['gradient_accumulation_steps']}")
            print(f"Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
            print(f"{'='*80}\n")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                token=os.getenv("HF_TOKEN"),
                trust_remote_code=True
            )
            
            # Set padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            # Tokenize dataset
            tokenized_ds = self.dataset.map(
                functools.partial(
                    self.tokenize_function, 
                    tokenizer=tokenizer,
                    max_length=config['max_length']
                ),
                batched=True,
                remove_columns=self.dataset['train'].column_names
            )
            
            # 4-bit quantization config for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            # Enhanced LoRA config for generative tasks
            lora_config = LoraConfig(
                r=32,  # Rank
                lora_alpha=32,  # Higher alpha for better learning
                target_modules=[
                    "q_proj", 
                    "k_proj", 
                    "v_proj", 
                    "o_proj",
                    "gate_proj",  # Important for generative tasks
                    "up_proj",
                    "down_proj"
                ],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"  # Critical: Use CAUSAL_LM, not SEQ_CLS
            )
            
            # Load model for causal language modeling
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                #device_map="auto",
                device_map={"": torch.cuda.current_device()},
                token=os.getenv("HF_TOKEN"),
                trust_remote_code=True
            )
            
            # Prepare model for k-bit training
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)
            
            # Print trainable parameters
            model.print_trainable_parameters()
            
            # Data collator for language modeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,  # We're doing causal LM, not masked LM
                pad_to_multiple_of=8
            )
            
            # Create compute_metrics function with tokenizer
            compute_metrics_fn = functools.partial(
                self.compute_metrics_for_generative,
                tokenizer=tokenizer
            )
            
            # Training arguments optimized for small dataset
            training_args = TrainingArguments(
                output_dir=f"./checkpoints/llm_generative/{model_name.replace('/', '_')}",
                learning_rate=2e-4,  # Higher LR for LoRA
                per_device_train_batch_size=config['batch_size'],
                #per_device_eval_batch_size=config['batch_size'],
                gradient_accumulation_steps=config['gradient_accumulation_steps'],
                num_train_epochs=100,  # More epochs for small dataset
                weight_decay=0.01,
                warmup_steps=10,
                logging_steps=3,
                eval_strategy="epoch",
                save_strategy="epoch",
                eval_accumulation_steps=1,
                eval_steps=3,
                save_steps=3,
                load_best_model_at_end=True,
                metric_for_best_model="f1_macro",  # Use F1 macro as best metric
                greater_is_better=True,
                fp16=False,
                bf16=True,  # Use bfloat16 for better stability
                gradient_checkpointing=True,
                save_total_limit=1,
                report_to="none",  # Disable wandb if not configured
                seed=self.seed,
                prediction_loss_only=False,  # Enable metric computation
                per_device_eval_batch_size=1,
                # Gradient optimization
                max_grad_norm=1.0,  # Gradient clipping
                optim="adamw_torch",  # Faster optimizer for multi-GPU
                # Memory optimization
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_ds['train'],
                eval_dataset=tokenized_ds['val'],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics_fn,
                preprocess_logits_for_metrics=self.preprocess_logits_for_metrics, 
                callbacks = [EarlyStoppingCallback(early_stopping_patience=10)], 
            )
            
            # Train
            print("\nStarting training...")
            trainer.train()
            
            # Perform final evaluation on validation set
            print(f"\n{'='*80}")
            print(f"--------{model_name}--task_name:{self.task_name}--------")
            print(f"{'='*80}")
            
            # Evaluation using trainer's evaluate (uses compute_metrics during training)
            print("\nEvaluation using Trainer.evaluate():")
            evaluation_metrics = trainer.evaluate()
            print("Evaluation metrics:")
            for key, value in evaluation_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            
            # Detailed evaluation with generation (more accurate for generative models)
            print("\n" + "="*80)
            print("Detailed Evaluation with Generation:")
            print("="*80)
            
            # Evaluate on validation set
            val_metrics = self.evaluate_model_on_dataset(
                model=trainer.model,
                tokenizer=tokenizer,
                dataset=self.dataset['val'],
                dataset_name="Validation"
            )
            
            # Evaluate on training set (to check for overfitting)
            train_metrics = self.evaluate_model_on_dataset(
                model=trainer.model,
                tokenizer=tokenizer,
                dataset=self.dataset['train'],
                dataset_name="Training"
            )
            
            print(f"\n{'='*80}")
            print("Training Summary:")
            print(f"{'='*80}")
            print(f"Train F1 Macro: {train_metrics['f1_macro']:.4f}")
            print(f"Val F1 Macro: {val_metrics['f1_macro']:.4f}")
            print(f"Overfitting Gap: {train_metrics['f1_macro'] - val_metrics['f1_macro']:.4f}")
            print(f"{'='*80}\n")
            
            # Save the fine-tuned model
            save_path = f"./checkpoints/llm_sft/{model_name.replace('/', '_')}/final"
            trainer.model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            # Save label mapping and metrics
            with open(f"{save_path}/label_mapping.json", "w") as f:
                json.dump({
                    "label2id": self.label2id_map,
                    "id2label": self.id2label_map
                }, f, indent=2)
            
            with open(f"{save_path}/evaluation_results.json", "w") as f:
                json.dump({
                    "validation_metrics": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                          for k, v in val_metrics.items()},
                    "training_metrics": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                        for k, v in train_metrics.items()},
                    "trainer_eval_metrics": {k: float(v) if isinstance(v, (float, np.floating, np.integer)) else v 
                                            for k, v in evaluation_metrics.items()}
                }, f, indent=2)
            
            print(f"\nModel and results saved to: {save_path}")
            print(f"{'='*80}\n")
            
            # Cleanup
            del model
            del trainer
            torch.cuda.empty_cache()
            
            print(f"\nCompleted training for {model_name}\n")

    def inference(self, model_path, test_texts, max_new_tokens=50):
        """
        Perform inference on test texts.
        """
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.eval()
        
        predictions = []
        
        for text in test_texts:
            # Create prompt without labels
            prompt = self.create_instruction_prompt(text, labels=None, is_training=False)
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,  # Low temperature for more deterministic output
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract predicted labels (text after "Labels:")
            if "Labels:" in generated_text:
                pred_text = generated_text.split("Labels:")[-1].strip()
                predictions.append(pred_text)
            else:
                predictions.append("")
        
        return predictions

    def parse_predictions_to_binary(self, pred_texts):
        """
        Convert predicted label strings to binary vectors.
        """
        binary_preds = []
        
        for pred_text in pred_texts:
            binary_vec = [0] * len(self.label2id_map)
            
            # Parse predicted labels
            if pred_text.lower() != "none":
                pred_labels = [label.strip() for label in pred_text.split(",")]
                for label in pred_labels:
                    if label in self.label2id_map:
                        idx = self.label2id_map[label]
                        binary_vec[idx] = 1
            
            binary_preds.append(binary_vec)
        
        return np.array(binary_preds)


class GenerativeTripleExtractor:
    def __init__(self, task_name, label_col, seed=11, prompt_file="./models/prompts/prompt_triples.txt"):
        self.setup_directory()
        self.task_name = task_name
        self.seed = seed
        self.label_col = label_col
        
        # Load custom prompt template
        self.prompt_template = self.load_prompt_template(prompt_file)
        
        # Load preprocessed data
        feature_col = "text"
        input_csv = "data/annotated/task_2_annotation.csv"
        output_csv = f"data/preprocessed/task_triples_{self.task_name}.csv"
        self.data = PrepareDataTriples(input_csv, output_csv, feature_col, label_col, task_name)
        self.data.preprocess()
        
        # Get all possible events and relations
        self.all_events = self.data.all_events
        self.all_relations = self.data.all_relations
        
        # Get dataset with split
        self.dataset = self.data.stratify_split(test_size=0.2)
        
        print(f"All events: {len(self.all_events)}")
        print(f"All relations: {self.all_relations}")
        print(f"Train size: {len(self.dataset['train'])}, Val size: {len(self.dataset['val'])}")

    def load_prompt_template(self, prompt_file):
        """Load prompt template from file."""
        with open(prompt_file, 'r') as f:
            return f.read().strip()
    
    def setup_directory(self):
        for folder in ["./logs/", "./export/", "./checkpoints/llm_triples/"]:
            path = Path(folder)
            if not path.exists():
                path.mkdir(parents=True)

    def create_instruction_prompt(self, text, triples=None, is_training=True):
        """
        Create instruction-based prompt for triple extraction using JSON format.
        """
        prompt = self.prompt_template + "\n\n"
        prompt += f"Text: {text}\n\n"
        
        if is_training and triples is not None:
            # Convert triples to JSON format
            if len(triples) == 0:
                json_output = '{"triples": []}'
            else:
                triple_dicts = [
                    {
                        "event_a": triple[0],
                        "relation": triple[1],
                        "event_b": triple[2]
                    }
                    for triple in triples
                ]
                json_output = json.dumps({"triples": triple_dicts}, ensure_ascii=False)
            
            prompt += f"Output: {json_output}"
        else:
            prompt += "Output:"
        
        return prompt

    def parse_generated_triples(self, generated_text):
        """
        Parse generated JSON text to extract triples.
        Handles various output formats and malformed JSON.
        Returns list of tuples: [(event_a, relation, event_b), ...]
        """
        try:
            # Extract text after "Output:" if present
            if "Output:" in generated_text:
                json_text = generated_text.split("Output:")[-1].strip()
            else:
                json_text = generated_text.strip()
            
            # Remove markdown fences if present
            json_text = json_text.replace("```json", "").replace("```", "").strip()
            
            # Try to parse JSON
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError:
                # Try to fix common JSON errors
                json_text = json_text.replace(",]", "]").replace(",}", "}")
                try:
                    data = json.loads(json_text)
                except:
                    return self.extract_triples_from_malformed_json(json_text)
            
            # Extract triples from JSON
            if isinstance(data, dict) and "triples" in data:
                triples_list = data["triples"]
                if isinstance(triples_list, list):
                    parsed_triples = []
                    for triple in triples_list:
                        if isinstance(triple, dict):
                            event_a = triple.get("event_a", "").strip()
                            relation = triple.get("relation", "").strip()
                            event_b = triple.get("event_b", "").strip()
                            
                            # Validate triple
                            if (event_a in self.all_events and 
                                relation in self.all_relations and 
                                event_b in self.all_events):
                                parsed_triples.append((event_a, relation, event_b))
                    
                    return parsed_triples
            
            return []
            
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            print(f"Generated text: {generated_text[:200]}")
            return []
    
    def extract_triples_from_malformed_json(self, text):
        """
        Extract triples from malformed JSON using pattern matching.
        """
        triples = []
        
        # Pattern: "event_a": "...", "relation": "...", "event_b": "..."
        pattern = r'"event_a"\s*:\s*"([^"]+)"[^}]*"relation"\s*:\s*"([^"]+)"[^}]*"event_b"\s*:\s*"([^"]+)"'
        matches = re.findall(pattern, text)
        
        for match in matches:
            event_a, relation, event_b = match
            event_a = event_a.strip()
            relation = relation.strip()
            event_b = event_b.strip()
            
            # Validate
            if (event_a in self.all_events and 
                relation in self.all_relations and 
                event_b in self.all_events):
                triples.append((event_a, relation, event_b))
        
        return triples

    def compute_triple_metrics(self, true_triples_list, pred_triples_list):
        """
        Compute comprehensive metrics for triple extraction.
        
        Metrics for each level (triple, event, relation):
        - Micro F1: Global TP/FP/FN across all samples
        - Macro F1: Average F1 per sample
        - Weighted F1: Weighted average by number of true instances per sample
        - Exact match: Percentage of samples with all triples correct
        """
        metrics = {}
        
        # Initialize counters
        total_samples = len(true_triples_list)
        
        # === MICRO METRICS (Global TP/FP/FN) ===
        # Triple-level
        triple_exact_match = 0
        triple_tp_micro = 0
        triple_fp_micro = 0
        triple_fn_micro = 0
        
        # Event-level
        event_tp_micro = 0
        event_fp_micro = 0
        event_fn_micro = 0
        
        # Relation-level
        relation_tp_micro = 0
        relation_fp_micro = 0
        relation_fn_micro = 0
        
        # === MACRO METRICS (Per-sample F1, then average) ===
        triple_f1_per_sample = []
        event_f1_per_sample = []
        relation_f1_per_sample = []
        
        # === WEIGHTED METRICS (Track weights for weighted average) ===
        sample_weights = []  # Number of true triples per sample
        
        for true_triples, pred_triples in zip(true_triples_list, pred_triples_list):
            true_set = set(true_triples)
            pred_set = set(pred_triples)
            
            # Weight for this sample
            sample_weight = len(true_triples) if len(true_triples) > 0 else 1
            sample_weights.append(sample_weight)
            
            # === TRIPLE-LEVEL ===
            # Exact match
            if true_set == pred_set:
                triple_exact_match += 1
            
            # Micro: accumulate global TP/FP/FN
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            
            triple_tp_micro += tp
            triple_fp_micro += fp
            triple_fn_micro += fn
            
            # Macro: compute F1 for this sample
            sample_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            sample_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            sample_f1 = 2 * sample_precision * sample_recall / (sample_precision + sample_recall) if (sample_precision + sample_recall) > 0 else 0
            triple_f1_per_sample.append(sample_f1)
            
            # === EVENT-LEVEL ===
            # Extract all unique events
            true_events = set()
            for t in true_triples:
                true_events.add(t[0])  # event_a
                true_events.add(t[2])  # event_b
            
            pred_events = set()
            for t in pred_triples:
                pred_events.add(t[0])
                pred_events.add(t[2])
            
            # Micro
            event_tp = len(true_events & pred_events)
            event_fp = len(pred_events - true_events)
            event_fn = len(true_events - pred_events)
            
            event_tp_micro += event_tp
            event_fp_micro += event_fp
            event_fn_micro += event_fn
            
            # Macro
            event_sample_precision = event_tp / (event_tp + event_fp) if (event_tp + event_fp) > 0 else 0
            event_sample_recall = event_tp / (event_tp + event_fn) if (event_tp + event_fn) > 0 else 0
            event_sample_f1 = 2 * event_sample_precision * event_sample_recall / (event_sample_precision + event_sample_recall) if (event_sample_precision + event_sample_recall) > 0 else 0
            event_f1_per_sample.append(event_sample_f1)
            
            # === RELATION-LEVEL ===
            # Check relations for matching event pairs
            true_pairs = {(t[0], t[2]): t[1] for t in true_triples}
            pred_pairs = {(t[0], t[2]): t[1] for t in pred_triples}
            
            matching_pairs = set(true_pairs.keys()) & set(pred_pairs.keys())
            
            relation_tp = sum(1 for pair in matching_pairs if true_pairs[pair] == pred_pairs[pair])
            relation_fp = len(set(pred_pairs.keys()) - set(true_pairs.keys())) + sum(1 for pair in matching_pairs if true_pairs[pair] != pred_pairs[pair])
            relation_fn = len(set(true_pairs.keys()) - set(pred_pairs.keys())) + sum(1 for pair in matching_pairs if true_pairs[pair] != pred_pairs[pair])
            
            # Micro
            relation_tp_micro += relation_tp
            relation_fp_micro += relation_fp
            relation_fn_micro += relation_fn
            
            # Macro
            relation_sample_precision = relation_tp / (relation_tp + relation_fp) if (relation_tp + relation_fp) > 0 else 0
            relation_sample_recall = relation_tp / (relation_tp + relation_fn) if (relation_tp + relation_fn) > 0 else 0
            relation_sample_f1 = 2 * relation_sample_precision * relation_sample_recall / (relation_sample_precision + relation_sample_recall) if (relation_sample_precision + relation_sample_recall) > 0 else 0
            relation_f1_per_sample.append(relation_sample_f1)
        
        # === COMPUTE FINAL METRICS ===
        
        # Exact match
        metrics['triple_exact_match'] = triple_exact_match / total_samples if total_samples > 0 else 0
        
        # TRIPLE METRICS
        # Micro
        triple_precision_micro = triple_tp_micro / (triple_tp_micro + triple_fp_micro) if (triple_tp_micro + triple_fp_micro) > 0 else 0
        triple_recall_micro = triple_tp_micro / (triple_tp_micro + triple_fn_micro) if (triple_tp_micro + triple_fn_micro) > 0 else 0
        triple_f1_micro = 2 * triple_precision_micro * triple_recall_micro / (triple_precision_micro + triple_recall_micro) if (triple_precision_micro + triple_recall_micro) > 0 else 0
        
        # Macro
        triple_f1_macro = np.mean(triple_f1_per_sample) if len(triple_f1_per_sample) > 0 else 0
        
        # Weighted
        if sum(sample_weights) > 0:
            triple_f1_weighted = np.average(triple_f1_per_sample, weights=sample_weights)
        else:
            triple_f1_weighted = 0
        
        metrics['triple_f1_micro'] = triple_f1_micro
        metrics['triple_f1_macro'] = triple_f1_macro
        metrics['triple_f1_weighted'] = triple_f1_weighted
        metrics['triple_precision_micro'] = triple_precision_micro
        metrics['triple_recall_micro'] = triple_recall_micro
        
        # EVENT METRICS
        # Micro
        event_precision_micro = event_tp_micro / (event_tp_micro + event_fp_micro) if (event_tp_micro + event_fp_micro) > 0 else 0
        event_recall_micro = event_tp_micro / (event_tp_micro + event_fn_micro) if (event_tp_micro + event_fn_micro) > 0 else 0
        event_f1_micro = 2 * event_precision_micro * event_recall_micro / (event_precision_micro + event_recall_micro) if (event_precision_micro + event_recall_micro) > 0 else 0
        
        # Macro
        event_f1_macro = np.mean(event_f1_per_sample) if len(event_f1_per_sample) > 0 else 0
        
        # Weighted
        if sum(sample_weights) > 0:
            event_f1_weighted = np.average(event_f1_per_sample, weights=sample_weights)
        else:
            event_f1_weighted = 0
        
        metrics['event_f1_micro'] = event_f1_micro
        metrics['event_f1_macro'] = event_f1_macro
        metrics['event_f1_weighted'] = event_f1_weighted
        metrics['event_precision_micro'] = event_precision_micro
        metrics['event_recall_micro'] = event_recall_micro
        
        # RELATION METRICS
        # Micro
        relation_precision_micro = relation_tp_micro / (relation_tp_micro + relation_fp_micro) if (relation_tp_micro + relation_fp_micro) > 0 else 0
        relation_recall_micro = relation_tp_micro / (relation_tp_micro + relation_fn_micro) if (relation_tp_micro + relation_fn_micro) > 0 else 0
        relation_f1_micro = 2 * relation_precision_micro * relation_recall_micro / (relation_precision_micro + relation_recall_micro) if (relation_precision_micro + relation_recall_micro) > 0 else 0
        
        # Macro
        relation_f1_macro = np.mean(relation_f1_per_sample) if len(relation_f1_per_sample) > 0 else 0
        
        # Weighted
        if sum(sample_weights) > 0:
            relation_f1_weighted = np.average(relation_f1_per_sample, weights=sample_weights)
        else:
            relation_f1_weighted = 0
        
        metrics['relation_f1_micro'] = relation_f1_micro
        metrics['relation_f1_macro'] = relation_f1_macro
        metrics['relation_f1_weighted'] = relation_f1_weighted
        metrics['relation_precision_micro'] = relation_precision_micro
        metrics['relation_recall_micro'] = relation_recall_micro
        
        return metrics

    def compute_metrics_for_generative(self, eval_pred, tokenizer):
        """
        Compute triple extraction metrics from generated text.
        """
        predictions, labels = eval_pred
        
        # Decode predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        decoded_preds = []
        for pred in predictions:
            pred = pred[pred != -100]
            decoded_text = tokenizer.decode(pred, skip_special_tokens=True)
            decoded_preds.append(decoded_text)
        
        # Decode labels
        decoded_labels = []
        for label in labels:
            label = label[label != -100]
            decoded_text = tokenizer.decode(label, skip_special_tokens=True)
            decoded_labels.append(decoded_text)
        
        # Parse to triples
        pred_triples_list = [self.parse_generated_triples(text) for text in decoded_preds]
        true_triples_list = [self.parse_generated_triples(text) for text in decoded_labels]
        
        # Calculate metrics
        metrics = self.compute_triple_metrics(true_triples_list, pred_triples_list)
        
        return metrics

    def evaluate_model_on_dataset(self, model, tokenizer, dataset, dataset_name="Validation"):
        """
        Perform comprehensive evaluation on a dataset using generation.
        """
        print(f"\n{'='*80}")
        print(f"Evaluating on {dataset_name} Set")
        print(f"{'='*80}\n")
        
        model.eval()
        all_predictions = []
        all_true_triples = []
        
        with torch.no_grad():
            for i, example in enumerate(dataset):
                text = example['text']
                true_triples = example['triples']
                
                # Create prompt
                prompt = self.create_instruction_prompt(text, triples=None, is_training=False)
                
                # Tokenize and generate
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                # Decode prediction
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Parse triples
                pred_triples = self.parse_generated_triples(generated_text)
                all_predictions.append(pred_triples)
                all_true_triples.append(true_triples)
                
                # Print examples
                if i < 3:
                    print(f"Example {i+1}:")
                    print(f"  Text: {text[:100]}...")
                    print(f"  True triples: {true_triples}")
                    print(f"  Predicted triples: {pred_triples}")
                    print()
        
        # Calculate metrics
        metrics = self.compute_triple_metrics(all_true_triples, all_predictions)
        
        # Print results
        print(f"\n{dataset_name} Set Metrics:")
        print("-" * 80)
        print(f"Triple Exact Match: {metrics['triple_exact_match']:.4f}")
        print()
        print("TRIPLE-LEVEL METRICS:")
        print(f"  F1 Micro:    {metrics['triple_f1_micro']:.4f}")
        print(f"  F1 Macro:    {metrics['triple_f1_macro']:.4f}")
        print(f"  F1 Weighted: {metrics['triple_f1_weighted']:.4f}")
        print(f"  Precision:   {metrics['triple_precision_micro']:.4f}")
        print(f"  Recall:      {metrics['triple_recall_micro']:.4f}")
        print()
        print("EVENT-LEVEL METRICS:")
        print(f"  F1 Micro:    {metrics['event_f1_micro']:.4f}")
        print(f"  F1 Macro:    {metrics['event_f1_macro']:.4f}")
        print(f"  F1 Weighted: {metrics['event_f1_weighted']:.4f}")
        print(f"  Precision:   {metrics['event_precision_micro']:.4f}")
        print(f"  Recall:      {metrics['event_recall_micro']:.4f}")
        print()
        print("RELATION-LEVEL METRICS:")
        print(f"  F1 Micro:    {metrics['relation_f1_micro']:.4f}")
        print(f"  F1 Macro:    {metrics['relation_f1_macro']:.4f}")
        print(f"  F1 Weighted: {metrics['relation_f1_weighted']:.4f}")
        print(f"  Precision:   {metrics['relation_precision_micro']:.4f}")
        print(f"  Recall:      {metrics['relation_recall_micro']:.4f}")
        print("-" * 80)
        
        return metrics
    
    def tokenize_function(self, examples, tokenizer, max_length=8000):
        """Tokenize examples with instruction format."""
        prompts = []
        for i in range(len(examples['text'])):
            text = examples['text'][i]
            triples = examples['triples'][i] if 'triples' in examples else None
            prompt = self.create_instruction_prompt(text, triples, is_training=True)
            prompts.append(prompt)
        
        tokenized = tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors=None
        )
        
        # Create labels
        labels = []
        for input_ids in tokenized["input_ids"]:
            label_ids = [
                token_id if token_id != tokenizer.pad_token_id else -100 
                for token_id in input_ids
            ]
            labels.append(label_ids)
        
        tokenized["labels"] = labels
        return tokenized

    def preprocess_logits_for_metrics(self, logits, labels):
        return logits.argmax(dim=-1)

    def train_llm_sft(self):
        """Train LLM for triple extraction using generative approach."""
        model_configs = {
            #"google/gemma-7b": {
            #    "batch_size": 1,
            #    "gradient_accumulation_steps": 16,
            #    "max_length": 8000
            #},
            #"Qwen/Qwen3-4B-Instruct-2507": {
            #    "batch_size": 2,
            #    "gradient_accumulation_steps": 8,
            #    "max_length": 8000
            #},
            #"Qwen/Qwen3-8B": {
            #    "batch_size": 1,
            #    "gradient_accumulation_steps": 16,
            #    "max_length": 8000
            #},
            "meta-llama/Llama-3.2-3B-Instruct": {
                "batch_size": 2,
                "gradient_accumulation_steps": 8,
                "max_length": 8000
            },
            "meta-llama/Llama-3.2-1B": {
                "batch_size": 2,
                "gradient_accumulation_steps": 8,
                "max_length": 8000
            },
            "meta-llama/Llama-3.2-3B": {
                "batch_size": 2,
                "gradient_accumulation_steps": 8,
                "max_length": 8000
            },
            #"mistralai/Mistral-7B-v0.3": {
            #    "batch_size": 4,
            #    "gradient_accumulation_steps": 4,
            #    "max_length": 8000
            #}
        }
        
        for model_name, config in model_configs.items():
            print(f"\n{'='*80}")
            print(f"Training model: {model_name}")
            print(f"Batch size: {config['batch_size']}, Gradient accumulation: {config['gradient_accumulation_steps']}")
            print(f"{'='*80}\n")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                token=os.getenv("HF_TOKEN"),
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            # Tokenize dataset
            tokenized_ds = self.dataset.map(
                functools.partial(
                    self.tokenize_function, 
                    tokenizer=tokenizer,
                    max_length=config['max_length']
                ),
                batched=True,
                remove_columns=self.dataset['train'].column_names
            )
            
            # Quantization config
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            # LoRA config
            lora_config = LoraConfig(
                r=32,
                use_dora=True,
                lora_alpha=32,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map={"": torch.cuda.current_device()},
                token=os.getenv("HF_TOKEN"),
                trust_remote_code=True
            )
            
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
            
            # Compute metrics function
            compute_metrics_fn = functools.partial(
                self.compute_metrics_for_generative,
                tokenizer=tokenizer
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"./checkpoints/llm_triples/{model_name.replace('/', '_')}",
                learning_rate=2e-4,
                per_device_train_batch_size=config['batch_size'],
                gradient_accumulation_steps=config['gradient_accumulation_steps'],
                num_train_epochs=100,
                weight_decay=0.01,
                warmup_steps=10,
                logging_steps=1,
                eval_strategy="epoch",
                save_strategy="epoch",
                eval_accumulation_steps=1,
                eval_steps=1,
                save_steps=1,
                load_best_model_at_end=True,
                metric_for_best_model="triple_f1_macro",
                greater_is_better=True,
                fp16=False,
                bf16=True,
                gradient_checkpointing=True,
                save_total_limit=1,
                report_to="none",
                seed=self.seed,
                prediction_loss_only=False,
                per_device_eval_batch_size=1,
                max_grad_norm=1.0,
                optim="adamw_torch",
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_ds['train'],
                eval_dataset=tokenized_ds['val'],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics_fn,
                preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
            )
            
            # Train
            print("\nStarting training...")
            trainer.train()
            
            # Final evaluation
            print(f"\n{'='*80}")
            print(f"Final Evaluation - {model_name} - Task: {self.task_name}")
            print(f"{'='*80}")
            
            evaluation_metrics = trainer.evaluate()
            print("\nTrainer evaluation metrics:")
            for key, value in evaluation_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
            
            # Save model
            save_path = f"./checkpoints/llm_triples/{model_name.replace('/', '_')}_dora/final"
            trainer.model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            # Save results
            with open(f"{save_path}/evaluation_results.json", "w") as f:
                json.dump({
                    "trainer_eval_metrics": {k: float(v) if isinstance(v, (float, np.floating, np.integer)) else v 
                                            for k, v in evaluation_metrics.items()}
                }, f, indent=2)
            
            print(f"\nModel and results saved to: {save_path}\n")
            
            # Cleanup
            del model
            del trainer
            torch.cuda.empty_cache()


# Example usage
if __name__ == "__main__":
    task_name = "full_story_classification_greater_than_2_overlap" #"adjacent_event_classification_greater_than_2_overlap"
    label_col = "feature_four"
    
    extractor = GenerativeTripleExtractor(
        task_name=task_name,
        label_col=label_col,
        seed=11,
        prompt_file="./models/prompts/prompt_triples.txt"
    )
    # Train with generative approach
    extractor.train_llm_sft()
