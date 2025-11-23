import os
import torch
import functools
import sys
sys.path.append('./')

from pathlib import Path
from transformers import AutoModelForSequenceClassification, TrainingArguments, AutoTokenizer, EarlyStoppingCallback
from models.base_models import MultilabelTrainer, compute_metrics, tokenize_examples, collate_fn, compute_metrics_setfit
from models.prepare_data import PrepareData
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from torch import nn
from setfit import SetFitModel, SetFitModelCardData, TrainingArguments as SetFitTrainingArguments, Trainer as SetFitTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"


class AdjacentEventClassification(object):
    def __init__(self, seed=11):
        self.setup_directory()
        self.task_name = "adjacent_event_classification"
        self.save_path = f"./data/preprocessed/task_2_{self.task_name}.csv"
        self.seed = seed
        feature_col = "text"
        label_cols = "feature_one"
        task_name = "adjacent_event_classification"
        input_csv = "data/annotated/task_2_annotation.csv"
        output_csv = "data/preprocessed/task_2_adjacent_event_classification.csv"
        data = PrepareData(input_csv, output_csv, feature_col, label_cols, task_name)
        data.preprocess()
        print(data.label2id_map)
        self.dataset, self.label_weights, self.labels, self.labels_in_string = data.stratify_split(test_size=0.2)

    def setup_directory(self):
        for folder in ["./logs/", "./export/"]:
            path = Path(folder)
            if not path.exists():  # Check existence
                path.mkdir(parents=True)

    def train_llm(self):
        model_names = { 
                        "meta-llama/Llama-3.2-1B": 16,
                        "mistralai/Mistral-7B-v0.3": 12,
                        "google/gemma-7b": 8,
                       #"distilbert/distilbert-base-uncased": 64          
                       #"ProsusAI/finbert": 64,
                       #"FacebookAI/roberta-base": 64,
                       #"google-bert/bert-base-uncased": 64,
                       #"worldbank/econberta-fs": 8,
                       #"worldbank/econberta": 4,
                       #"MAPAi/InflaBERT": 8, 
                       }
        for model_name, batch_size in model_names.items():
            print(f"Training model: {model_name} with batch size {batch_size}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, hf_token=os.getenv("HF_TOKEN"))
            tokenizer.pad_token = tokenizer.eos_token
            tokenized_ds = self.dataset.map(functools.partial(tokenize_examples, tokenizer=tokenizer), batched=True)
            tokenized_ds = tokenized_ds.with_format('torch')

            # qunatization config
            quantization_config = BitsAndBytesConfig(
                load_in_4bit = True, # enable 4-bit quantization
                bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
                bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
                bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
            )

            # lora config
            lora_config = LoraConfig(
                r = 16, # the dimension of the low-rank matrices
                lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
                target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                lora_dropout = 0.05, # dropout probability of the LoRA layers
                bias = 'none', # wether to train bias weights, set to 'none' for attention layers
                task_type = 'SEQ_CLS'
            )

            # load model
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                num_labels=self.labels.shape[1],
                torch_dtype=torch.float16,
            )

            print("Using", torch.cuda.device_count(), "GPUs!")

            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)
            model.config.pad_token_id = tokenizer.pad_token_id

            # define training args
            training_args = TrainingArguments(
                output_dir = 'multilabel_classification',
                learning_rate = 1e-5,
                per_device_train_batch_size = batch_size, # tested with 16gb gpu ram
                per_device_eval_batch_size = batch_size,
                num_train_epochs = 20,
                weight_decay = 0.01,
                eval_strategy = 'epoch',
                save_strategy = 'best',
                load_best_model_at_end = True
            )

            # train
            trainer = MultilabelTrainer(
                model = model,
                args = training_args,
                train_dataset = tokenized_ds['train'],
                eval_dataset = tokenized_ds['val'],
                tokenizer = tokenizer,
                data_collator = functools.partial(collate_fn, tokenizer=tokenizer),
                compute_metrics = compute_metrics,
                label_weights = torch.tensor(self.label_weights, device=model.device)
            )

            trainer.train()

            # save model
            peft_model_id = f"checkpoints/llm/{model_name}"
            trainer.model.save_pretrained(peft_model_id)
            tokenizer.save_pretrained(peft_model_id)

            del model
            torch.cuda.empty_cache()

    def train_setfit(self):
        dataset = self.dataset.rename_column("labels", "label")
        model_names = {"allenai/longformer-base-4096": 4}
        configs = {"one-vs-rest": [True, False], 
                   "multi-output": [True, False], 
                   "classifier-chain": [False]}
        for model_name, batch_size in model_names.items():
            print(f"Training model: {model_name} with batch size {batch_size}")
            for multi_target_strategy, use_differentiable_head in configs.items():
                for head in use_differentiable_head:
                    print("multi_target_strategy", multi_target_strategy)
                    print("use_differentiable_head", head)
                    model = SetFitModel.from_pretrained(
                                            model_name, 
                                            multi_target_strategy=multi_target_strategy,
                                            use_differentiable_head=head,
                                            labels=self.labels_in_string,
                                            head_params={"out_features": self.labels.shape[1]} 
                                                        if head else {"solver": "liblinear", "max_iter": 300},
                                            model_card_data=SetFitModelCardData(
                                                                    language="en",
                                                                    license="apache-2.0"))

                    print("Using", torch.cuda.device_count(), "GPUs!")

                    # define training args
                    training_args = SetFitTrainingArguments(
                        output_dir = f'outputs/setfit/{model_name}/{multi_target_strategy}_differentiable_{head}',
                        batch_size=(batch_size, batch_size),
                        num_epochs=(2, 40),
                        end_to_end=False,
                        body_learning_rate=(2e-6, 5e-6),
                        head_learning_rate=2e-5,
                        l2_weight=0.01,
                        eval_strategy = 'steps',
                        save_strategy = 'steps',
                        load_best_model_at_end = True
                    )

                    # train
                    trainer = SetFitTrainer(
                        model = model,
                        args = training_args,
                        train_dataset = dataset['train'],
                        eval_dataset = dataset['val'],
                        metric = compute_metrics_setfit,
                        callbacks = [EarlyStoppingCallback(early_stopping_patience=15)], 
                    )

                    trainer.train()
                    evaluation_metrics = trainer.evaluate()
                    print(f"---------------{model_name}-multilabel: {multi_target_strategy}--------differentiable: {head}---------------")
                    print("Evaluation metrics:\n", evaluation_metrics)
                    print(f"--------------------------------------------------------------------")
                    # save model
                    peft_model_id = f"checkpoints/setfit/{model_name}/{multi_target_strategy}_differentiable_{head}"
                    trainer.model.save_pretrained(peft_model_id)
                    #tokenizer.save_pretrained(peft_model_id)

                    del model
                    torch.cuda.empty_cache()

if __name__ == "__main__":
    classifier = AdjacentEventClassification(seed=11)
    #classifier.train_llm()
    classifier.train_setfit()