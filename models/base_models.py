import torch
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve
from torch import nn
import torch.nn.functional as F
from transformers import Trainer


class CustomTrainerWeightedCELoss(Trainer):
    def __init__(self, label_weights: list[float], **kwargs):
        super().__init__(**kwargs)
        self.label_weights = label_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.label_weights, device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    

class MultilabelTrainer(Trainer):

    def __init__(self, label_weights, **kwargs):
        super().__init__(**kwargs)
        self.label_weights = label_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        self.label_weights = self.label_weights.to(logits.device)

        loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.float32), pos_weight=self.label_weights)
        return (loss, outputs) if return_outputs else loss
    

def compute_metrics_setfit(y_pred, y_test):
    #best_p = best_threshold_from_roc(labels, probs)
    #print(f"Best threshold: {best_p}")
    f1_micro = f1_score(y_test, y_pred, average = 'micro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average = 'macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average = 'weighted', zero_division=0)
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

def compute_metrics(p):
    logits, labels = p
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.2).astype(int)
    f1_micro = f1_score(labels, preds, average = 'micro', zero_division=0)
    f1_macro = f1_score(labels, preds, average = 'macro', zero_division=0)
    f1_weighted = f1_score(labels, preds, average = 'weighted', zero_division=0)
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

def best_threshold_from_roc(labels, probs):
    precision, recall, thresh = precision_recall_curve(labels, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return thresh[np.argmax(f1)]

# preprocess dataset with tokenizer
def tokenize_examples(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['text'])
    tokenized_inputs['labels'] = examples['labels']
    return tokenized_inputs


# define custom batch preprocessor
def collate_fn(batch, tokenizer):
    dict_keys = ['input_ids', 'attention_mask', 'labels']
    d = {k: [dic[k] for dic in batch] for k in dict_keys}
    d['input_ids'] = torch.nn.utils.rnn.pad_sequence(
        d['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id
    )
    d['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
        d['attention_mask'], batch_first=True, padding_value=0
    )
    d['labels'] = torch.stack(d['labels'])
    return d