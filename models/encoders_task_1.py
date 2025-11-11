import os
import pandas as pd
import ast
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel, LongformerTokenizer, LongformerModel
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import L1Loss
import torch
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.nn import functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


class CDLDataModule(pl.LightningDataModule):
    def __init__(self, 
                 model_name="allenai/longformer-base-4096", 
                 data_dir: str = "/storage/huang/cdl-inflation-narrative/data/annotated", 
                 batch_size: int = 64,
                 force=False):
        super().__init__()
        self.force = force
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_labels = 2
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    @staticmethod
    def get_soft_label(row):
        label_aggregate_map = {"non-inflation-related": 0, "inflation-related": 0.5, "inflation-cause-dominant": 1}
        annotator_columns = [col for col in row.index if col.startswith('annotator_')]
        likelihoods = sum([label_aggregate_map[label]/4 for label in row[annotator_columns].values.tolist()])
        return [1-likelihoods, likelihoods]

    def prepare_data(self):
        if not self.force and Path(self.data_dir, "task_1_train.csv").exists() and Path(self.data_dir, "task_1_validation.csv").exists() and Path(self.data_dir, "task_1_test.csv").exists():
            print("Data already prepared, skipping preparation.")
            pass 
        else:
            print("Preparing data...")
            df = pd.read_csv(Path(self.data_dir, "task_1_annotation.csv"), index_col=False)
            df["soft_label"] = df.apply(self.get_soft_label, axis=1)
            df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
            df_train, df_valid = train_test_split(df_train, test_size=0.2, random_state=42)
            df_train.to_csv(Path(self.data_dir, "task_1_train.csv"), index=False)
            df_valid.to_csv(Path(self.data_dir, "task_1_validation.csv"), index=False)
            df_test.to_csv(Path(self.data_dir, "task_1_test.csv"), index=False)
            print("Data preparation complete.")

    def instantiate_hf_dataset(self, mode: str):
        df = pd.read_csv(Path(self.data_dir, f"task_1_{mode}.csv"), index_col=False)[["text", "soft_label"]]
        df['soft_label'] = df['soft_label'].apply(ast.literal_eval)
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(lambda batch: self.tokenizer(batch["text"], padding="max_length", max_length=4096, truncation=True), batched=True)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'soft_label'])
        return dataset
    
    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = self.instantiate_hf_dataset(mode="train")
            print("dataset column names", self.train_dataset.column_names)
            self.valid_dataset = self.instantiate_hf_dataset(mode="validation")
            
        if stage == "test":
            self.test_dataset = self.instantiate_hf_dataset(mode="test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class AverageMDMetric(torchmetrics.Metric):
    is_differentiable = False
    def __init__(self):
        super().__init__()
        self.add_state("distance_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples",    default=torch.tensor(0),    dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Args:
            preds   (Tensor): shape [B, C] on any device
            targets (Tensor): shape [B, C] on same device as preds
        """
        if preds.shape != targets.shape:
            raise ValueError("preds and targets must have identical shape")

        # ℓ₁ distance for each pair in batch, stay on‐device
        batch_dist = torch.abs(preds - targets).sum(dim=1)  # [B]
        self.distance_sum += batch_dist.sum()
        self.n_samples    += preds.size(0)

    def compute(self) -> float:
        if self.n_samples == 0:
            return torch.tensor(0.0)
        avg = self.distance_sum / self.n_samples
        return torch.tensor(round(avg.item(), 5))
    

class EncoderBasedClassifier(pl.LightningModule):
    def __init__(self, model_name="allenai/longformer-base-4096", num_labels=2, lr=2e-5, label_col="soft_label"):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = AutoModel.from_pretrained(model_name).train()
        self.linear = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.lr = lr
        self.label_col = label_col
        self.use_mean_pooling = True  # Use mean pooling instead of CLS token
        self.loss = L1Loss(reduction='mean') # Mean Absolute Error as baseline loss function
        self.valid_avg_md_metric = AverageMDMetric()
        self.test_avg_md_metric = AverageMDMetric()

    @torch.no_grad()
    def _build_global_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Longformer needs a global attention mask. Common practice: set the first token (CLS/<s>) to global.
        """
        gmask = torch.zeros_like(attention_mask)
        gmask[:, 0] = 1
        return gmask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns L2-normalized embeddings: [B, D]
        """
        global_attention_mask = self._build_global_attention_mask(attention_mask)
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            return_dict=True,
        )
        if self.use_mean_pooling:
            pooled = self.masked_mean_pooling(outputs.last_hidden_state, attention_mask)
        else:
            # CLS/<s> token at position 0 for Longformer (RoBERTa-like)
            pooled = outputs.last_hidden_state[:, 0]

        logits = self.linear(pooled)
        return logits
    
    @staticmethod
    def masked_mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean pooling with attention mask.
        last_hidden_state: [B, L, H]
        attention_mask:    [B, L] (1 for tokens, 0 for padding)
        returns:           [B, H]
        """
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, L, 1]
        summed = (last_hidden_state * mask).sum(dim=1)                  # [B, H]
        denom = mask.sum(dim=1).clamp(min=1e-6)                         # [B, 1]
        return summed / denom

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        labels = torch.tensor(batch[self.label_col], device=self.device)
        prob = nn.functional.softmax(logits, dim=-1).to(self.device)
        loss = self.loss(prob, labels) 
        self.log("train_mae_loss", loss, prog_bar=True, on_epoch=True, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        probs = nn.functional.softmax(logits, dim=-1).to(self.device)
        labels = torch.tensor(batch[self.label_col], device=self.device)
        loss = self.loss(probs, labels) 
        labels.to(self.device)
        self.valid_avg_md_metric.update(probs, labels)
        self.log("val_mae_loss", loss, prog_bar=True, on_epoch=True, sync_dist=False)
        return loss

    def test_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        labels = torch.tensor(batch[self.label_col], device=self.device)
        probs = nn.functional.softmax(logits, dim=-1).to(self.device)
        loss = self.loss(probs, labels) 
        labels.to(self.device)
        self.test_avg_md_metric.update(probs, labels)
        self.log("test_mae_loss", loss, prog_bar=True, on_epoch=True, sync_dist=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def on_validation_epoch_end(self):
        avg_md = self.valid_avg_md_metric.compute()
        self.log('val_avg_MD', avg_md, prog_bar=True, on_epoch=True, sync_dist=False)
        self.valid_avg_md_metric.reset()

    def on_test_epoch_end(self):
        avg_md = self.test_avg_md_metric.compute()
        self.log('test_avg_MD', avg_md, prog_bar=True, on_epoch=True, sync_dist=False)
        self.test_avg_md_metric.reset()


class SoftSupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.3, reduction='mean', normalize=True):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.normalize = normalize

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        N = logits.size(0)
        #normalized_targets = torch.nn.functional.normalize(targets, p=2, dim=1)
        w_ij = torch.matmul(targets, targets.T)
        if self.normalize:
            embeddings = torch.nn.functional.normalize(logits, p=2, dim=1)
        else:
            embeddings = logits
        distance_matrix_embeddings = torch.matmul(embeddings, embeddings.T) / self.temperature
        # for numerical stability
        distance_matrix_embeddings_max, _ = torch.max(distance_matrix_embeddings, dim=1, keepdim=True)
        distance_matrix_embeddings = distance_matrix_embeddings - distance_matrix_embeddings_max.detach()
        log_probs = torch.nn.functional.log_softmax(distance_matrix_embeddings, dim=1)
        loss = -torch.sum(w_ij * log_probs)/ (N*N)
        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            return loss * N * N
        else:
            return loss  # no reduction
        

class ContrastiveClassifier(EncoderBasedClassifier):
    def __init__(self, model_name="allenai/longformer-base-4096", num_labels=2, lr=2e-5, label_col="soft_label", tau=1.0, alpha=0.5):
        super().__init__(model_name=model_name, num_labels=num_labels, lr=lr, label_col=label_col)
        self.loss_sscl = SoftSupervisedContrastiveLoss(temperature=tau, normalize=False)
        self.loss_cls = L1Loss(reduction="mean")
        self.alpha = alpha
    
    def forward(self, input_ids, attention_mask, **kwargs):
        """
        Returns L2-normalized embeddings: [B, D]
        """
        global_attention_mask = self._build_global_attention_mask(attention_mask)
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            return_dict=True,
        )
        if self.use_mean_pooling:
            pooled = self.masked_mean_pooling(outputs.last_hidden_state, attention_mask)
        else:
            pooled = outputs.last_hidden_state[:, 0]
        normalized_features = torch.nn.functional.normalize(pooled, p=2, dim=1)
        logits = self.linear(normalized_features)
        logits = torch.nn.functional.normalize(logits, p=2, dim=1)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        labels = torch.tensor(batch[self.label_col], device=self.device)
        probs = nn.functional.softmax(logits, dim=-1).to(self.device)
        loss_cls = self.loss_cls(probs, labels) 
        loss_sscl = self.loss_sscl(logits, labels) 
        loss = self.alpha * loss_cls + (1 - self.alpha) * loss_sscl
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=False)
        self.log("train_mae_loss", loss_cls, prog_bar=True, on_epoch=True, sync_dist=False)
        self.log("train_sscl_loss", loss_sscl, prog_bar=True, on_epoch=True, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        probs = nn.functional.softmax(logits, dim=-1).to(self.device)
        labels = torch.tensor(batch[self.label_col], device=self.device)
        loss_cls = self.loss_cls(probs, labels) 
        loss_sscl = self.loss_sscl(logits, labels) 
        loss = self.alpha * loss_cls + (1 - self.alpha) * loss_sscl
        labels.to(self.device)
        self.valid_avg_md_metric.update(probs, labels)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=False)
        self.log("val_cls_loss", loss_cls, prog_bar=True, on_epoch=True, sync_dist=False)
        self.log("val_sscl_loss", loss_sscl, prog_bar=True, on_epoch=True, sync_dist=False)
        return loss

    def test_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        labels = torch.tensor(batch[self.label_col], device=self.device)
        probs = nn.functional.softmax(logits, dim=-1).to(self.device)
        loss_cls = self.loss_cls(probs, labels) 
        loss_sscl = self.loss_sscl(logits, labels) 
        loss = self.alpha * loss_cls + (1 - self.alpha) * loss_sscl
        labels.to(self.device)
        self.test_avg_md_metric.update(probs, labels)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, sync_dist=False)
        self.log("test_cls_loss", loss_cls, prog_bar=True, on_epoch=True, sync_dist=False)
        self.log("test_sscl_loss", loss_sscl, prog_bar=True, on_epoch=True, sync_dist=False)
        return loss
    

class PretrainingContrastiveClassifier(EncoderBasedClassifier):
    def __init__(self, model_name="allenai/longformer-base-4096", num_labels=100, lr=2e-5, label_col="soft_label", tau=1.0):
        super().__init__(model_name=model_name, num_labels=num_labels, lr=lr, label_col=label_col)
        self.loss_sscl = SoftSupervisedContrastiveLoss(temperature=tau, normalize=False)
        self.linear = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        """
        Returns L2-normalized embeddings: [B, D]
        """
        global_attention_mask = self._build_global_attention_mask(attention_mask)
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            return_dict=True,
        )
        if self.use_mean_pooling:
            pooled = self.masked_mean_pooling(outputs.last_hidden_state, attention_mask)
        else:
            pooled = outputs.last_hidden_state[:, 0]
        normalized_features = torch.nn.functional.normalize(pooled, p=2, dim=1)
        logits = self.linear(normalized_features)
        logits = torch.nn.functional.normalize(logits, p=2, dim=1)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        labels = torch.tensor(batch[self.label_col], device=self.device)
        loss_sscl = self.loss_sscl(logits, labels) 
        self.log("train_sscl_loss", loss_sscl, prog_bar=True, on_epoch=True, sync_dist=False)
        return loss_sscl

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        labels = torch.tensor(batch[self.label_col], device=self.device)
        loss_sscl = self.loss_sscl(logits, labels) 
        self.log("val_sscl_loss", loss_sscl, prog_bar=True, on_epoch=True, sync_dist=False)
        return loss_sscl

    def test_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        labels = torch.tensor(batch[self.label_col], device=self.device)
        loss_sscl = self.loss_sscl(logits, labels) 
        self.log("test_sscl_loss", loss_sscl, prog_bar=True, on_epoch=True, sync_dist=False)
        return loss_sscl        
    

class FinetuningClassifier(EncoderBasedClassifier):
    def __init__(self, model_name, num_labels=2, lr=2e-5, label_col="soft_label"):
        super().__init__(model_name=model_name, num_labels=num_labels, lr=lr, label_col=label_col)
        self.encoder = PretrainingContrastiveClassifier.load_from_checkpoint(f"./checkpoints/longformer_pretrained_sscl.ckpt")
        self.encoder.train()
        self.linear = nn.Linear(100, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        global_attention_mask = self._build_global_attention_mask(attention_mask)
        hidden_state = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            return_dict=True,
        )
        logits = self.linear(hidden_state)
        return logits
    

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    data_module = CDLDataModule(model_name="allenai/longformer-base-4096", batch_size=4, force=False)
    model = EncoderBasedClassifier(model_name="allenai/longformer-base-4096", num_labels=2, lr=2e-5, label_col="soft_label")
    exp_name = "longformer_mae"
    checkpoint_callback = ModelCheckpoint(monitor="val_avg_MD", mode="min", save_top_k=1, dirpath="checkpoints/", filename=exp_name, enable_version_counter=False)
    early_stop_callback = EarlyStopping(monitor="val_avg_MD", min_delta=0.001, patience=10, verbose=False, mode="min")
    device = model.device
    logger = CSVLogger("logs", name=exp_name)
    trainer = pl.Trainer(accelerator="auto", max_epochs=200, logger=logger, callbacks=[checkpoint_callback, early_stop_callback], strategy="ddp_find_unused_parameters_true")
    trainer.fit(model=model, datamodule=data_module)
    print("best model avg MD", round(checkpoint_callback.best_model_score.detach().cpu().item(), 5))
    trainer.test(ckpt_path="best", datamodule=data_module)
    print("test avg MD", round(trainer.callback_metrics["test_avg_MD"].detach().cpu().item(), 5))