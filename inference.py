import os
import json
import glob
import re
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from models.encoders_task_1 import EncoderBasedClassifier
import pytorch_lightning as pl
from transformers import LongformerTokenizer
from huggingface_hub import hf_hub_download


class NarrativeClassification(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = model

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
        logits = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            return_dict=True,
        )
        
        probs = torch.softmax(logits, dim=-1)
        return probs    

def load_narrative_classifier(checkpoint_path = "./models/checkpoints/task_1/longformer_mae.ckpt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "allenai/longformer-base-4096"
    tokenizer = LongformerTokenizer.from_pretrained(model_name)
    
    model = EncoderBasedClassifier(model_name=model_name, num_labels=2)
    
    # Load checkpoint
    model = EncoderBasedClassifier.load_from_checkpoint(checkpoint_path)

    model = NarrativeClassification(model)
    model.to(device)
    model.eval()
    return model, tokenizer, device

 
def classify_narratives(
    texts: list[str],
    batch_size: int = 4,
    checkpoint_path: str = "./models/checkpoints/task_1/longformer_mae.ckpt",
) -> list[list[float]]:
    """
    Classify a list of texts as inflation-related or not.
 
    Pipeline:
      1. Texts without any inflation keyword  → default [0.999, 0.001]
      2. Texts with at least one keyword      → batched Longformer inference
      3. Results are concatenated in original order and returned.
 
    Args:
        texts:           Input texts to classify.
        batch_size:      Mini-batch size for model inference (default 4).
        checkpoint_path: Path to the LightningModule checkpoint.
 
    Returns:
        List of [p_non_inflation, p_inflation] probability pairs,
        one per input text, in the original order.
    """
    model, tokenizer, device = load_narrative_classifier(checkpoint_path)
    print("narrative identification device: ", device)

    # ── Step 1: keyword filter ────────────────────────────────────────────────
    # predictions[i] = final result or None (= needs model inference)
    predictions: list[list[float] | None] = []
    keyword_indices: list[int] = []          # original positions that need inference
    keyword_texts:   list[str]  = []         # corresponding texts
 
    for i, text in tqdm(enumerate(texts)):
        predictions.append(None)          # placeholder
        keyword_indices.append(i)
        keyword_texts.append(text)

 
    # ── Step 2: batched inference for keyword texts ───────────────────────────
    if keyword_texts:
        model_predictions: list[list[float]] = []
 
        for start in range(0, len(keyword_texts), batch_size):
            batch_texts = keyword_texts[start : start + batch_size]
 
            encoded = tokenizer(
                batch_texts,
                max_length=4096,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
 
            input_ids      = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
 
            with torch.no_grad():
                probs = model(input_ids, attention_mask)   # [B, 2]
 
            model_predictions.extend(probs.cpu().tolist())
 
        # ── Step 3: place model results back into original positions ──────────
        for original_idx, probs in zip(keyword_indices, model_predictions):
            predictions[original_idx] = probs
 
    return predictions  # list of [p_non_inflation, p_inflation]
    


class TripleExtractorInference:
    def __init__(
        self,
        checkpoint_path: str,
        base_model_name: str,
        prompt_file: str = "./models/prompts/prompt_triples.txt",
        load_in_4bit: bool = True,
        device: str = None,
    ):
        """
        Inference wrapper for the trained GenerativeTripleExtractor model.

        Args:
            checkpoint_path:  Path to the saved LoRA/DoRA adapter directory
                              (the one containing adapter_config.json).
            base_model_name:  HuggingFace model ID used during training,
                              e.g. "meta-llama/Llama-3.2-3B-Instruct".
            prompt_file:      Path to the same prompt template used at training time.
            all_events:       Set of valid event strings for output validation.
                              If None, validation is skipped.
            all_relations:    Set of valid relation strings for output validation.
                              If None, validation is skipped.
            load_in_4bit:     Quantise the base model with the same NF4 config
                              used during training (recommended, saves VRAM).
            device:           "cuda" / "cpu" / None (auto-detects).
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.base_model_name = base_model_name
        self.all_events = self.get_all_events()
        self.all_relations = self.get_all_relations()
        self.validate_outputs = bool(self.all_events and self.all_relations)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("narrative extraction device: ", self.device)
        self.prompt_template = self._load_prompt_template(prompt_file)
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model(load_in_4bit)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _load_prompt_template(self, prompt_file: str) -> str:
        with open(prompt_file, "r") as f:
            return f.read().strip()

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path,          # saved alongside the adapter
            token=os.getenv("HF_TOKEN"),
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def _load_model(self, load_in_4bit: bool):
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device} if self.device == "cuda" else "auto",
            token=os.getenv("HF_TOKEN"),
            trust_remote_code=True,
        )

        model = PeftModel.from_pretrained(base_model, self.checkpoint_path)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Prompt construction  (mirrors training-time logic)
    # ------------------------------------------------------------------

    def _build_prompt(self, text: str) -> str:
        return self.prompt_template + "\n\n" + f"Text: {text}\n\nOutput:"

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict(
        self,
        text: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        do_sample: bool = False,
    ) -> list[tuple[str, str, str]]:
        """
        Run inference on a single text and return extracted triples.

        Returns:
            List of (event_a, relation, event_b) tuples.
        """
        prompt = self._build_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode only the newly generated tokens (strip the prompt)
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return self._parse_triples(generated_text)

    def predict_batch(
        self,
        texts: list[str],
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        do_sample: bool = False,
        batch_size: int = 4,
    ) -> list[list[tuple[str, str, str]]]:
        """
        Run inference on a list of texts.  Processes in mini-batches to
        avoid OOM while still being faster than one-by-one generation.

        Returns:
            List of triple-lists, one per input text.
        """
        all_results = []

        for start in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[start : start + batch_size]
            prompts = [self._build_prompt(t) for t in batch_texts]

            # Left-pad so all prompts align on the right (generation side)
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.model.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Strip prompt tokens from each sequence
            prompt_len = inputs["input_ids"].shape[-1]
            for seq in outputs:
                new_tokens = seq[prompt_len:]
                generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                all_results.append(self._parse_triples(generated_text))

        return all_results

    # ------------------------------------------------------------------
    # Parsing  (mirrors GenerativeTripleExtractor.parse_generated_triples)
    # ------------------------------------------------------------------

    def _parse_triples(self, generated_text: str) -> list[tuple[str, str, str]]:
        """Parse generated JSON text into a list of (event_a, relation, event_b) tuples."""
        try:
            json_text = generated_text.strip()
            json_text = json_text.replace("```json", "").replace("```", "").strip()

            try:
                data = json.loads(json_text)
            except json.JSONDecodeError:
                json_text = json_text.replace(",]", "]").replace(",}", "}")
                try:
                    data = json.loads(json_text)
                except Exception:
                    return self._parse_malformed(json_text)

            if isinstance(data, dict) and "triples" in data:
                triples_list = data["triples"]
                if isinstance(triples_list, list):
                    parsed = []
                    for triple in triples_list:
                        if isinstance(triple, dict):
                            event_a = triple.get("event_a", "").strip()
                            relation = triple.get("relation", "").strip()
                            event_b = triple.get("event_b", "").strip()
                            if self._is_valid(event_a, relation, event_b):
                                parsed.append((event_a, relation, event_b))
                    return parsed

            return []

        except Exception as e:
            print(f"[TripleExtractorInference] Parse error: {e}")
            return []

    def _parse_malformed(self, text: str) -> list[tuple[str, str, str]]:
        """Regex fallback for malformed JSON."""
        pattern = (
            r'"event_a"\s*:\s*"([^"]+)"[^}]*'
            r'"relation"\s*:\s*"([^"]+)"[^}]*'
            r'"event_b"\s*:\s*"([^"]+)"'
        )
        triples = []
        for event_a, relation, event_b in re.findall(pattern, text):
            event_a, relation, event_b = event_a.strip(), relation.strip(), event_b.strip()
            if self._is_valid(event_a, relation, event_b):
                triples.append((event_a, relation, event_b))
        return triples

    def _is_valid(self, event_a: str, relation: str, event_b: str) -> bool:
        """Validate a triple against known events/relations (if provided)."""
        if not self.validate_outputs:
            # Accept anything non-empty when no vocabulary is supplied
            return bool(event_a and relation and event_b)
        return (
            event_a in self.all_events
            and relation in self.all_relations
            and event_b in self.all_events
        )
    
    def get_all_relations(self):
        return ["Increases", "Decreases"]
    
    def get_all_events(self):
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



if __name__ == "__main__":
    # download ckpt
    task_1_ckpt_path = "./models/checkpoints/task_1/longformer_mae.ckpt"
    if not Path(task_1_ckpt_path).exists():
        print(f"Checkpoint 1 not existing, downloading...")
        hf_hub_download(repo_id="tinyfeet/longformer-base-4096-mae", filename="longformer_mae.ckpt", local_dir="./models/checkpoints/task_1")
    task_2_ckpt_path = "./models/checkpoints/task_2/"
    if not Path(task_2_ckpt_path).exists():
        print(f"Checkpoint 2 not existing, downloading...")
        hf_hub_download(repo_id="tinyfeet/llama_Llama-3.2-1B_dora", local_dir="./models/checkpoints/task_2")
    
    # sample data
    # done: 1984 - 1996
    # ongoing: 1996 - 2000, 2001-2011, 2011-2024
    years = list(range(2011, 2024))

    for year in tqdm(years):
        file = f"./data/DJN/samples/random_samples/{year}.csv"
        df = pd.read_csv(file, index_col=False)
        df = df.drop(columns=["Unnamed: 0"], errors="ignore")
        narratives = df.text.tolist()

        # task 1: narrative classification
        classification_predictions = classify_narratives(texts=narratives, batch_size=50, checkpoint_path=task_1_ckpt_path)
        inflation_narratives = [text for text, pred in zip(narratives, classification_predictions) if pred[1] > 0.5]
        df["p_inflation"]     = [pred[1] for pred in classification_predictions]

        # task 2: narrative extraction
        vocab_path  = os.path.join(task_2_ckpt_path, "evaluation_results.json")

        extractor = TripleExtractorInference(
            checkpoint_path  = task_2_ckpt_path,
            base_model_name  = "meta-llama/Llama-3.2-1B",
            prompt_file      = "./models/prompts/prompt_triples.txt",
            load_in_4bit     = True,
        )
        # Batch predictinon
        batch_results = extractor.predict_batch(inflation_narratives, batch_size=8)
        # Map triples back to original dataframe rows
        triples_map = {text: triples for text, triples in zip(inflation_narratives, batch_results)}
        df["triples"] = df["text"].map(lambda t: json.dumps(triples_map[t]) if t in triples_map else json.dumps([]))
        del extractor

        # Save
        output_path = f"./data/predictions/random_sample/{year}.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} rows to {output_path}")
        print(f"{year}: inflation narratives/ all narratives = {len(inflation_narratives)}/{len(narratives)}")