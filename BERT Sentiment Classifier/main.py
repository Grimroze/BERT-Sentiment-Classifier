# Sentiment Classifier - Single Script

import os
import random
import numpy as np
import torch
from datasets import load_dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

# 0) Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 1) Config
model_name = "distilbert-base-uncased"
max_length = 256
batch_size = 16
num_epochs = 2
lr = 5e-5
weight_decay = 0.01
warmup_ratio = 0.06
output_dir = "./sentiment_outputs"

os.makedirs(output_dir, exist_ok=True)

# 2) Load dataset (IMDB)
# If your notebook used a different dataset, swap here accordingly.
raw_datasets = load_dataset("imdb")
# Split train into train/validation for faster experiments
raw_datasets = raw_datasets.rename_column("text", "sentence")

# 3) Labels
label_list = [0, 1]  # 0=neg, 1=pos
num_labels = len(label_list)

# 4) Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

def preprocess_function(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        max_length=max_length,
    )

encoded = {}
for split in raw_datasets:
    encoded[split] = raw_datasets[split].map(preprocess_function, batched=True, remove_columns=["sentence"])

# 5) Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

# 6) Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 7) Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# 8) Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=weight_decay,
    warmup_ratio=warmup_ratio,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=50,
    report_to=[]  # disable wandb/tensorboard by default
)

# 9) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"].shuffle(seed=seed).select(range(20000)),  # smaller subset to speed up if desired
    eval_dataset=encoded["test"].select(range(5000)),
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 10) Train
train_result = trainer.train()
print(train_result)

# 11) Evaluate
metrics = trainer.evaluate()
print(metrics)

# 12) Save
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# 13) Inference helper
def predict_sentences(sentences: List[str], topk: int = 2) -> List[Tuple[str, Dict[str, float]]]:
    toks = tokenizer(
        sentences,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt"
    )
    model.eval()
    with torch.no_grad():
        outputs = model(**{k: v.to(model.device) for k, v in toks.items()})
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    results = []
    for s, p in zip(sentences, probs):
        scores = {id2label[i]: float(p[i]) for i in range(len(p))}
        results.append((s, scores))
    return results

# 14) Quick test
examples = [
    "This movie was absolutely wonderful and uplifting!",
    "Terrible plot and wooden acting."
]
for s, scores in predict_sentences(examples):
    print(s, "->", scores)
