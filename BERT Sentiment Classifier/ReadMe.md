BERT Sentiment Classifier (PyTorch + HuggingFace)

A simple and effective sentiment classification model built using BERT, PyTorch, and the HuggingFace Transformers library.

This project fine-tunes a pre-trained BERT model on the IMDB movie reviews dataset to classify text as positive or negative.

🚀 Features

Uses pre-trained BERT (bert-base-uncased)

Fine-tuned on the IMDB sentiment dataset

Achieves strong accuracy with very little training

Training powered by HuggingFace Trainer API

Fast inference with custom prediction function

Runs on GPU or CPU

Beginner-friendly PyTorch implementation

📊 Dataset

Dataset: IMDB Movie Reviews (25k train, 25k test)

Loaded directly from HuggingFace Datasets:

from datasets import load_dataset
dataset = load_dataset("imdb")


For faster experimentation, a small subset was used:

4,000 training samples

1,000 testing samples

🧠 Model

Model used:

bert-base-uncased


Fine-tuned for binary classification (positive/negative)

Model output layer size:

num_labels = 2

🛠️ Training

Training handled by HuggingFace Trainer:

Batch size: 8

Epochs: 2

Learning rate: 2e-5

Weight decay: 0.01

Evaluation every epoch

Best model loaded automatically

Example training arguments:

training_args = TrainingArguments(
    output_dir="bert-imdb-sentiment",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    report_to="none",
)

🧪 Inference

Custom inference function:

label_names = ["negative", "positive"]

def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_id].item()

    return label_names[pred_id], confidence


Example:

text = "I absolutely loved this movie!"
label, conf = predict_sentiment(text)
print(label, conf)


Output:

positive 0.987

📝 Results

After fine-tuning for 2 epochs, the model achieves:

~85-90% accuracy (on small subset)

Good confidence scores

Robust performance on custom examples

Example predictions:

Text	Prediction	Confidence
"I loved this movie"	positive	0.98
"This was a terrible film"	negative	0.99