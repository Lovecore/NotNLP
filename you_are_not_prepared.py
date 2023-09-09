from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch import nn, optim
import torch
import json
import logging

logging.basicConfig(level=logging.INFO)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_dataset(json_file_path):
    with open(json_file_path, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    logging.info("Loading the dataset...")
    dataset = load_dataset("reviews.json")
    texts = [item['text'] for item in dataset]
    labels = [item['label'] + 1 for item in dataset]  # Shift labels from -1, 0, 1 to 0, 1, 2

    logging.info("Initializing the tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(texts, truncation=True, padding=True)

    logging.info("Creating a PyTorch Dataset...")
    dataset = CustomDataset(encodings, labels)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logging.info("Setting up training configurations...")
    training_args = TrainingArguments(
        output_dir='./trained_model',
        num_train_epochs=1000,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        logging_dir='./logs',
    )

    logging.info("Initializing the model...")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    logging.info("Initializing the Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    logging.info("Starting training...")
    trainer.train()

    logging.info("Saving the trained model...")
    model.save_pretrained('./trained_model')
    tokenizer.save_pretrained('./trained_model')

    logging.info("Training complete.")
