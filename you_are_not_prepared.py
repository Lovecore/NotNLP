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
    parser = argparse.ArgumentParser(description='Train a BERT model for sentiment analysis.')
    parser.add_argument('--data', type=str, default='reviews.json', help='Path to the dataset JSON file.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs.')
    parser.add_argument('--train_batch_size', type=int, default=16, help='Training batch size.')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='Evaluation batch size.')
    args = parser.parse_args()

    logging.info(f"Loading the dataset from {args.data}...")
    dataset = load_dataset(args.data)
    texts = [item['text'] for item in dataset]
    labels = [item['label'] + 1 for item in dataset]  # Shift labels from -1, 0, 1 to 0, 1, 2

    logging.info("Initializing the tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(texts, truncation=True, padding=True)

    logging.info("Creating a PyTorch Dataset...")
    dataset = CustomDataset(encodings, labels)

    train_size = int(0.8 * len(dataset)) # Thijs might be wrong
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logging.info("Setting up training configurations...")
    training_args = TrainingArguments(
        output_dir='./trained_model',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
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