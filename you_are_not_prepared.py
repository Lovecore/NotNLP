import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification

# Load JSON dataset
with open('reviews.json', 'r') as f:
    data = json.load(f)

texts = [item['text'] for item in data]
labels = [item['label'] + 1 for item in data]  # Shift labels to start from 0

# Split dataset
texts_train, texts_val, labels_train, labels_val = train_test_split(texts, labels, test_size=0.2)

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(labels)))

optimizer = optim.AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

# Training loop (simplified)
for epoch in range(3):  # Change the number of epochs if needed
    for text, label in zip(texts_train, labels_train):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        loss = criterion(outputs.logits, torch.tensor([label]))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} completed")

# Save the model
model.save_pretrained('./trained_model/')
