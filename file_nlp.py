from textblob import TextBlob
import os
import torch
import logging
from transformers import BertTokenizer, BertForSequenceClassification

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Updated label mapping
sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

try:
    tokenizer = BertTokenizer.from_pretrained('./trained_model/')
    model = BertForSequenceClassification.from_pretrained('./trained_model/')
except Exception as e:
    logging.error(f"Failed to load model or tokenizer: {e}")
    exit(1)

def classify_sentiment(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        logits = outputs.logits
        softmax = torch.nn.functional.softmax(logits, dim=1)
        prediction = torch.argmax(softmax, dim=1)
        human_readable_label = sentiment_map.get(prediction.item(), "Unknown")
        return prediction, human_readable_label
    except Exception as e:
        logging.error(f"Error in sentiment classification: {e}")
        return None, "Unknown"

def analyze_text(text):
    try:
        blob = TextBlob(text)
        sentiment = blob.sentiment
        return sentiment
    except Exception as e:
        logging.error(f"Error in text analysis: {e}")

def load_trigger_words(file_path):
    try:
        with open(file_path, 'r') as file:
            return [line.strip().lower() for line in file.readlines()]
    except Exception as e:
        logging.error(f"Error loading trigger words: {e}")

def calculate_happiness_with_triggers(transcript, trigger_words):
    text_blob = TextBlob(transcript)
    polarity = text_blob.sentiment.polarity
    subjectivity = text_blob.sentiment.subjectivity
    happiness = (1 + polarity) * (1 - 0.5 * abs(subjectivity - 0.5))
    
    found_trigger_words = [word for word in trigger_words if word in transcript.lower().split()]
    if found_trigger_words:
        logging.info(f"Trigger words detected: {', '.join(found_trigger_words)}")
        happiness *= 0.5
    
    return happiness

if __name__ == "__main__":
    try:
        trigger_words = load_trigger_words('trigger_words.txt')
        transcript_dir = 'transcripts/'
        
        for filename in os.listdir(transcript_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(transcript_dir, filename)
                
                with open(filepath, 'r') as f:
                    transcript = f.read()
                
                logging.info(f"Analyzing {filename}...")
                
                happiness = calculate_happiness_with_triggers(transcript, trigger_words)
                logging.info(f"Happiness Score considering triggers: {happiness}")

                predicted_label, human_readable_label = classify_sentiment(transcript)
                logging.info(f"Predicted Transformer Sentiment (tensor): {predicted_label}")
                logging.info(f"Predicted Transformer Sentiment (human-readable): {human_readable_label}")

                sentiment = analyze_text(transcript)
                if sentiment:
                    logging.info(f"Sentiment: {sentiment}")
                    logging.info(f"Happiness Score: {happiness}\n")
                else:
                    logging.error("Sentiment analysis failed.")
                    
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
