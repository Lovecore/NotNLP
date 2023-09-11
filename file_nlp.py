import os
import argparse
from textblob import TextBlob
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Initialize Argument Parser
parser = argparse.ArgumentParser(description='Sentiment and Happiness Analysis')
parser.add_argument('--transcript_dir', default='transcripts/', type=str, help='Directory containing transcript files')
parser.add_argument('--trigger_file', default='trigger_words.txt', type=str, help='Path to trigger words file')
parser.add_argument('--model_dir', default='./trained_model/', type=str, help='Path to trained model directory')
parser.add_argument('--context', default='general', type=str, help='Context for sentiment and happiness analysis')
parser.add_argument('--min_happiness', default=0.5, type=float, help='Minimum happiness score for "happy" label')
parser.add_argument('--min_sentence_length', default=5, type=int, help='Minimum sentence length for sentiment analysis')
parser.add_argument('--output_file', default='results.txt', type=str, help='Output file for saving results')
parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')
args = parser.parse_args()

# Update the sentiment mapping to reflect shifted labels
sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

tokenizer = BertTokenizer.from_pretrained(args.model_dir)
model = BertForSequenceClassification.from_pretrained(args.model_dir)

def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    softmax = torch.nn.functional.softmax(logits, dim=1)
    prediction = torch.argmax(softmax, dim=1)
    human_readable_label = sentiment_map.get(prediction.item(), "Unknown")
    return prediction, human_readable_label

def read_transcript_from_file(filename):
    with open(filename, 'r') as f:
        return f.read()

def analyze_text(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment

def load_trigger_words(file_path):
    with open(file_path, 'r') as file:
        return [line.strip().lower() for line in file.readlines()]

def calculate_happiness_with_triggers(transcript, trigger_words):
    text_blob = TextBlob(transcript)
    polarity = text_blob.sentiment.polarity
    subjectivity = text_blob.sentiment.subjectivity
    happiness = (1 + polarity) * (1 - 0.5 * abs(subjectivity - 0.5))
    found_trigger_words = []
    for word in trigger_words:
        if word in transcript.lower().split():
            found_trigger_words.append(word)
            happiness *= 0.5
    if found_trigger_words:
        print(f"Trigger words detected: {', '.join(found_trigger_words)}")
    return happiness

if __name__ == "__main__":
    try:
        # Load trigger words
        trigger_words = load_trigger_words(args.trigger_file)
        
        for filename in os.listdir(args.transcript_dir):
            if filename.endswith(".txt"):
                transcript_path = os.path.join(args.transcript_dir, filename)
                transcript = read_transcript_from_file(transcript_path)
                
                if args.verbose:
                    print(f"Analyzing transcript from file: {filename}")
                
                # Calculate happiness
                happiness = calculate_happiness_with_triggers(transcript, trigger_words)
                
                if args.verbose:
                    print(f"Happiness Score considering triggers: {happiness}")
                
                # Classify sentiment
                predicted_label, human_readable_label = classify_sentiment(transcript)
                
                if args.verbose:
                    print(f"Predicted Transformer Sentiment (tensor): {predicted_label}")
                    print(f"Predicted Transformer Sentiment (human-readable): {human_readable_label}")
                
                sentiment = analyze_text(transcript)
                
                if args.verbose:
                    print(f"Sentiment: {sentiment}")    
                
                # Save or print results
                if args.output_file:
                    with open(args.output_file, 'a') as f:
                        f.write(f"Results for file: {filename}\n")
                        f.write(f"Happiness Score: {happiness}\n")
                        f.write(f"Predicted Sentiment: {human_readable_label}\n")
                        f.write(f"Sentiment: {sentiment}\n")
                        f.write("="*40 + "\n")
                else:
                    print(f"Results for file: {filename}")
                    print(f"Happiness Score: {happiness}")
                    print(f"Predicted Sentiment: {human_readable_label}")
                    print(f"Sentiment: {sentiment}")
                    print("="*40)
                
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
