from textblob import TextBlob
import os
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    softmax = torch.nn.functional.softmax(logits, dim=1)
    prediction = torch.argmax(softmax, dim=1)
    sentiment_map = {0: "Negative", 1: "Positive"}
    human_readable_label = sentiment_map.get(prediction.item(), "Unknown")
    return prediction, human_readable_label

def read_transcript_file(transcript_file: str):
    try:
        print(f"Reading transcript file from {transcript_file}...")
        with open(transcript_file, 'r') as file:
            transcript = file.read()
        print("Transcript successfully loaded.")
        return transcript
    except Exception as e:
        print(f"An error occurred while reading the transcript file: {e}")
        return None

def analyze_text(text):
    try:
        print("Analyzing text for sentiment...")
        blob = TextBlob(text)
        sentiment = blob.sentiment
        print(f"Sentiment analysis complete: {sentiment}")
        return sentiment
    except Exception as e:
        print(f"Error in text analysis: {e}")

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
        # Uncomment the following line to read from a transcript file
        transcript = read_transcript_file('orlandohealth.txt')

        if transcript:
            # Trigger words
            trigger_words_file_path = 'trigger_words.txt'
            trigger_words = load_trigger_words(trigger_words_file_path)

            happiness = calculate_happiness_with_triggers(transcript, trigger_words)
            print(f"Happiness Score considering triggers: {happiness}")

            predicted_label, human_readable_label = classify_sentiment(transcript)
            print(f"Predicted Transformer Sentiment (tensor): {predicted_label}")
            print(f"Predicted Transformer Sentiment (human-readable): {human_readable_label}")

            sentiment = analyze_text(transcript)
            if sentiment:
                print(f"Sentiment: {sentiment}")    
                print(f"Happiness Score: {happiness}")
        else:
            print("Transcription or Sentiment Analysis failed. Check above messages for details.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
