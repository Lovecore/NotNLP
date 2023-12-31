import os
import argparse
from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.nn.functional import softmax
from termcolor import colored

# Initialize Argument Parser
parser = argparse.ArgumentParser(description='Sentiment and Happiness Analysis')
parser.add_argument('--transcript_dir', default='transcripts/', type=str, help='Directory containing transcript files')
parser.add_argument('--trigger_file', default='trigger_words.txt', type=str, help='Path to trigger words file')
parser.add_argument('--sentiment_model_dir', default='./sentiment_model/', type=str, help='Path to sentiment model directory')
parser.add_argument('--emotion_model_dir', default='./emotion_model/', type=str, help='Path to emotion model directory')
parser.add_argument('--show_percent', action='store_true', help='Display emotion probabilities as percentages')
parser.add_argument('--output_file', default='results.txt', type=str, help='Output file for saving results')
parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')
args = parser.parse_args()

# Sentiment mapping
sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Initialize tokenizer and model for sentiment analysis
tokenizer = BertTokenizer.from_pretrained(args.sentiment_model_dir)
model = BertForSequenceClassification.from_pretrained(args.sentiment_model_dir)

# Initialize tokenizer and model for emotion analysis
emotion_tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
emotion_model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
emotion_map = {0: "anger", 1: "disgust", 2: "fear", 3: "joy", 4: "neutral", 5: "sadness", 6: "surprise"}  # Update based on the actual model

# The colors Duke, the colors!
def print_colored(text, color):
    print(colored(text, color))

def print_colored_sentiment(sentiment):
    color_map = {'Negative': 'red', 'Neutral': 'yellow', 'Positive': 'green'}
    return colored(sentiment, color_map[sentiment])

# Function to classify emotion based on https://huggingface.co/j-hartmann/emotion-english-distilroberta-base
def classify_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = emotion_model(**inputs)
    logits = outputs.logits
    softmax = torch.nn.functional.softmax(logits, dim=1)
    prediction = torch.argmax(logits, dim=1).item()
    emotion_probability = {emotion_map[i]: float(softmax[0][i]) for i in range(len(emotion_map))}
    return emotion_map[prediction], emotion_probability

# Formatting the emotion probabilities for better readability
def format_emotion_probability(emotion_probability, show_percent=False):
    color_map = {
        "anger": "red",
        "disgust": "magenta",
        "fear": "white",
        "joy": "cyan",
        "neutral": "white",
        "sadness": "yellow",
        "surprise": "magenta"
    }
    if show_percent:
        formatted_emotion_probability = '\n'.join([f"{colored(k, color_map[k])}: {v*100:.2f}%" for k, v in emotion_probability.items()])
    else:
        formatted_emotion_probability = '\n'.join([f"{colored(k, color_map[k])}: {v:.4f}" for k, v in emotion_probability.items()])
    return f"Predicted Emotion Probabilities:\n{formatted_emotion_probability}"


# Function to classify sentiment
def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return sentiment_map[prediction]

# Function to read transcript
def read_transcript_from_file(filename):
    with open(filename, 'r') as f:
        return f.read()
    
# Blog analysis
def analyze_text(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment

# Function to load trigger words
def load_trigger_words(file_path):
    with open(file_path, 'r') as file:
        return [line.strip().lower() for line in file.readlines()]

if __name__ == "__main__":
    try:
        # Load trigger words
        trigger_words = load_trigger_words(args.trigger_file)

        for filename in os.listdir(args.transcript_dir):
            if filename.endswith(".txt"):
                transcript_path = os.path.join(args.transcript_dir, filename)
                transcript = read_transcript_from_file(transcript_path)

                if args.verbose:
                    print(f"\nAnalyzing transcript from file: {filename}")

                # Classify sentiment and color it
                sentiment = classify_sentiment(transcript)
                colored_sent = print_colored_sentiment(sentiment)
                
                # Classify emotion and its probabilities
                emotion, emotion_probability = classify_emotion(transcript)
                
                if args.verbose:
                    print(f"Predicted Sentiment: {colored_sent}")
                    print(format_emotion_probability(emotion_probability, show_percent=args.show_percent))

                # Save or print results
                if args.output_file:
                    with open(args.output_file, 'a') as f:
                        f.write(f"Results for file: {filename}\n")
                        f.write(f"Predicted Sentiment: {colored_sent}\n")
                        f.write(format_emotion_probability(emotion_probability) + '\n')
                        f.write("="*40 + "\n")
                else:
                    print(f"Results for file: {filename}")
                    print(f"Predicted Sentiment: {colored_sent}")
                    print(format_emotion_probability(emotion_probability))
                    print("="*40)
                
    except Exception as e:
        print(f"An unexpected error occurred: {e}")