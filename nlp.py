from google.cloud import speech
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
    
    sentiment_map = {0: "Negative", 1: "Positive"}  # Update this mapping as per your model's output
    human_readable_label = sentiment_map.get(prediction.item(), "Unknown")
    
    return prediction, human_readable_label

def transcribe_file(speech_file: str, sample_rate_hertz=44100):
    try:
        print("")
        print("Initializing Google Speech-to-Text client...")
        client = speech.SpeechClient()

        print(f"Reading audio file from {speech_file}...")
        with open(speech_file, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate_hertz,
            language_code="en-US",
            use_enhanced=True,
            model="phone_call"
        )

        print("Sending audio to Google Cloud for transcription...")
        operation = client.long_running_recognize(config=config, audio=audio)

        print("Waiting for operation to complete...")
        response = operation.result(timeout=90)

        transcript_text = ''
        
        # Newer model?
        for result in response.results:
            transcript = result.alternatives[0].transcript

        # Use Transformers model to predict the sentiment
        predicted_label = classify_sentiment(transcript)
        #print(f"Predicted Transformer Sentiment: {predicted_label}")

        for result in response.results:
            transcript_text += result.alternatives[0].transcript

        print(f"Transcription complete!")

        transcript_file = os.path.splitext(speech_file)[0] + ".txt"

        with open(transcript_file, 'w') as f:
                for result in response.results:
                   f.write(f"{result.alternatives[0].transcript}\n")
	
        print(f"Transcript saved to {transcript_file}\n")

        return transcript_text

    except Exception as e:
        print(f"An error occurred: {e}")

def analyze_text(text):
    try:
        blob = TextBlob(text)
        sentiment = blob.sentiment
        #print(f"Sentiment analysis complete: {sentiment}")
        return sentiment
    except Exception as e:
        print(f"Error in text analysis: {e}")

def load_trigger_words(file_path):
    with open(file_path, 'r') as file:
        return [line.strip().lower() for line in file.readlines()]

def calculate_happiness(polarity, subjectivity):
    # An example formula that considers both polarity and subjectivity
    return (1 + polarity) * (1 - 0.5*abs(subjectivity - 0.5))

def calculate_happiness_with_triggers(transcript, trigger_words):
    text_blob = TextBlob(transcript)
    polarity = text_blob.sentiment.polarity
    subjectivity = text_blob.sentiment.subjectivity
    happiness = (1 + polarity) * (1 - 0.5 * abs(subjectivity - 0.5))

    # Detect trigger words to modulate the happiness score
    found_trigger_words = []
    for word in trigger_words:
        if word in transcript.lower().split():  # Splitting to compare words, not substrings
            found_trigger_words.append(word)
            happiness *= 0.5  # Halves the happiness score if a trigger word is found

    if found_trigger_words:
        print(f"Trigger words detected: {', '.join(found_trigger_words)}")

    return happiness

if __name__ == "__main__":
    try:
        sample_rate_hertz = 44100
        transcript = transcribe_file('orlandohealth.wav', sample_rate_hertz)

        if transcript:
            # Trigger words
            trigger_words_file_path = 'trigger_words.txt'
            trigger_words = load_trigger_words(trigger_words_file_path)

            # Calculate happiness considering triggers
            happiness = calculate_happiness_with_triggers(transcript, trigger_words)
            print("Scoring:\n")
            print(f"Happiness Score considering triggers: {happiness}")

            # Transformer sentiment
            predicted_label, human_readable_label = classify_sentiment(transcript)
            print(f"Predicted Transformer Sentiment (tensor): {predicted_label}")
            print(f"Predicted Transformer Sentiment (human-readable): {human_readable_label}")

            # TextBlob sentiment
            sentiment = analyze_text(transcript)
            if sentiment:
                print(f"Sentiment Values: {sentiment}")    

        else:
            print("Transcription or Sentiment Analysis failed. Check above messages for details.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
