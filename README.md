# Sentiment Analysis and Happiness Scoring with BERT and TextBlob

## Overview

This project aims to provide sentiment analysis and happiness scoring for transcripts using a fine-tuned BERT model and TextBlob. The project has the following features:

- Sentiment Classification: Classify the sentiment of the text as Negative, Neutral, or Positive using a BERT-based classifier.
- Happiness Scoring: Calculate a happiness score using the TextBlob library and a set of trigger words.
- Trigger Word Detection: Identify and flag specific words in the text that can alter the happiness score.
- Batch Processing: Ability to analyze multiple transcripts in a directory.

## Prerequisites

- Python 3.7+
- PyTorch
- Transformers
- TextBlob

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/sentiment-analysis.git
    ```

2. Change into the project directory:

    ```bash
    cd sentiment-analysis
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preparation and Training

Run the following command to prepare the dataset and train the BERT-based model:

```bash
python data_preparation_and_training.py
```

### Sentiment Analysis

Run the following command:

```bash
python sentiment_analysis.py
```

### Troubleshooting

1. Ensure that the trained_model directory exists and contains the model files (config.json, pytorch_model.bin, vocab.txt).
2. If you run into any tokenization issues, make sure the tokenizer files are properly saved in the trained_model directory.
3. Make sure to use Python 3.7 or later versions.
4. Make sure you have enough space to train the model.