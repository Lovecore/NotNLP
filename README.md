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

### You still need to have a 'review' or 'comment' in the transcripts directory. `nlp.py` is designed to be used in conjunction with Google Cloud Platform for translating Speech to Text, then analyzing the outcome. While the `file_nlp.py` is meant to simply be used to analyze the transcripts.

### Data Preparation and Training

**WARNING** The larger your cycles, the larger the space the model consumes, the current training model consumes ~60gb @ 1000 epochs. Adjust accordingly.

Run the following command to prepare the dataset and train the BERT-based model: 

```bash
python you_are_not_prepared.py
```

### Sentiment Analysis

Run the following command to read from the transcript directory and output the value of each file. There are commands you can specify in order to get a bit more control over the run. You can specify: `--transcript_dir`, `--trigger_file`, `--model_dir`, `--context`, `--min_happiness`, `--min_sentence_length`, `--output_file` and the `--verbose` flag. 

```bash
python file_nlp.py --verbose
```


### Troubleshooting

1. Ensure that the `trained_model` directory exists and contains the model files (`config.json`, `pytorch_model.bin`, `vocab.txt`. The first two are generated when first training the model with `you_are_not_prepared.py`).
2. If you run into any tokenization issues, make sure the tokenizer files are properly saved in the `trained_model` directory.
3. Make sure to use `Python 3.7` or later versions.
4. Make sure you have enough space to train the model.

### To Do

Adjust Happiness algorithm since its' VERY harsh. 
Consider `positive` words as well as the currently used negative words.
Better logging.
Maybe some colors, people like colors.
Modulize.
Add larger data to the training set.
and other stuffs.