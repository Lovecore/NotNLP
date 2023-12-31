# Sentiment Analysis and Happiness Scoring with BERT

## Overview

This project aims to provide sentiment analysis and happiness scoring for transcripts using a fine-tuned BERT. The project has the following features:

- Sentiment Classification: Classify the sentiment of the text as Negative, Neutral, or Positive using a BERT-based classifier.
- Batch Processing: Ability to analyze multiple transcripts in a directory.
- Output feelings of the conversation based on an [emotion model](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)

## Prerequisites

- Python 3.7+
- PyTorch
- Transformers
- Termcolor
- Argparse

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Lovecore/NotNLP
    ```

2. Change into the project directory:

    ```bash
    cd NotNLP
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### You still need to have a 'review' or 'comment' in the transcripts directory. `nlp.py` is designed to be used in conjunction with Google Cloud Platform for translating Speech to Text, then analyzing the outcome. While the `file_nlp.py` is meant to simply be used to analyze the transcripts.

### Data Preparation and Training

**WARNING** The larger your cycles, the larger the space the model consumes, the current training model consumes ~30gb @ 1000 epochs. Adjust accordingly.

Run the following command to prepare the dataset and train the BERT-based model. You can specify a series of options: `--data`, `--epochs`, `--train_batch_size` and `--eval_batch_size`.

```bash
python you_are_not_prepared.py
```

### Sentiment Analysis

Run the following command to read from the transcript directory and output the value of each file. There are commands you can specify in order to get a bit more control over the run. You can specify a series of options: `--transcript_dir`, `--trigger_file`, `--sentiment_model_dir`, `--emotion_model_dir`, `--output_file`, `--show_percent` and the `--verbose` flag. 

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
~~Maybe some colors, people like colors.~~
Modulize.
Add larger data to the training set.
and other stuffs.