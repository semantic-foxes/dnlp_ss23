# Tasks

## Part 1: Stanford Sentiment Treebank

Using a movie review sentence, predict the movie's sentiment, which can be one of the following:
- Negative
- Somewhat negative
- Neutral
- Somewhat positive
- Positive

This is a multi-label classification problem. The pre-split data is available in:
- `data/ids-sst-train.csv` (Training set, 8544 entries)
- `data/ids-sst-dev.csv` (Validation set, 1101 entries)
- `data/ids-sst-test-student.csv` (Test set, 2210 entries)

## Part 2

### Stanford Sentiment Treebank
This dataset is identical to the one described in Part 1.

### Quora Dataset
The Quora dataset consists of pairs of questions, labeled to indicate if the questions are paraphrases of one another. This is a binary classification problem. The pre-split data is available in:
- `data/quora-train.csv` (Training set, 141506 entries)
- `data/quora-dev.csv` (Validation set, 20215 entries)
- `data/quora-test-student.csv` (Test set, 40431 entries)

### SemEval STS Benchmark Dataset
This dataset provides pairs of sentences and scores reflecting the similarity between each pair. As its structure is close to the Quora dataset, they share the same `dataset` class. The pre-split data is available in:
- `data/sts-train.csv` (Training set, 6041 entries)
- `data/sts-dev.csv` (Validation set, 864 entries)
- `data/sts-test-student.csv` (Test set, 1726 entries)

# Running the Code

## Setup
The project is developed in Python 3.8. To install necessary dependencies, run:
```
pip3 install -r requirements.txt
```
We recommend using a virtual environment like `conda` or `venv`.

## Configuration
The `config.yaml` file streamlines workflow by centralizing the model's parameters. As the tasks share similarities and use primarily the BERT core, a single config file serves both tasks.

## Execution
- For Task 1:
```
python3 run_task_1.py
```

- For Task 2:
```
python3 run_task_2.py
```

**All the commands are to be run from the project root.**

# Codebase Overview

The repository has undergone significant changes. Here's a brief overview:

- `src`: Contains shared code, further divided into:
  - `core`: Holds functions common to models like the training loop and prediction generation.
  - `datasets`: Houses all dataset classes. `SSTDataset` is for the SST task, while `SentenceSimilarityDataset` serves both Quora and SemEval datasets.
  - `models`: Includes all model classes, subdivided into:
    - `base_bert`: Non-maintained foundational class for BERT.
    - `bert`: Generates embeddings for input.
    - `classifier`: Multi-label classifier for the SST task.
    - `multitask_classifier`: Used for the second task. Has a shared BERT core and task-specific "heads".
  - `optim`: Contains our `AdamW` optimizer implementation.
  - `utils`: Features utility functions and logger settings. Import `logger` from this module to log events. The `utils.py` is third-party and not maintained by us, while `good_utils` is our contribution.
  
- `tests`: Contains various test scripts. To run:
  1. Add `src` to the `PYTHONPATH`.
  2. Execute:
    ```
    python3 tests/<test_filename>
    ```
**All the commands are to be run from the project root.**