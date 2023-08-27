# DNLP SS23 Final Project - Multitask BERT

Team: G04 Semantic Foxes

### Acknowledgements

The project description, partial implementation, and scripts were adapted from the default final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John, Hewitt, Amelie Byun, John Cho, and their (large) team (Thank you!)

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig  (Thank you!)

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

Parts of the scripts and code were altered by [Jan Philip Wahle](https://jpwahle.com/) and [Terry Ruas](https://terryruas.com/).

# Datasets

### Stanford Sentiment Treebank

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

## Experiments

| Model | SST | Quora | STS |
|---|---|---|---|
| BERT<sub>pretrain</sub> | 0.387 | 0.699 | 0.261 |
### BERT<sub>pretrain</sub>
Using pretrained BERT with frozen weights, we used simple classifiers for SST and Quora datasets and a simple regressor for STS dataset.

## Setup instructions

The project is developed in Python 3.8.

- Follow `setup_gwdg.sh` to properly setup a conda environment and install dependencies.

```
source setup_gwdg.sh
```

- Alternatively, to install necessary dependencies, run:

```
pip3 install -r requirements.txt
```

We recommend using a virtual environment like `conda` or `venv`.

## Configuration

The `config.yaml` file streamlines workflow by centralizing the model's parameters.

## Execution

- For Sentiment Classification Task, run:

```
python3 run_task_1.py
```

- For Multiple Task Training, run:

```
python3 run_task_2.py
```

- For Tests, run:

```
python3 -m tests.optimizer_test
python3 -m tests.sanity_check
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
  - `utils`: Features utility functions and logger settings. Import `logger` from this module to log events. The `utils.py` is third-party and not maintained by us, while `model_utils` is our contribution.
  
- `tests`: Contains various test scripts. To run:
  1. Add `src` to the `PYTHONPATH`.
  2. Execute:

    ```
    python3 tests/<test_filename>
    ```

**All the commands are to be run from the project root.**
