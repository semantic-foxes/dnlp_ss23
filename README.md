# DNLP SS23 Final Project - Multitask BERT

Team: G04 Semantic Foxes

### Acknowledgements

The project description, partial implementation, and scripts were adapted from the default final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John, Hewitt, Amelie Byun, John Cho, and their (large) team (Thank you!)

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig  (Thank you!)

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

Parts of the scripts and code were altered by [Jan Philip Wahle](https://jpwahle.com/) and [Terry Ruas](https://terryruas.com/).

## Datasets

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

## Methodology

### BERT setup

We used `bert-base-uncased` BERT.

### Data combination

Since 3 datasets have different amount of data, we can combine them differently for training purposes.
There are several options for `dataloader_mode`

- `min` - crops all datasets by minimal length
- `sequential` - traverse the whole dataset one after another (good for pretrain)
- `continuos` - 'infinitely' iterate over datasets (one epoch is determined by the minimal length among datasets)
- `exhaust` - choose batch randomly without replacements

### Best metric selection

In order to compare metrics, we use their sum.

## Experiments

| Model | SST | Quora | STS |
|---|---|---|---|
| Pretrain | 0.387 | 0.699 | 0.261 |
| Finetune | 0.498 | 0.708 | 0.376 |

### Pretrain model

Using pretrained BERT with frozen weights, we used simple classifiers for SST and Quora datasets and a simple regressor for STS dataset.

### Finetune model

Same approach as for Pretrain model, but now BERT weights are not frozen.

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

## Codebase Overview

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

## Implemented features

### Gradual unfreeze [Sergei Zakharov], [Universal Language Model Fine-tuning](https://paperswithcode.com/method/ulmfit#:~:text=Universal%20Language%20Model%20Fine%2Dtuning%2C%20or%20ULMFiT%2C%20is%20an,LSTM%20architecture%20for%20its%20representations)

It has been shown that the gradual unfreeze of BERT layers can lead to
performance increase. To make use of this, we implemented the `BasicGradualUnfreezer` located at `src.core.unfreezer.BasicGradualUnfreezer`.
It allows to unfreeze BERT layers one by one (this behaviour can be modified in the `step` method) and is used almost the same as the PyTorch schedulers are used.
To make common functions for various future unfreezers possible, we also made the `GradualUnfreezerTemplate` from which the `BasicGradualUnfreezer` is inherited.

We can report that this feature almost always appeared to actually improve the quality in our tests.
This corresponds with the idea that the "top" layers of the model are to be trained
more heavy than the "bottom" ones.


### Scheduler support [Sergei Zakharov], [Universal Language Model Fine-tuning](https://paperswithcode.com/method/ulmfit#:~:text=Universal%20Language%20Model%20Fine%2Dtuning%2C%20or%20ULMFiT%2C%20is%20an,LSTM%20architecture%20for%20its%20representations)

It has been shown that using a scheduler leads to performance increase.
Though we ended up not using the scheduler provided in the stated article
and rather used the simple `ExponentialLR` from PyTorch. 

We can report that this feature almost always appeared to actually improve the quality in our tests.
This corresponds with the idea that the "top" layers of the model are to be trained
more heavy than the "bottom" ones.

### Hyperparameter search [Sergei Zakharov]

In order to automate some processes, we used the `optuna` framework. The
hyperparameters to tune are specified in the `configs/hyperparameter_config.yaml`
either in `low`-`high` format or as a list. The parameters not stated in the config
are taken from the general config `config.yaml`.

We ended up not using this feature much since it takes a lot of time
(running 14 trials took about 20 hours). However, we ran it once and
made sure that `lr` around `1e-5` is actually the optimal one in case we
talk about using not a completely frozen BERT.

### "Dilated" batches [Sergei Zakharov]

When we started exploring the data provided, we realized that the `Quora`
dataset is way larger than all the others. In order to make the training
more "uniform" for all the datasets, we decided to introduce different
effective batch sizes for the dataloaders.

Since we don't have the resource to actually make the batch size bigger,
we instead allow multiple forward passes across different batches
(the dataloader obviously remains the same) while
accumulating their gradients. An optimizer step is then taken using the mean
loss of these accumulated gradients.

The feature can be accessed by the `exhaust` dataloader mode in the `config.yaml`.
The "weight" are specified in each of the datasets in the same file and specify
the number of these forward passes done before the optimizer step is done.

This feature was implemented relatively early since we have the common
BERT core and want it to train on all the tasks rather than just on the `Quora`.
Hence, most of our runs shared this mode.

### [WandB](wandb.ai) logging [Sergei Zakharov]

Since we wanted to have a common space for our team to store the results,
we decided to use the [WandB](wandb.ai) for this purpose. This logging is
turned on by the `'wandb'` watcher type in `config.yaml`. In order to enable
it on the cluster as well (where we don't have the internet access when
using GPU for some reason), we also enabled the `offline` mode. In case
`offline` mode is used, the user should then manually sync the run to the
website when possible.

To disable this logging, use `null` value in the `config.yaml`. The
implementation is done in a way to enable a relatively easy add of other
watchers.

### CosineSimilarity head for STS task and prediction clipping [Sergei Zakharov]

We decided to use the CosineSimilarity layer rather than simple concatenation
of the two vectors in the tasks since this layer appears to be closer to
the original task by its idea. It proved to provide better results for us.

However, since the cosine similarity has [-1, 1] range, we needed to 
somehow project it to the desired [0, 5] range. To do that, we tried
various approaches and ended up with multiplying the output by 5 and
clipping it to the [0, 5] range.

This specific feature seems to be of real importance since it provides a major
boost on the metric. We tried other clipping strategies as well, but this is a
heuristic that worked the best for us.