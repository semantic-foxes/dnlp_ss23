# DNLP SS23 Final Project - Multitask BERT

Team: G04 Semantic Foxes

## Acknowledgements

The project description, partial implementation, and scripts were adapted from the default final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John, Hewitt, Amelie Byun, John Cho, and their (large) team (Thank you!)

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig  (Thank you!)

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

Parts of the scripts and code were altered by [Jan Philip Wahle](https://jpwahle.com/) and [Terry Ruas](https://terryruas.com/).

## Setup instructions

The project is developed in Python 3.8.

- Follow `setup_gwdg.sh` to properly setup a conda environment and install dependencies.

```python
source setup_gwdg.sh
```

- Alternatively, to install necessary dependencies, run:

```python
pip3 install -r requirements.txt
```

We recommend using a virtual environment like `conda` or `venv`.

## Configuration

The `config.yaml` file streamlines workflow by centralizing the model's parameters.

## Execution

- For Multiple Task Training, run:

```python
python3 run_train.py
```

- For pretraining on `Quora`, run:

```python
python3 pretrain_quora.py
```

- For hyperparameter search, run:

```python
python3 hyperparameter_search.py
```

- For Tests, run:

```python
python3 -m tests.optimizer_test
python3 -m tests.sanity_check
```

**All the commands are to be run from the project root.**

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

The metric to test this dataset is accuracy.

### Quora Dataset

The Quora dataset consists of pairs of questions, labeled to indicate if the questions are paraphrases of one another. This is a binary classification problem. The pre-split data is available in:

- `data/quora-train.csv` (Training set, 141506 entries)
- `data/quora-dev.csv` (Validation set, 20215 entries)
- `data/quora-test-student.csv` (Test set, 40431 entries)

The metric to test this dataset is accuracy.

### SemEval STS Benchmark Dataset

This dataset provides pairs of sentences and scores reflecting the similarity between each pair. As its structure is close to the Quora dataset, they share the same `dataset` class. The pre-split data is available in:

- `data/sts-train.csv` (Training set, 6041 entries)
- `data/sts-dev.csv` (Validation set, 864 entries)
- `data/sts-test-student.csv` (Test set, 1726 entries)

The metric to test this dataset is Pearson correlation.

## Methodology

### BERT setup

We used `bert-base-uncased` BERT.

### Multitask trainig stategy

Since 3 datasets have different amount of data, we can combine them differently for training purposes.
There are several options for `dataloader_mode`:

- `sequential` - traverse the whole dataset one after another (good for training with freezed BERT)
- `continuos` - 'infinitely' iterate over datasets (one epoch is determined by the minimal length of datasets)
- `exhaust` - choose batch randomly without replacements

Additionally, there are several options to enhance behaviour of these combinations:

- `weights` - determine how many batches to consider for each dataset (e.g. 1 from `SST`, 10 from `Quora` and 1 from `STS`).
- `skip_optimizer_step` - allows to make an optimizer step every n-th times.
- `is_multi_batch` - makes a 'multi-batch' from all datasets.

### Best metric selection

Since there is an ambiguity how one can compare `3` metrics simultaneously, we determine a better model simply using sum of metrics.

### Loss Functions

By default we are using the following losses:

- `CrossEntropyLoss` - `SST` and `Quora`
- `MSELoss` - `STS`

There is an option to use a negative Pearson correlation as a loss for `STS`.

Also, one can add additional
[`CosineEmbeddingsLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html)
for `Quora`.

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

### Continuous mode with Multi-batch [Danila Litskevich]

The idea is to sample from all datasets and produce one batch, which we call a 'multi-batch'.
That is performed in `continuos` mode by providing skipping `3` steps of optimizer using `skip_optimizer_step`
and setting `is_multi_batch` to true.

Opposed to "Dilated" batches approach, 'multi-batch' ensures that on `3n`, `3n+1`, `3n+2` steps we would sample from all datasets,
therefore creating a 'multi-batch'.

Since the datasets have different amount of samples,
`countinuous` mode allows to continue training by resetting an exhausted dataset without resetting others.
One epoch is determined by the minimal length of datasets.

Also, one can balance the amount of batches in a 'multi-batch' by providing corresponding `weight`'s.

This approach is robust and quick and performing well with other improvements, in particular with an additional pretrain on `Quora`.

### Additional pretrain on `Quora` [Danila Litskevich]

Since we have 10 times more data for `Quora` than for other datasets, we can use it to pretrain our model without overfitting during further finetuning.

Multi-batch approach turned out to preserve same good performance for `Quora` during finetuning while also achieving great results for other datasets.

### [`CosineEmbeddingsLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html) [Danila Litskevich]

`CosineEmbeddingsLoss` can be applied to `Quora` for futher regularization.

### Mix freeze/unfreeze steps [Danila Litskevich]

Another possible way to regularize our model is to mix optimization steps freezing and unfreezing BERT.
That behaviour is controlled by `freeze_bert_steps`.

It was enhancing the performance, however Gradual unfreeze is working better.

## Experiments

### MSE vs PearsonLoss [Danila Litskevich]

As a loss function for `STS` we use MSE loss by default. However, we can also use negative Pearson correlation as another loss, called PearsonLoss.

PearsonLoss has a significant downside that the predictions are not forced to be in the original targets range `[0,5]`.

On the other hand, MSE turned out to be sensitive to predictions range: we stick with `[-5,5]` range.
Due to that during evaluation and prediction's generation phrase we clip predictions to the original targets range `[0,5]`.
This clipping is known to reduce MSE loss as a projection.

Generally, MSE loss produce better results.

### Additional pretrain on Quora [Danila Litskevich]

Since `continuous` mode covers `SST` and `STS` datasets more offen than `Quora` due to their difference in size,
we were underfitting on `Quora` and overfitting on others.

We tried to mitigate that by providing corresponding `weight`'s.

Another approach is to first pretrain the model on `Quora` and then use our regular procedure for training.

That approach appeared to train on `SST` and `STS` datasets while preserving the quality on `Quora`.
In the end, it produced the best results.

## Results

| Model | SST | Quora | STS |
|---|---|---|---|
| Pretrain | 0.387 | 0.699 | 0.261 |
| Finetune | 0.498 | 0.708 | 0.376 |
| Exhaust | 0. | 0. | 0. |
| Continuous | 0.498 | 0.733 | 0.516 |
| Continuous<sub>Quora</sub> | 0.500 | 0.771 | 0.592 |

### Pretrain model

Using pretrained BERT with frozen weights, we used simple classifiers for SST and Quora datasets and a simple regressor for STS dataset.

Configuration file: `configs/pretrain-config.yaml`.

### Finetune model

Same approach as for Pretrain model, but now BERT weights are not frozen.

Configuration file: `configs/finetune-config.yaml`.

### Exhaust model

The model that utilize `exhaust` mode.

### Continuous model

The model that utilize `continuous` mode.

Configuration file: `config.yaml`.

### Continuous<sub>Quora</sub> model

This is our best model which combines Continuous model with additional pretraining on `Quora`.

Configuration file for pretrain on `Quora`: `configs/quora-pretrain-config.yaml`.

Configuration file: `config.yaml`.

To train the model, first run `pretrain_quora.py`, after that run `run_train.py`.

## Codebase Overview

The repository has undergone significant changes. Here's a brief overview:

- `configs`: Contains config files for training.
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

    ```python
    python3 tests/<test_filename>
    ```

**All the commands are to be run from the project root.**
