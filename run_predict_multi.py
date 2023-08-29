import yaml
import pandas as pd

from sklearn.metrics import accuracy_score, r2_score

from torch.utils.data import DataLoader
from torch import nn
from src.core.multitask_predictions import generate_predictions_multitask

from src.models import MultitaskBERT
from src.optim import AdamW
from src.datasets import SSTDataset, SentenceSimilarityDataset
from src.utils import seed_everything, generate_device, logger
from src.core import train_validation_loop_multitask, generate_predictions

# TODO: check
if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        CONFIG = yaml.load(f, Loader=yaml.FullLoader)

    config_sst = CONFIG['data']['sst_dataset']
    config_quora = CONFIG['data']['quora_dataset']
    config_sts = CONFIG['data']['sts_dataset']
    config_dataloader = CONFIG['data']['dataloader']
    config_prediction = CONFIG['predict']


    config_bert = CONFIG['bert_model']
    config_train = CONFIG['train']

    seed_everything(CONFIG['seed'])
    device = generate_device(CONFIG['use_cuda'])

    # Create datasets

    sst_test_dataset = SSTDataset(
        config_sst['test_path'],
        return_targets=False
    )

    quora_test_dataset = SentenceSimilarityDataset(
        config_quora['test_path'],
        return_targets=False,
        index_col=False
    )

    sts_test_dataset = SentenceSimilarityDataset(
        config_sts['test_path'],
        binary_task=False,
        return_targets=False
    )

    # Create dataloader
    test_dataloaders = [
        DataLoader(
            x,
            shuffle=False,
            collate_fn=x.collate_fn,
            batch_size=config_dataloader['batch_size'],
            num_workers=config_dataloader['num_workers'],
        )
        for x in [sst_test_dataset, quora_test_dataset, sts_test_dataset]
        # the order of datasets must match the order in config.yaml (predictions save_path)
    ]

    model = MultitaskBERT(
        num_labels=5,
        option=config_bert['mode'],
        local_files_only=config_bert['local_files_only'],
        hidden_size=config_bert['hidden_size'],
        hidden_dropout_prob=config_bert['dropout_prob'],
        attention_dropout_prob=config_bert['dropout_prob'],
    )

    model = model.to(device)

    for test_loader, save_path in zip(test_dataloaders, config_prediction):
        predictions = generate_predictions_multitask(
            model=model,
            dataloader=test_loader,
            device=device,
            dataloader_message='test'
        )

        result = predictions.set_index(test_loader.dataset.dataset.index)
        result.to_csv(config_prediction[save_path])
