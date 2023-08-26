import yaml
import pandas as pd

from torch.utils.data import DataLoader
from torch import nn

from src.models import MultitaskBERT
from src.optim import AdamW
from src.datasets import SSTDataset, SentenceSimilarityDataset
from src.utils import seed_everything, generate_device, logger
from src.core import train_validation_loop_multitask, generate_predictions_multitask
from src.metrics import accuracy, pearson_correlation
from src.utils.good_utils import load_state


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
    sst_train_dataset = SSTDataset(
        config_sst['train_path'],
        return_targets=True
    )
    sst_val_dataset = SSTDataset(
        config_sst['val_path'],
        return_targets=True
    )
    sst_test_dataset = SSTDataset(
        config_sst['test_path'],
        return_targets=False
    )

    quora_train_dataset = SentenceSimilarityDataset(
        config_quora['train_path'],
        return_targets=True
    )
    quora_val_dataset = SentenceSimilarityDataset(
        config_quora['val_path'],
        return_targets=True
    )
    quora_test_dataset = SentenceSimilarityDataset(
        config_quora['test_path'],
        return_targets=False
    )

    sts_train_dataset = SentenceSimilarityDataset(
        config_sts['train_path'],
        binary_task=False,
        return_targets=True
    )
    sts_val_dataset = SentenceSimilarityDataset(
        config_sts['val_path'],
        binary_task=False,
        return_targets=True
    )
    sts_test_dataset = SentenceSimilarityDataset(
        config_sts['test_path'],
        binary_task=False,
        return_targets=False
    )

    # Create dataloaders
    train_dataloaders = [
        DataLoader(
            x,
            shuffle=True,
            collate_fn=x.collate_fn,
            batch_size=config_dataloader['batch_size'],
            num_workers=config_dataloader['num_workers'],
        )
        for x in [sst_train_dataset, quora_train_dataset, sts_train_dataset]
    ]
    val_dataloaders = [
        DataLoader(
            x,
            shuffle=False,
            collate_fn=x.collate_fn,
            batch_size=config_dataloader['batch_size'],
            num_workers=config_dataloader['num_workers'],
        )
        for x in [sst_val_dataset, quora_val_dataset, sts_val_dataset]
    ]
    test_dataloaders = [
        DataLoader(
            x,
            shuffle=False,
            collate_fn=x.collate_fn,
            batch_size=config_dataloader['batch_size'],
            num_workers=config_dataloader['num_workers'],
        )
        for x in [sst_test_dataset, quora_test_dataset, sts_test_dataset]
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

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config_train['lr'])

    logger.info(f'Starting training the {config_bert["mode"]} BERT model on '
                f'all the tasks.')

    train_validation_loop_multitask(
        model=model,
        optimizer=optimizer,
        criterion=[nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.MSELoss()],
        metric=[accuracy, accuracy, pearson_correlation],
        train_loader=train_dataloaders,
        val_loader=val_dataloaders,
        n_epochs=config_train['n_epochs'],
        device=device,
        save_best_path=config_train['checkpoint_path'],
        overall_config=CONFIG,
        verbose=False,
    )

    load_state(model, device,  config_train['checkpoint_path'])

    logger.info(f'Starting testing the {config_bert["mode"]} BERT model on '
                f'all the tasks.')
    
    generate_predictions_multitask(model, device, test_dataloaders, config_prediction.values())
