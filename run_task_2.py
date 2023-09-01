import argparse
import yaml
import pandas as pd
import wandb

from torch.utils.data import DataLoader
from torch import nn
from src.core.evaluation_multitask import evaluate_model_multitask
from src.core.pretrain_multitask import pretrain_validation_loop_multitask

from src.models import MultitaskBERT
from src.optim import AdamW
from src.datasets import SSTDataset, SentenceSimilarityDataset
from src.utils import seed_everything, generate_device, logger
from src.core import train_validation_loop_multitask, generate_predictions_multitask
from src.metrics import accuracy, pearson_correlation
from src.utils.model_utils import load_state, save_results

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config.yaml')
    parser.add_argument("--restore", action='store_true')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    with open(args.config, 'r') as f:
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

    # TODO: keep default values separately
    train_mode = config_train.get('train_mode', 'standard')
    if train_mode == 'contrastive':
        exp_factor = config_train.get('exp_factor', 2)
    if train_mode == 'triplet_unsupervised':
        triplet_dropout_rate = config_train.get('triplet_dropout_rate', 0.1)

    if CONFIG['watcher']['type'] == 'wandb':
        wandb.init(
            project=CONFIG['watcher']['project_name'],
            config=CONFIG,
            mode=CONFIG['watcher']['mode'],
        )
        watcher = 'wandb'

    elif CONFIG['watcher']['type'] == 'none':
        watcher = None

    else:
        message = 'ERROR: Unsupported watcher selected.'
        logger.error(message)
        raise NotImplementedError(message)

    # Create datasets
    sst_train_dataset = SSTDataset(
        config_sst['train_path'],
        return_targets=True,
        nrows=config_train.get('max_train_size'),
    )
    sst_val_dataset = SSTDataset(
        config_sst['val_path'],
        return_targets=True,
        nrows=config_train.get('max_eval_size')
    )
    sst_test_dataset = SSTDataset(
        config_sst['test_path'],
        return_targets=False,
        nrows=config_train.get('max_eval_size')
    )

    quora_train_dataset = SentenceSimilarityDataset(
        config_quora['train_path'],
        return_targets=True,
        nrows=config_train.get('max_train_size'),
    )
    quora_val_dataset = SentenceSimilarityDataset(
        config_quora['val_path'],
        return_targets=True,
        nrows=config_train.get('max_eval_size')
    )
    quora_test_dataset = SentenceSimilarityDataset(
        config_quora['test_path'],
        return_targets=False,
        index_col=False,
        nrows=config_train.get('max_eval_size')
    )

    sts_train_dataset = SentenceSimilarityDataset(
        config_sts['train_path'],
        binary_task=False,
        return_targets=True,
        nrows=config_train.get('max_train_size'),
    )
    sts_val_dataset = SentenceSimilarityDataset(
        config_sts['val_path'],
        binary_task=False,
        return_targets=True,
        nrows=config_train.get('max_eval_size')
    )
    sts_test_dataset = SentenceSimilarityDataset(
        config_sts['test_path'],
        binary_task=False,
        return_targets=False,
        nrows=config_train.get('max_eval_size')
    )

    # Create dataloaders
    sst_train_dataloader = DataLoader(
        sst_train_dataset,
        shuffle=True,
        drop_last=True,
        collate_fn=sst_train_dataset.collate_fn,
        batch_size=config_dataloader['batch_size'],
        num_workers=config_dataloader['num_workers'],
    )
    
    train_eval_dataloaders = [sst_train_dataloader] + [
        DataLoader(
            x,
            shuffle=True,
            drop_last=True,
            collate_fn=x.collate_fn,
            batch_size=config_dataloader['batch_size'],
            num_workers=config_dataloader['num_workers'],
        )
        for x in [quora_train_dataset, sts_train_dataset]
    ]

    if train_mode == 'contrastive':
        train_dataloaders = [sst_train_dataloader] + [
            DataLoader(
                x,
                shuffle=True,
                drop_last=True,
                collate_fn=x.collate_fn_contrastive(exp_factor),
                batch_size=config_dataloader['batch_size'],
                num_workers=config_dataloader['num_workers'],
            )
            for x in [quora_train_dataset, sts_train_dataset]
        ]
    elif train_mode == 'triplet_unsupervised':
        train_dataloaders = [sst_train_dataloader] + [
            DataLoader(
                x,
                shuffle=True,
                drop_last=True,
                collate_fn=x.collate_fn_triplet_unsupervised(triplet_dropout_rate),
                batch_size=config_dataloader['batch_size'],
                num_workers=config_dataloader['num_workers'],
            )
            for x in [quora_train_dataset, sts_train_dataset]
        ]
    else:
        train_dataloaders = train_eval_dataloaders    

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
        # the order of datasets must match the order in config.yaml (predictions save_path)
    ]

    model = MultitaskBERT(
        num_labels=5,
        bert_mode=config_bert['bert_mode'],
        local_files_only=config_bert['local_files_only'],
        hidden_size=config_bert['hidden_size'],
        hidden_dropout_prob=config_bert['hidden_dropout_prob'],
        attention_dropout_prob=config_bert['attention_dropout_prob'],
    )

    model = model.to(device)

    metrics = [accuracy, accuracy, pearson_correlation]
    criteria = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.MSELoss()]

    best_metric = {}
    if args.restore:
        load_state(model, device, config_bert['weights_path'])
        best_scores = evaluate_model_multitask(model, val_dataloaders, device, metrics, criteria)
        best_metric = best_scores['metric']

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config_train['lr'])

    logger.info(f'Starting training the {config_bert["bert_mode"]} BERT model on '
                f'all the tasks.')

    if config_bert['bert_mode'] == 'pretrain':
        train_function = pretrain_validation_loop_multitask
    else:
        train_function = train_validation_loop_multitask

    train_function(
        model=model,
        optimizer=optimizer,
        criterion=criteria,
        metric=metrics,
        train_loader=train_dataloaders,
        train_eval_loader=train_eval_dataloaders,
        val_loader=val_dataloaders,
        n_epochs=config_train['n_epochs'],
        device=device,
        save_best_path=config_train['checkpoint_path'],
        overall_config=CONFIG,
        dataloader_mode=config_train['dataloader_mode'],
        train_mode=train_mode,
        weights=[1, 10, 1],
        verbose=False,
        watcher=watcher,
        skip_train_eval=config_train['skip_train_eval'],
        best_metric=best_metric
    )

    load_state(model, device, config_train['checkpoint_path'])

    logger.info(f'Starting testing the {config_bert["bert_mode"]} BERT model on '
                f'all the tasks.')
    
    evaluate_model_multitask(model, val_dataloaders, device, metrics, criteria)
    
    generate_predictions_multitask(model, device, test_dataloaders, config_prediction.values())
