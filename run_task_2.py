import argparse
import yaml
import wandb

from torch.utils.data import DataLoader
from torch import nn
from src.core.evaluation_multitask import evaluate_model_multitask
from src.core.pretrain_multitask import pretrain_validation_loop_multitask
from src.metrics.regression_metrics import pearson_correlation_loss

from src.models import MultitaskBERT
from src.optim import AdamW
from src.datasets import SSTDataset, SentenceSimilarityDataset
from src.utils import seed_everything, generate_device, logger
from src.core import train_validation_loop_multitask, generate_predictions_multitask
from src.metrics import accuracy, pearson_correlation
from src.utils.model_utils import load_state
from src.core.unfreezer import BasicGradualUnfreezer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config.yaml')
    parser.add_argument("--restore", action='store_true')
    parser.add_argument("--id", type=str, default='')
    parser.add_argument("--silent", action='store_false')

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

    skip_optimizer_step = config_train.get('skip_optimizer_step', 1)

    seed_everything(CONFIG['seed'])
    device = generate_device(CONFIG['use_cuda'])

    # TODO: keep default values separately
    train_mode = config_train.get('train_mode', 'standard')
    if train_mode == 'contrastive':
        exp_factor = config_train.get('exp_factor', 2)
    elif train_mode == 'triplet':
        triplet_dropout_rates = config_train.get(
            'triplet_dropout_rates', {})
        dropout_sts = triplet_dropout_rates.get('sts', 0.2)
        dropout_quora =triplet_dropout_rates.get('quora', 0.05)

    if CONFIG['watcher']['type'] == 'wandb':
        wandb.init(
            project=CONFIG['watcher']['project_name'],
            config=CONFIG,
            mode=CONFIG['watcher']['mode'],
            resume='must' if args.restore else args.restore,
            id=args.id if args.id else None,
        )
        watcher = 'wandb'

    elif CONFIG['watcher']['type'] is None:
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
    sts_train_dataloader = DataLoader(
        sts_train_dataset,
        shuffle=True,
        drop_last=True,
        collate_fn=sts_train_dataset.collate_fn,
        batch_size=config_dataloader['batch_size'],
        num_workers=config_dataloader['num_workers'],
    )
    quora_train_eval_dataloader = DataLoader(
        quora_train_dataset,
        shuffle=True,
        drop_last=True,
        collate_fn=quora_train_dataset.collate_fn,
        batch_size=config_dataloader['batch_size'],
        num_workers=config_dataloader['num_workers'],
    )
    if train_mode == 'contrastive':
        quora_train_dataloader = DataLoader(
            quora_train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=quora_train_dataset.collate_fn_contrastive(exp_factor),
            batch_size=config_dataloader['batch_size'],
            num_workers=config_dataloader['num_workers'],
        )
    elif train_mode == 'triplet':
        quora_train_dataloader = DataLoader(
            quora_train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=quora_train_dataset.collate_fn_triplet(dropout_quora),
            batch_size=config_dataloader['batch_size'],
            num_workers=config_dataloader['num_workers'],
        )
    else:
        quora_train_dataloader = quora_train_eval_dataloader

    train_eval_dataloaders = [
        sst_train_dataloader,
        quora_train_eval_dataloader,
        sts_train_dataloader
    ]

    train_dataloaders = [
        sst_train_dataloader,
        quora_train_dataloader,
        sts_train_dataloader
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
        # the order of datasets must match the order in config.yaml (predictions save_path)
    ]

    weights = [1, 10, 1]
    if config_quora.get('weight'):
        weights = [1, config_quora.get('weight'), 1]
    if config_train.get('dataloader_mode') == 'exhaust':
        weights = [2, 20, 1]

    logger.info('Create Model')
    model = MultitaskBERT(
        num_labels=5,
        bert_mode=config_bert['bert_mode'],
        local_files_only=config_bert['local_files_only'],
        hidden_size=config_bert['hidden_size'],
        hidden_dropout_prob=config_bert['hidden_dropout_prob'],
        attention_dropout_prob=config_bert['attention_dropout_prob'],
    )

    logger.info('Model to device')
    model = model.to(device)

    metrics = [accuracy, accuracy, pearson_correlation]
    criteria = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.MSELoss()]

    if CONFIG.get('train', {}).get('use_pearson_loss'):
        criteria = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), pearson_correlation_loss]

    cosine_loss = None
    if config_train.get('add_cosine_loss'):
        cosine_loss = nn.CosineEmbeddingLoss(reduction='mean')

    best_metric = {}
    if args.restore:
        load_state(model, device, config_bert['weights_path'])
        best_scores = evaluate_model_multitask(model, val_dataloaders, device, metrics, criteria, cosine_loss)
        best_metric = best_scores['metric']

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config_train['lr'])
    config_unfreeze=config_train.get('unfreezer', False)
    if config_unfreeze:
        unfreezer = BasicGradualUnfreezer(model, layers_per_step=1, steps_to_hold=1)
    else:
        unfreezer = None

    default_args = {
        'model': model,
        'optimizer': optimizer,
        'criterion': criteria,
        'metric': metrics,
        'train_loader': train_dataloaders,
        'val_loader': val_dataloaders,
        'n_epochs': config_train['n_epochs'],
        'device': device,
        'save_best_path': config_train['checkpoint_path'],
        'overall_config': CONFIG,
        'unfreezer': unfreezer,
        'dataloader_mode': config_train['dataloader_mode'],
        'weights': weights,
        'verbose': args.silent,
        'watcher': CONFIG['watcher']['type'],
        'skip_train_eval': config_train['skip_train_eval'],
        'best_metric': best_metric,
        'skip_optimizer_step': skip_optimizer_step,
        'cosine_loss': cosine_loss,
        'train_mode': train_mode,
        'train_eval_loader': train_eval_dataloaders,
    }
    # Pre train
    config_pre_train = CONFIG.get('pre_train', {})
    if config_pre_train.get('n_epochs', 0):
        logger.info(f'Starting PRE train on all the tasks.')
        model.freeze_bert(True)
        optimizer_pre = AdamW(model.parameters(), lr=config_pre_train['lr'])
        _, best_metric = pretrain_validation_loop_multitask(
            **{
                **default_args,
                'optimizer': optimizer_pre,
                'n_epochs': 1,
                'dataloader_mode': config_pre_train['dataloader_mode'],
                'weights': [1, 1, 1],
                'best_metric': best_metric,
                'skip_optimizer_step': config_pre_train.get('skip_optimizer_step', 1),
                'cosine_loss': None,
            }
        )
        load_state(model, device, config_train['checkpoint_path'])

    logger.info(f'Starting training the {config_bert["bert_mode"]} BERT model'
                f'in {train_mode} mode on all the tasks.')

    _, best_metric = train_validation_loop_multitask(**{**default_args, 'best_metric': best_metric})

    # Post train
    config_post_train = CONFIG.get('post_train', {})
    if config_post_train.get('n_epochs', 0):
        logger.info(f'Starting POST train on all the tasks.')
        load_state(model, device, config_train['checkpoint_path'])
        model.freeze_bert(True)
        optimizer_post = AdamW(model.parameters(), lr=config_post_train['lr'])
        _, best_metric = pretrain_validation_loop_multitask(
            **{
                **default_args,
                'optimizer': optimizer_post,
                'n_epochs': config_post_train['n_epochs'],
                'dataloader_mode': config_post_train['dataloader_mode'],
                'weights': [1, 1, 1],
                'best_metric': best_metric,
                'skip_optimizer_step': config_post_train.get('skip_optimizer_step', 1),
                'cosine_loss': None,
            }
        )

    load_state(model, device, config_train['checkpoint_path'])

    logger.info(f'Starting testing the {config_bert["bert_mode"]} BERT model on '
                f'all the tasks.')
    
    evaluate_model_multitask(model, val_dataloaders, device, metrics, criteria, cosine_loss=cosine_loss, overall_config=CONFIG, verbose=args.silent)
    
    generate_predictions_multitask(model, device, test_dataloaders, filepaths=config_prediction.values(), overall_config=CONFIG)
