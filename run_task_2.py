import argparse
import yaml
import wandb

from torch.utils.data import DataLoader
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

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
    config_pretrain = CONFIG['pretrain']
    config_post_train = CONFIG['post-train']

    seed_everything(CONFIG['seed'])
    device = generate_device(CONFIG['use_cuda'])

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

    # Create train dataloaders
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

    train_eval_dataloaders = [
        sst_train_dataloader,
        quora_train_eval_dataloader,
        sts_train_dataloader
    ]

    # Special train modes require a specific collate function.
    train_mode = config_train['train_mode']
    if train_mode == 'contrastive':
        exp_factor = config_train.get('exp_factor', 2)
        quora_train_dataloader = DataLoader(
                quora_train_dataset,
                shuffle=True,
                drop_last=True,
                collate_fn=quora_train_dataset.collate_fn_contrastive(exp_factor),
                batch_size=config_dataloader['batch_size'],
                num_workers=config_dataloader['num_workers'],
            )

    elif train_mode == 'triplet':
        dropout_quora = config_train['triplet_dropout_rates']['quora']

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

    logger.info('Creating the model')
    model = MultitaskBERT(
        num_labels=5,
        bert_mode=config_bert['bert_mode'],
        local_files_only=config_bert['local_files_only'],
        hidden_size=config_bert['hidden_size'],
        hidden_dropout_prob=config_bert['hidden_dropout_prob'],
        attention_dropout_prob=config_bert['attention_dropout_prob'],
        use_pearson_loss=config_train['use_pearson_loss'],
    )
    model = model.to(device)
    unfreezer = BasicGradualUnfreezer(model, layers_per_step=1, steps_to_hold=1)

    metrics = [accuracy, accuracy, pearson_correlation]

    if config_train['use_pearson_loss']:
        criteria = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), pearson_correlation_loss]
    else:
        criteria = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.MSELoss()]

    if config_train['add_cosine_loss']:
        cosine_loss = nn.CosineEmbeddingLoss(reduction='mean')
    else:
        cosine_loss = None

    best_metric = {}
    if args.restore:
        load_state(model, device, config_bert['weights_path'])
        best_scores = evaluate_model_multitask(
            model=model,
            eval_dataloaders=val_dataloaders,
            device=device,
            metrics=metrics,
            criterions=criteria,
            cosine_loss=cosine_loss,
            verbose=args.silent,
            set_name='val',
        )
        best_metric = best_scores['metric']

    weights = [x['weight'] for x in (config_sst, config_quora, config_sts)]

    # Pretrain
    if config_pretrain['n_epochs'] > 0:
        model.bert.requires_grad_(False)
        optimizer_pre = AdamW(model.parameters(), lr=config_pretrain['lr'])

        logger.info(f'Starting *pre*train on all the tasks.')
        _, best_metric = pretrain_validation_loop_multitask(
            model=model,
            optimizer=optimizer_pre,
            criterion=criteria,
            metric=metrics,
            train_loader=train_dataloaders,
            train_eval_loader=train_eval_dataloaders,
            val_loader=val_dataloaders,
            n_epochs=config_pretrain['n_epochs'],
            device=device,
            watcher=watcher,
            verbose=args.silent,
            weights=weights,
            save_best_path=config_train['checkpoint_path'],
            overall_config=CONFIG,
            dataloader_mode=config_pretrain['dataloader_mode'],
            train_mode=train_mode,
            skip_train_eval=config_train['skip_train_eval'],
            best_metric=best_metric,
            skip_optimizer_step=config_pretrain['skip_optimizer_step'],
            cosine_loss=None,
        )
        load_state(model, device, config_train['checkpoint_path'])

    logger.info(f'Starting training the {config_bert["bert_mode"]} BERT model'
                f'in {train_mode} mode on all the tasks.')

    model.bert.requires_grad_(True)
    optimizer = AdamW(model.parameters(), lr=config_train['lr'])
    scheduler = ExponentialLR(
        optimizer,
        gamma=1.2
    )
    _, best_metric = train_validation_loop_multitask(
        model=model,
        optimizer=optimizer,
        criterion=criteria,
        metric=metrics,
        train_loader=train_dataloaders,
        train_eval_loader=train_eval_dataloaders,
        val_loader=val_dataloaders,
        n_epochs=config_train['n_epochs'],
        device=device,
        unfreezer=unfreezer,
        scheduler=scheduler,
        watcher=watcher,
        verbose=args.silent,
        weights=weights,
        save_best_path=config_train['checkpoint_path'],
        overall_config=CONFIG,
        dataloader_mode=config_train['dataloader_mode'],
        train_mode=train_mode,
        skip_train_eval=config_train['skip_train_eval'],
        best_metric=best_metric,
        skip_optimizer_step=config_train['skip_optimizer_step'],
        cosine_loss=cosine_loss,
    )

    # Post-train
    if config_post_train['n_epochs'] > 0:

        model.bert.requires_grad_(False)
        optimizer_post = AdamW(model.parameters(), lr=config_post_train['lr'])
        load_state(model, device, config_train['checkpoint_path'])

        logger.info(f'Starting *post*-train on all the tasks.')
        _, best_metric = pretrain_validation_loop_multitask(
            model=model,
            optimizer=optimizer_post,
            criterion=criteria,
            metric=metrics,
            train_loader=train_dataloaders,
            train_eval_loader=train_eval_dataloaders,
            val_loader=val_dataloaders,
            n_epochs=config_post_train['n_epochs'],
            device=device,
            watcher=watcher,
            verbose=args.silent,
            weights=weights,
            save_best_path=config_train['checkpoint_path'],
            overall_config=CONFIG,
            dataloader_mode=config_post_train['dataloader_mode'],
            train_mode=train_mode,
            skip_train_eval=config_train['skip_train_eval'],
            best_metric=best_metric,
            skip_optimizer_step=config_post_train['skip_optimizer_step'],
            cosine_loss=None,
        )

    load_state(model, device, config_train['checkpoint_path'])

    logger.info(f'Starting testing the {config_bert["bert_mode"]} BERT model on '
                f'all the tasks.')
    
    evaluate_model_multitask(
        model=model,
        eval_dataloaders=val_dataloaders,
        device=device,
        metrics=metrics,
        criterions=criteria,
        cosine_loss=cosine_loss,
        verbose=args.silent,
        set_name='val',
    )
    
    generate_predictions_multitask(
        model=model,
        device=device,
        dataloaders=test_dataloaders,
        filepaths=config_prediction.values()
    )
