import yaml
import os

from torch.utils.data import DataLoader
from torch import nn

import optuna
from optuna.samplers import TPESampler

from shutil import rmtree

from src.optim import AdamW
from src.models import MultitaskBERT
from src.core.unfreezer import BasicGradualUnfreezer
from src.metrics.regression_metrics import pearson_correlation_loss
from src.metrics import accuracy, pearson_correlation
from src.datasets import SSTDataset, SentenceSimilarityDataset
from src.core import train_one_epoch_multitask, evaluate_model_multitask
from src.utils import (logger, generate_device,
                       parse_hyperparameters_dict, seed_everything,
                       optuna_save_callback)


def main():

    # Clear the space
    if os.path.exists('temp'):
        logger.debug('Found the `temp` folder present, cleaning before start.')
        rmtree('temp')
        logger.debug('Successfully removed the `temp` folder.')

    # Load common config for all the models.
    common_path = 'config.yaml'
    with open(common_path, 'r') as f:
        COMMON_CONFIG = yaml.load(f, Loader=yaml.FullLoader)
    logger.debug(f'Common config successfully loaded from {common_path}.')

    config_sst = COMMON_CONFIG['data']['sst_dataset']
    config_quora = COMMON_CONFIG['data']['quora_dataset']
    config_sts = COMMON_CONFIG['data']['sts_dataset']

    seed_everything(COMMON_CONFIG['seed'])
    device = generate_device(COMMON_CONFIG['use_cuda'])

    # Create datasets
    sst_train_dataset = SSTDataset(
        config_sst['train_path'],
        return_targets=True
    )
    sst_val_dataset = SSTDataset(
        config_sst['val_path'],
        return_targets=True
    )

    quora_train_dataset = SentenceSimilarityDataset(
        config_quora['train_path'],
        return_targets=True
    )
    quora_val_dataset = SentenceSimilarityDataset(
        config_quora['val_path'],
        return_targets=True
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

    # Create dataloaders
    sst_train_dataloader = DataLoader(
        sst_train_dataset,
        shuffle=True,
        drop_last=True,
        collate_fn=sst_train_dataset.collate_fn,
        batch_size=COMMON_CONFIG['data']['dataloader']['batch_size'],
        num_workers=COMMON_CONFIG['data']['dataloader']['num_workers'],
    )
    sts_train_dataloader = DataLoader(
        sts_train_dataset,
        shuffle=True,
        drop_last=True,
        collate_fn=sts_train_dataset.collate_fn,
        batch_size=COMMON_CONFIG['data']['dataloader']['batch_size'],
        num_workers=COMMON_CONFIG['data']['dataloader']['num_workers'],
    )

    train_mode = COMMON_CONFIG['train']['train_mode']
    if train_mode == 'contrastive':
        exp_factor = COMMON_CONFIG['train'].get('exp_factor', 2)
        quora_train_dataloader = DataLoader(
            quora_train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=quora_train_dataset.collate_fn_contrastive(exp_factor),
            batch_size=COMMON_CONFIG['data']['dataloader']['batch_size'],
            num_workers=COMMON_CONFIG['data']['dataloader']['num_workers'],
        )

    elif train_mode == 'triplet':
        dropout_quora = COMMON_CONFIG['train']['triplet_dropout_rates']['quora']

        quora_train_dataloader = DataLoader(
            quora_train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=quora_train_dataset.collate_fn_triplet(dropout_quora),
            batch_size=COMMON_CONFIG['data']['dataloader']['batch_size'],
            num_workers=COMMON_CONFIG['data']['dataloader']['num_workers'],
        )

    else:
        quora_train_dataloader = DataLoader(
            quora_train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=quora_train_dataset.collate_fn,
            batch_size=COMMON_CONFIG['data']['dataloader']['batch_size'],
            num_workers=COMMON_CONFIG['data']['dataloader']['num_workers'],
        )

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
            batch_size=COMMON_CONFIG['data']['dataloader']['batch_size'],
            num_workers=COMMON_CONFIG['data']['dataloader']['num_workers'],
        )
        for x in [sst_val_dataset, quora_val_dataset, sts_val_dataset]
    ]

    weights = [x['weight'] for x in (config_sst, config_quora, config_sts)]

    # Hyperparameter config handling
    hyperparameter_config_path = 'configs/hyperparameter_config.yaml'
    with open(hyperparameter_config_path, 'r') as f:
        HYPERPARAMETER_CONFIG = yaml.load(f, Loader=yaml.FullLoader)
    try:
        model_config = HYPERPARAMETER_CONFIG['bert_model']
        train_config = HYPERPARAMETER_CONFIG['train']
    except KeyError as e:
        logger.error(f'ERROR: Error parsing the hyperparameter config: {e}')
        raise e
    logger.debug(f'Hyperparameter config successfully loaded from {hyperparameter_config_path}.')

    def objective(trial: optuna.Trial):
        # Changing all the model hyperparameters
        new_model_config = COMMON_CONFIG['bert_model'].copy()
        model_hyperparameters = parse_hyperparameters_dict(trial, model_config)
        for key, value in model_hyperparameters.items():
            new_model_config[key] = value
        new_model_config.pop('weights_path')
        model = MultitaskBERT(
            num_labels=5,
            **new_model_config
        )
        model = model.to(device)
        unfreezer = BasicGradualUnfreezer(model, layers_per_step=1, steps_to_hold=1)

        # Changing all the train hyperparameters
        new_train_config = COMMON_CONFIG['train'].copy()
        train_hyperparameters = parse_hyperparameters_dict(trial, train_config)
        for key, value in train_hyperparameters.items():
            new_train_config[key] = value

        if new_train_config['use_pearson_loss']:
            criteria = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), pearson_correlation_loss]
        else:
            criteria = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.MSELoss()]

        if new_train_config['add_cosine_loss']:
            cosine_loss = nn.CosineEmbeddingLoss(reduction='mean')
        else:
            cosine_loss = None

        train_mode = new_train_config['train_mode']
        optimizer = AdamW(model.parameters(), lr=new_train_config['lr'])

        epoch_train_state = None
        unfreezer.start()
        for i in range(new_train_config['n_epochs']):

            epoch_train_state = train_one_epoch_multitask(
                model=model,
                train_dataloaders=train_dataloaders,
                optimizer=optimizer,
                criterions=criteria,
                device=device,
                dataloader_mode=new_train_config['dataloader_mode'],
                train_mode=train_mode,
                verbose=True,
                current_epoch=i,
                weights=weights,
                prev_state=epoch_train_state,
                skip_optimizer_step=new_train_config['skip_optimizer_step'],
                cosine_loss=cosine_loss,
                overall_config=COMMON_CONFIG
            )
            logger.info(f'Finished training epoch {i}')

            unfreezer.step()

        val_metrics = evaluate_model_multitask(
            model,
            val_dataloaders,
            device,
            [accuracy, accuracy, pearson_correlation],
        )['metric']

        for key, value in val_metrics.items():
            trial.set_user_attr(key, value)

        trial_result = (val_metrics['sentiment']
                        + val_metrics['paraphrase_classifier']
                        + (val_metrics['paraphrase_regressor'] + 1) / 2)

        logger.info(f'Trial ended with resulting metric {trial_result}')

        return trial_result

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)

    study.optimize(
        objective,
        n_trials=HYPERPARAMETER_CONFIG['n_trials'],
        show_progress_bar=False,
        callbacks=[optuna_save_callback, ]
    )

    logger.info(f'All trials finished, best result is {study.best_trial.user_attrs}\n'
                f'achieved with {study.best_trial.params}')


if __name__ == '__main__':
    main()
