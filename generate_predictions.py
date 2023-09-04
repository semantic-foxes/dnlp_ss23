import argparse
import yaml
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader

from src.models import MultitaskBERT
from src.datasets import SSTDataset, SentenceSimilarityDataset
from src.utils import seed_everything, generate_device, logger
from src.core import generate_predictions_multitask
from src.utils.model_utils import load_state


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config.yaml')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    with open(args.config, 'r') as f:
        CONFIG = yaml.load(f, Loader=yaml.FullLoader)

    config_sst = CONFIG['data']['sst_dataset']
    config_quora = CONFIG['data']['quora_dataset']
    config_sts = CONFIG['data']['sts_dataset']
    config_dataloader = CONFIG['data']['dataloader']
    config_prediction = CONFIG['predict']
    config_bert = CONFIG['bert_model']

    seed_everything(CONFIG['seed'])
    device = generate_device(CONFIG['use_cuda'])

    sst_val_dataset = SSTDataset(
        config_sst['val_path'],
        return_targets=False,
    )
    sst_test_dataset = SSTDataset(
        config_sst['test_path'],
        return_targets=False,
    )

    quora_val_dataset = SentenceSimilarityDataset(
        config_quora['val_path'],
        return_targets=False,
    )
    quora_test_dataset = SentenceSimilarityDataset(
        config_quora['test_path'],
        return_targets=False,
        index_col=False,
    )

    sts_val_dataset = SentenceSimilarityDataset(
        config_sts['val_path'],
        binary_task=False,
        return_targets=False,
    )
    sts_test_dataset = SentenceSimilarityDataset(
        config_sts['test_path'],
        binary_task=False,
        return_targets=False,
    )

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

    logger.info('Loading the model to evaluate')
    model = MultitaskBERT(
        num_labels=5,
        bert_mode=config_bert['bert_mode'],
        local_files_only=config_bert['local_files_only'],
        hidden_size=config_bert['hidden_size'],
        hidden_dropout_prob=config_bert['hidden_dropout_prob'],
        attention_dropout_prob=config_bert['attention_dropout_prob'],
        use_pearson_loss=False
    )
    model = model.to(device)
    load_state(model, device, CONFIG['train']['checkpoint_path'])

    predict_names = ['Predicted_Sentiment', 'Predicted_Is_Paraphrase',
                     'Predicted_Similarity']

    generate_predictions_multitask(
        model=model,
        device=device,
        dataloaders=val_dataloaders,
        predict_column_names=predict_names,
        filepaths=config_prediction['val'].values()
    )

    generate_predictions_multitask(
        model=model,
        device=device,
        dataloaders=test_dataloaders,
        predict_column_names=predict_names,
        filepaths=config_prediction['test'].values()
    )

    # Handling None-s
    ## Val
    predict_path = config_prediction['val']['quora_prediction_path']

    predicted = pd.read_csv(predict_path, sep=', ')
    original = pd.read_csv(config_quora['val_path'], sep='\t')
    missing = set(original['id']) - set(predicted['id'])
    for id in missing:
        line = {'id': id, 'Predicted_Is_Paraphrase': 0}
        print(line)
        predicted = predicted.append(line, ignore_index=True)

    np.savetxt(
        predict_path,
        predicted,
        delimiter=', ',
        header=', '.join(predicted.columns.values),
        fmt='%s',
        comments='',
        encoding=None
    )


    ## Test
    predict_path = config_prediction['test']['quora_prediction_path']

    predicted = pd.read_csv(predict_path, sep=', ')
    original = pd.read_csv(config_quora['test_path'], sep='\t')
    missing = set(original['id']) - set(predicted['id'])
    for id in missing:
        line = {'id': id, 'Predicted_Is_Paraphrase': 0}
        print(line)
        predicted = predicted.append(line, ignore_index=True)

    np.savetxt(
        predict_path,
        predicted,
        delimiter=', ',
        header=', '.join(predicted.columns.values),
        fmt='%s',
        comments='',
        encoding=None
    )

