from types import SimpleNamespace
import yaml

from sklearn.metrics import accuracy_score

from torch import nn
from torch.utils.data import DataLoader

from src.core import train_validation_loop
from src.utils import seed_everything, logger, generate_device
from src.models import BertSentimentClassifier
from src.optim import AdamW
from src.datasets import SSTDataset


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        CONFIG = yaml.load(f, Loader=yaml.FullLoader)

    config_data = CONFIG['data']
    config_bert = CONFIG['bert_model']
    config_train = CONFIG['train']

    seed_everything(CONFIG['seed'])
    device = generate_device(CONFIG['use_cuda'])

    train_dataset = SSTDataset(
        config_data['sst_dataset']['train_path'],
        return_targets=True
    )
    val_dataset = SSTDataset(
        config_data['sst_dataset']['val_path'],
        return_targets=True
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        batch_size=config_data['dataloader']['batch_size'],
        num_workers=config_data['dataloader']['num_workers'],
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        batch_size=config_data['dataloader']['batch_size'],
        num_workers=config_data['dataloader']['num_workers'],
    )

    # Create a model
    model = BertSentimentClassifier(
        num_labels=5,
        bert_mode=config_bert['bert_mode'],
        local_files_only=config_bert['local_files_only'],
        hidden_size=config_bert['hidden_size'],
        hidden_dropout_prob=config_bert['dropout_prob'],
        attention_dropout_prob=config_bert['dropout_prob'],
    )
    model = model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config_train['lr'])

    logger.info(f'Starting training the {config_bert["mode"]} BERT model.')

    train_validation_loop(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        metric=accuracy_score,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        n_epochs=config_train['n_epochs'],
        device=device,
        save_best_path=config_train['checkpoint_path'],
        overall_config=CONFIG,
        verbose=False
    )
