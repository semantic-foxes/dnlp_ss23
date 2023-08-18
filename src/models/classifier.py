import torch
from torch import nn

from src.models.bert import BertModel


class BertSentimentClassifier(torch.nn.Module):
    def __init__(self, config):
        super(BertSentimentClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained(
            'bert-base-uncased',
            local_files_only=config.local_files_only  # TODO: report bug
        )

        # Pretrain mode does not require updating bert parameters.
        if config.option == 'pretrain':
            for param in self.bert.parameters():
                param.requires_grad = False
        elif config.option == 'finetune':
            for param in self.bert.parameters():
                param.requires_grad = True
        else:
            raise AttributeError('Incorrect mode for BERT model. Should be'
                                 'either \'pretrain\' or \'finetune\'.')

        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask)['pooler_output']
        result = self.classifier(bert_output)
        return result

