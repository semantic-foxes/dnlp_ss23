import torch
from torch import nn

from src.models.bert import BertModel


class BertSentimentClassifier(torch.nn.Module):
    def __init__(
            self,
            num_labels,
            option: str = 'finetune',
            bert_model_name: str = 'bert-base-uncased',
            local_files_only: bool = False,
            vocab_size: int = 30522,
            type_vocab_size: int = 2,
            hidden_size: int = 768,
            intermediate_size: int = 3072,
            pad_token_id: int = 0,
            max_sequence_len: int = 512,
            initializer_range: float = 0.02,
            eps: float = 1e-12,
            hidden_dropout_prob: float = 0.1,
            attention_dropout_prob: float = 0.1,
            num_attention_heads: int = 12,
            num_bert_layers: int = 12
    ):
        super(BertSentimentClassifier, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(
            model_name=bert_model_name,
            local_files_only=local_files_only,
            vocab_size=vocab_size,
            type_vocab_size=type_vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            pad_token_id=pad_token_id,
            max_sequence_len=max_sequence_len,
            initializer_range=initializer_range,
            eps=eps,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_dropout_prob=attention_dropout_prob,
            num_attention_heads=num_attention_heads,
            num_bert_layers=num_bert_layers
        )

        # Pretrain mode does not require updating bert parameters.
        if option == 'pretrain':
            for param in self.bert.parameters():
                param.requires_grad = False
        elif option == 'finetune':
            for param in self.bert.parameters():
                param.requires_grad = True
        else:
            raise AttributeError('Incorrect mode for BERT model. Should be'
                                 'either \'pretrain\' or \'finetune\'.')

        self.classifier = nn.Linear(hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask)['pooler_output']
        result = self.classifier(bert_output)
        return result

