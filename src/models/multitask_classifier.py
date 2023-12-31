import torch
from torch import nn
from src.models.bert import BertModel

class MultitaskBERT(nn.Module):
    def __init__(
            self,
            num_labels,
            bert_mode: str = 'finetune',
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
            num_bert_layers: int = 12,
            use_pearson_loss: bool = False,
    ):
        super(MultitaskBERT, self).__init__()

        self.num_labels = num_labels
        self.use_pearson_loss = use_pearson_loss
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
        if bert_mode == 'pretrain':
            self.requires_grad_(False)
        elif bert_mode == 'finetune':
            self.requires_grad_(True)
        else:
            raise AttributeError('Incorrect mode for BERT model. Should be'
                                 'either \'pretrain\' or \'finetune\'.')


        self.sentiment_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, self.num_labels),
        )
        self.paraphrase_classifier = nn.Sequential(
            nn.Linear(2*hidden_size, 2)
        )
        self.paraphrase_regressor_1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
        )
        self.paraphrase_regressor_2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
        )
        self.paraphrase_decision = nn.CosineSimilarity()

    def forward(
            self,
            task: str,
            input_ids_1: torch.Tensor,
            attention_mask_1: torch.Tensor,
            input_ids_2: torch.Tensor = None,
            attention_mask_2: torch.Tensor = None,
            return_embeddings: bool = False,
    ):
        if task == 'embed':
            result = self.embed(input_ids_1, attention_mask_1)

        elif task == 'sentiment':
            result = self.predict_sentiment(input_ids_1, attention_mask_1)

        elif task == 'paraphrase_classifier':
            if input_ids_2 is None or attention_mask_2 is None:
                raise AttributeError
            
            result, *embeddings = self.predict_paraphrase(input_ids_1, attention_mask_1,
                                             input_ids_2, attention_mask_2)
            if return_embeddings:
                return result, embeddings


        elif task == 'paraphrase_regressor':
            if input_ids_2 is None or attention_mask_2 is None:
                raise AttributeError
            result = self.predict_similarity(input_ids_1, attention_mask_1,
                                             input_ids_2, attention_mask_2)

        else:
            raise AttributeError

        return result

    def embed(self, input_ids, attention_mask):
        result = self.bert(input_ids, attention_mask)['pooler_output']
        return result

    def predict_sentiment(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask)['pooler_output']
        result = self.sentiment_classifier(bert_output)
        return result

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        bert_output_1 = self.bert(input_ids_1, attention_mask_1)['pooler_output']
        bert_output_2 = self.bert(input_ids_2, attention_mask_2)['pooler_output']
        bert_output = torch.cat((bert_output_1, bert_output_2), dim=1)
        result = self.paraphrase_classifier(bert_output)
        return result, bert_output_1, bert_output_2

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        bert_output_1 = self.bert(input_ids_1, attention_mask_1)['pooler_output']
        bert_output_2 = self.bert(input_ids_2, attention_mask_2)['pooler_output']
        embedding_processed_1 = self.paraphrase_regressor_1(bert_output_1)
        embedding_processed_2 = self.paraphrase_regressor_2(bert_output_2)
        # Since the target is 0-5 in this task as well :/
        # Though the result is in [-5,5], we found out it to produce better outcome
        result = self.paraphrase_decision(embedding_processed_1, embedding_processed_2) * 5

        if not self.training and not self.use_pearson_loss:
            # projection usually decreases error for MSE
            # also it maps predictions to the target interval [0,5]
            result = torch.clip(result, 0, 5)

        return result.flatten()
