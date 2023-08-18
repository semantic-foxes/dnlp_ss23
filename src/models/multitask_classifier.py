import torch
from torch import nn
from src.models.bert import BertModel


class MultitaskBERT(nn.Module):
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()

        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", local_files_only=config.local_files_only
        )

        # Pretrain mode does not require updating bert parameters.
        for param in self.bert.parameters():
            if config.option == "pretrain":
                param.requires_grad = False
            elif config.option == "finetune":
                param.requires_grad = True

        self.sentiment_classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.paraphrase_classifier = nn.Linear(2 * config.hidden_size, 1)
        self.similarity_regressor = nn.Linear(2 * config.hidden_size, 1)

    def forward(
        self,
        input_ids_1: torch.Tensor,
        attention_mask_1: torch.Tensor,
        input_ids_2: torch.Tensor = None,
        attention_mask_2: torch.Tensor = None,
        task: str = "sentiment",
    ):
        if task == "sentiment":
            result = self.predict_sentiment(input_ids_1, attention_mask_1)

        elif task == "paraphrase_classifier":
            if input_ids_2 is None or attention_mask_2 is None:
                raise AttributeError
            result = self.predict_paraphrase(
                input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
            )

        elif task == "paraphrase_regressor":
            if input_ids_2 is None or attention_mask_2 is None:
                raise AttributeError
            result = self.predict_similarity(
                input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
            )

        else:
            raise AttributeError

        return result

    def predict_sentiment(self, input_ids, attention_mask):
        """Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        """
        bert_output = self.bert(input_ids, attention_mask)["pooler_output"]
        result = self.sentiment_classifier(bert_output)
        return result

    def predict_paraphrase(
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):
        """Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        """
        bert_output_1 = self.bert(input_ids_1, attention_mask_1)["pooler_output"]
        bert_output_2 = self.bert(input_ids_2, attention_mask_2)["pooler_output"]
        bert_output = torch.cat((bert_output_1, bert_output_2), dim=1)
        result = self.paraphrase_classifier(bert_output)
        return result

    def predict_similarity(
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):
        """Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        """
        bert_output_1 = self.bert(input_ids_1, attention_mask_1)["pooler_output"]
        bert_output_2 = self.bert(input_ids_2, attention_mask_2)["pooler_output"]
        bert_output = torch.cat((bert_output_1, bert_output_2), dim=1)
        result = self.similarity_regressor(bert_output)
        return result
