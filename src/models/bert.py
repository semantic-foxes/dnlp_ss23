import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_bert import BertPreTrainedModel
from src.utils.utils import *


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Initialize the linear transformation layers for key, value, query
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # This dropout is applied to normalized attention scores following the
        # original implementation of transformer. Although it is a bit unusual,
        # we empirically observe that it yields better performance
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(
            self,
            x: torch.Tensor,
            linear_layer: nn.Module
    ) -> torch.Tensor:
        """
        Projects the input "x" using the provided linear layer. Splits the
        result into the required number of heads of the desired size (both
        defined at __init__).

        Parameters
        ----------
        x : torch.Tensor
            The tensor to project. It is supposed to have shape of
            `[batch_size, sequence_len, hidden_size]`.

        linear_layer : nn.Module
            The linear layer used to project the input.

        Returns
        -------
        proj : torch.Tensor
            The resulting projection split into the required number of heads.
            The resulting shape is
            `[batch_size, num_attention_heads, sequence_len, attention_head_size]`
            so it is **different** from the original one.

        Notes
        -------
        Used to project the hidden state to the key, value and query using the
        corresponding linear layers from attributes.

        """
        batch_size, sequence_len = x.shape[:2]
        proj = linear_layer(x)

        # Split into the required shape.
        proj = proj.view(
            batch_size,
            sequence_len,
            self.num_attention_heads,
            self.attention_head_size
        )

        # Transpose to the stated order of [batch_size, num_attention_heads,
        # sequence_len, attention_head_size] for the future ease.
        proj = proj.transpose(1, 2)

        return proj

    def attention(
            self,
            key: torch.Tensor,
            query: torch.Tensor,
            value: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Implementation of the self-attention mechanism.

        Parameters
        ----------
        key : torch.Tensor
            Key tensor.

        query : torch.Tensor
            Query tensor.

        value : torch.Tensor
            Value tensor.

        attention_mask : torch.Tensor
            A binary mask showing which tokens are padding ones
            and which are "real". Should be "0" for a "real" token
            and a big negative number for a padding token.

        Returns
        -------
        result : torch.Tensor

        Notes
        -------
        The shape of each of the `key`, `query` and `value` tensors is supposed
        to be `[batch_size, num_attention_heads, sequence_len, attention_head_size]`.
        """

        attention = torch.matmul(query, key.transpose(2, 3))

        # We do not want any attention for the padded tokens. Since
        # the mask is 0 for a "real" token and a large negative number for
        # another one, we do this. However, it appears to me to be incorrect:
        # I believe that instead of using a simple "sum" we should instead
        # _replace_ with a number.

        attention += attention_mask
        attention /= (self.attention_head_size ** 0.5)

        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        result = torch.matmul(attention, value)
        result = result.transpose(1, 2)
        result = result.reshape(query.shape[0], query.shape[2], -1)

        return result

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Feedforward of a Bert layer.

        Parameters
        ----------
        hidden_states : torch.Tensor,
            The input hidden states values to transform. The shape is expected
            to be `[batch_size, sequence_len, hidden_state_len]`.

        attention_mask : torch.Tensor
            A binary mask showing which tokens are padding ones
            and which are "real". Should be "0" for a "real" token
            and a big negative number for a padding token.

        Returns
        -------
        attention : torch.Tensor
            The resulting attention tensor, with shape
            `[batch_size, sequence_len, hidden_state_len]`
        """
        key = self.transform(hidden_states, self.key)
        value = self.transform(hidden_states, self.value)
        query = self.transform(hidden_states, self.query)

        # calculate the multi-head attention
        attention = self.attention(key, query, value, attention_mask)
        return attention


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # multi-head attention
        self.self_attention = BertSelfAttention(config)

        # add-norm
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)

        # feed forward
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu

        # another add-norm
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    @staticmethod
    def add_norm(
            previous_layer_input: torch.Tensor,
            previous_layer_output: torch.Tensor,
            dense_layer: nn.Module,
            dropout_layer: nn.Module,
            layer_norm_layer: nn.Module
    ):
        """
        An add and normalize operation used in BERT implementation.

        Parameters
        ----------
        previous_layer_input : torch.Tensor
            The input the previous layer got. Either the input for the multihead
            attention, or the input for the fully-connected network on top of it.

        previous_layer_output : torch.Tensor
            The result of applying the previous layer.

        dense_layer : torch.Module
            Dense layer used.

        dropout_layer : torch.Module
            Dropout layer used.

        layer_norm_layer : torch.Module
            Layer normalization layer used.

        Returns
        -------
        result : torch.Tensor
            The result of the add_norm layer.
        """
        x = dense_layer(previous_layer_output)
        x = dropout_layer(x)

        # residual connection
        x += previous_layer_input

        result = layer_norm_layer(x)
        return result

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor
    ):
        """
        An implementation of a forward pass of a single BERT layer.

        Parameters
        ----------
        hidden_states : torch.Tensor
            The input for the layer. Either the input embeddings for the
            first BERT layer or the output of the previous BERT layer.

        attention_mask : torch.Tensor
            A binary mask showing which tokens are padding ones
            and which are "real". Should be "0" for a "real" token
            and a big negative number for a padding token.

        Returns
        -------
        result : torch.Tensor
            The processed input for the layer.

        Notes
        -------
        Each BERT layer includes 4 steps:

        - A multi-head attention layer (BertSelfAttention);
        - An add-norm that takes the input and output of the multi-head attention layer;
        - A feed forward (dense) layer;
        - An add-norm that takes the input and output of the feed forward layer.

        The implementation is guided by the Figure 1 of the original paper
        found at https://arxiv.org/pdf/1706.03762.pdf
        """

        attention = self.self_attention(hidden_states, attention_mask)

        normalized_attention = self.add_norm(
            hidden_states,
            attention,
            self.attention_dense,
            self.attention_dropout,
            self.attention_layer_norm
        )

        processed_attention = self.interm_dense(normalized_attention)
        processed_attention = self.interm_af(processed_attention)

        result = self.add_norm(
            normalized_attention,
            processed_attention,
            self.out_dense,
            self.out_dropout,
            self.out_layer_norm
        )

        return result

class BertModel(BertPreTrainedModel):
    """
    the bert model returns the final embeddings for each token in a sentence
    it consists
    1. embedding (used in self.embed)
    2. a stack of n bert layers (used in self.encode)
    3. a linear transformation layer for [CLS] token (used in self.forward, as given)
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # embedding
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is a constant, register to buffer
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)

        # bert encoder
        self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

        # for [CLS] token
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()

        self.init_weights()

    def embed(self, input_ids):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        # Get word embedding from self.word_embedding into input_embeds.
        inputs_embeds = self.word_embedding(input_ids)

        # Get position index and position embedding from self.pos_embedding into pos_embeds.
        pos_ids = self.position_ids[:, :seq_length]

        pos_embeds = self.pos_embedding(pos_ids)

        # Get token type ids, since we are not consider token type, just a placeholder.
        tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        tk_type_embeds = self.tk_type_embedding(tk_type_ids)

        # Add three embeddings together; then apply embed_layer_norm and dropout and return.
        return self.embed_dropout(self.embed_layer_norm(tk_type_embeds + pos_embeds + inputs_embeds))

    def encode(self, hidden_states, attention_mask):
        """
        hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
        """
        # get the extended attention mask for self attention
        # returns extended_attention_mask of [batch_size, 1, 1, seq_len]
        # non-padding tokens with 0 and padding tokens with a large negative number
        extended_attention_mask = get_extended_attention_mask(attention_mask, self.dtype)

        # pass the hidden states through the encoder layers
        for i, layer_module in enumerate(self.bert_layers):
            # feed the encoding from the last bert_layer to the next
            hidden_states = layer_module(hidden_states, extended_attention_mask)

        return hidden_states

    def forward(self, input_ids, attention_mask):
        """
        input_ids: [batch_size, seq_len], seq_len is the max length of the batch
        attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
        """
        # get the embedding for each input token
        embedding_output = self.embed(input_ids=input_ids)

        # feed to a transformer (a stack of BertLayers)
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

        # get cls token hidden state
        first_tk = sequence_output[:, 0]
        first_tk = self.pooler_dense(first_tk)
        first_tk = self.pooler_af(first_tk)

        return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}
