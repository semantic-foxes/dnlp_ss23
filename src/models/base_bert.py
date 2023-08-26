import re
import os

from typing import List

import torch
from torch import nn

from src.utils import (logger, is_remote_url, cached_path,
                       get_parameter_dtype, hf_bucket_url)


class BertPreTrainedModel(nn.Module):
    base_model_prefix = 'bert'
    _keys_to_ignore_on_load_missing = ['position_ids']
    _keys_to_ignore_on_load_unexpected = None

    def __init__(self, initializer_range):
        super().__init__()
        self.initializer_range = initializer_range

    def init_weights(self):
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @property
    def dtype(self) -> torch.dtype:
        return get_parameter_dtype(self)

    @staticmethod
    def convert_old_keys(state_dict):
        mapping = {
            'embeddings.word_embeddings': 'word_embedding',
            'embeddings.position_embeddings': 'pos_embedding',
            'embeddings.token_type_embeddings': 'tk_type_embedding',
            'embeddings.LayerNorm': 'embed_layer_norm',
            'embeddings.dropout': 'embed_dropout',
            'encoder.layer': 'bert_layers',
            'pooler.dense': 'pooler_dense',
            'pooler.activation': 'pooler_af',
            'attention.self': "self_attention",
            'attention.output.dense': 'attention_dense',
            'attention.output.LayerNorm': 'attention_layer_norm',
            'attention.output.dropout': 'attention_dropout',
            'intermediate.dense': 'interm_dense',
            'intermediate.intermediate_act_fn': 'interm_af',
            'output.dense': 'out_dense',
            'output.LayerNorm': 'out_layer_norm',
            'output.dropout': 'out_dropout',
            'gamma': 'weight',
            'beta': 'bias'
        }
        result = {}

        for key in state_dict.keys():
            new_key = key

            for old_name, new_name in mapping.items():
                if old_name in new_key:
                    new_key = new_key.replace(old_name, new_name)

            result[new_key] = state_dict[key]

        return result


    @classmethod
    def from_pretrained(
            cls,
            model_name: str = None,
            model_path: str = None,
            state_dict: dict = None,
            cache_dir: str = None,
            force_download: bool = False,
            resume_download: bool = False,
            proxies: list = None,
            output_loading_info: bool = False,
            local_files_only: bool = False,
            use_auth_token: str = None,
            revision: str = None,
            mirror: str = None,
            *model_args,
            **model_kwargs
    ):

        # Instantiate model
        model = cls(*model_args, **model_kwargs)

        # Load weights file for the model
        if model_path is not None:
            # In case there is a local checkpoint
            if os.path.isdir(model_path):
                archive_file = model_path

            # In case there is an url
            elif is_remote_url(model_path):
                archive_file = model_path

            else:
                raise AttributeError

        # In case only the model name is provided to look up on HF
        else:
            archive_file = hf_bucket_url(
                model_name,
                filename='pytorch_model.bin',
                revision=revision,
                mirror=mirror,
            )

        # Try to load a model
        try:
            resolved_archive_file = cached_path(
                archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
            )
        except EnvironmentError:
            if model_path:
                error_message = (
                    'Can\'t load weights for the model. Make sure that '
                    f'\'{model_path}\' is either a valid path or a URL to the model.'
                )
            else:
                error_message = (
                    'Can\'t load weights for the model. Make sure that '
                    f'\'{model_name}\' is a valid name of a model listed on '
                    '\'https://huggingface.co/models\'.'
                )

            logger.error(error_message)
            raise EnvironmentError(error_message)

        if state_dict is None:
            try:
                state_dict = torch.load(resolved_archive_file, map_location="cpu")
            except Exception:
                raise OSError('Unable to load weights from pytorch checkpoint file.')

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        state_dict = cls.convert_old_keys(state_dict)

        # TODO: WHY?
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        your_bert_params = [f"bert.{x[0]}" for x in model.named_parameters()]
        for k in state_dict:
            if k not in your_bert_params and not k.startswith("cls."):
                # possible_rename = [x for x in k.split(".")[1:-1] if x in m.values()]
                raise ValueError(
                    f"{k} cannot be reload to your model, one/some of params we provided have been renamed"
                )

        # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
        # so we need to apply the function recursively.
        def load(module: nn.Module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        has_prefix_module = any(
            s.startswith(cls.base_model_prefix) for s in state_dict.keys()
        )
        if not hasattr(model, cls.base_model_prefix) and has_prefix_module:
            start_prefix = cls.base_model_prefix + "."
        if hasattr(model, cls.base_model_prefix) and not has_prefix_module:
            model_to_load = getattr(model, cls.base_model_prefix)
        load(model_to_load, prefix=start_prefix)

        if model.__class__.__name__ != model_to_load.__class__.__name__:
            base_model_state_dict = model_to_load.state_dict().keys()
            head_model_state_dict_without_base_prefix = [
                key.split(cls.base_model_prefix + ".")[-1]
                for key in model.state_dict().keys()
            ]
            missing_keys.extend(
                head_model_state_dict_without_base_prefix - base_model_state_dict
            )

        # Some models may have keys that are not in the state by design,
        # removing them before needlessly warning the user.
        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        if len(error_msgs) > 0:
            error_messages_string = '\n\t'.join(error_msgs)
            raise RuntimeError(
                f'Error(s) in loading state_dict for {model.__class__.__name__}:\n'
                f'{error_messages_string}'
            )

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info

        return model
