import torch


class GradualUnfreezerTemplate:
    def __init__(self, model):
        self.layers = model.bert.bert_layers
        self.total_layers = len(self.layers)

    def start(self):
        self._freeze_layers()

    def _freeze_layers(self, start_layer=0, end_layer=-1):
        """
        Freeze all BERT layers.
        """
        if end_layer == -1:
            end_layer = self.total_layers

        for i in range(start_layer, end_layer):
            for param in self.layers[i].parameters():
                param.requires_grad = False

    def _unfreeze_layer(self, layer_number):
        if not 0 <= layer_number < self.total_layers:
            raise AttributeError(f'Incorrect layer number called to unfreeze: {layer_number}')

        for param in self.layers[layer_number].parameters():
            param.requires_grad = True

    def step(self):
        raise NotImplementedError