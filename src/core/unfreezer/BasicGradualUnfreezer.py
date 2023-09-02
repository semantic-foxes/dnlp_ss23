from src.core.unfreezer import GradualUnfreezerTemplate


class BasicGradualUnfreezer(GradualUnfreezerTemplate):
    def __init__(self, model):
        super().__init__(model)

        self.current_layer = self.total_layers - 1

    def step(self):
        if self.current_layer > 0:
            self._unfreeze_layer(self.current_layer)
            self.current_layer -= 1
        else:
            pass
