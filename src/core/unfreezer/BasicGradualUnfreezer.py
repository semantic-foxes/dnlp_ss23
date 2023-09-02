from src.core.unfreezer import GradualUnfreezerTemplate


class BasicGradualUnfreezer(GradualUnfreezerTemplate):
    def __init__(
            self,
            model,
            layers_per_step: int = 1,
            steps_to_hold: int = 1
    ):
        super().__init__(model)

        self.current_layer = self.total_layers - 1
        self.layers_per_step = layers_per_step
        self.steps_to_hold = steps_to_hold
        self.current_step = 0

    def step(self):
        self.current_step = self.current_step % self.steps_to_hold

        if self.current_step == 0:
            for _ in range(self.layers_per_step):
                if self.current_layer > 0:
                    self._unfreeze_layer(self.current_layer)
                    self.current_layer -= 1
                else:
                    pass
        self.current_step += 1
