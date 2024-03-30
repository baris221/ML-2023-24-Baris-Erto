from Module import *

class Sequential:
    def __init__(self,modules):
        self.modules = modules
        self.modules_copy = self.modules[:]
        self.inputs = []

    def ajouter_mudule(self, module):
        """Add a module to the network."""
        self.modules.append(module)


    def reset(self):
        """Reset network to initial parameters and modules."""
        self.modules = self.modules_copy[:]
        return self

    def forward(self, input):
        self.inputs = [input]

        for module in self.modules:

            input = module(input)
            self.inputs.append(input)



        return input

    def backward(self, input, delta):
        # Pas sur des indices des listes !
        self.inputs.reverse()

        # print(f"\tDelta's (loss) shape : {delta.shape}")

        for i, module in enumerate(reversed(self.modules)):

            module.backward_update_gradient(self.inputs[i + 1], delta)
            delta = module.backward_delta(self.inputs[i + 1], delta)

        return delta

    def update_parameters(self, eps=1e-3):
        for module in self.modules:

            module.update_parameters(learning_rate=eps)

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

