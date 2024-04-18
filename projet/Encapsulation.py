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

            input = module.forward(input)
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



class Optim:
    def __init__(self, network, loss, eps):
        """
        network est une sequence
        loss est une fonction de loss"""
        self.network = network
        self.loss = loss
        self.eps = eps

    def create_batches(self, X, y, batch_size):
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        batch_list = []
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            batch_list.append((X_batch, y_batch))
        return batch_list
    
    def step(self, batch_x, batch_y):
        """
        batch_x et batch_y doivent avoir même longeur"""
        # Forward pass
        y_hat = self.network.forward(batch_x)
        loss_value = self.loss.forward(batch_y, y_hat)

        # Backward pass
        loss_delta = self.loss.backward(batch_y, y_hat)
        self.network.zero_grad()
        self.network.backward(batch_x, loss_delta)
        self.network.update_parameters(self.eps)

        return loss_value


    def sgd(self,X,Y,batch_size,nb_iter):
        losses=[]
        for _ in range(nb_iter):
            loss_epoch_i=0
            for  (batch_x, batch_y) in self.create_batches(X, Y, batch_size):
                #batch_y=batch_y[:,np.newaxis]
                loss_epoch_i+=self.step(batch_x, batch_y).sum()
            losses.append(loss_epoch_i/len(Y))       
        return losses
       
    def score(self,X,y):
        if len(y.shape) != 1:  #Si ce n'est pas onehot
            y = y.argmax(axis=1)
        y_hat = np.argmax(self.network.forward(X), axis=1) #On calcule la classe plus elevé
        return np.where(y == y_hat, 1, 0).mean() #retourne accuracy