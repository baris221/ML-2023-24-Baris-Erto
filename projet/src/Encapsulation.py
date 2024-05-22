from src.Module import *
from tqdm import tqdm, trange
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from copy import deepcopy

class Sequential:
    def __init__(self, Modules):
        self.modules = Modules
        self.modules_copy = deepcopy(self.modules)
        self.inputs = []


    def insert(self, idx: int, module: Module):
        """Insert a module to the network at a specified indice."""
        self.modules.insert(idx, module)

    def reset(self):
        """Reset network to initial parameters and modules."""
        self.modules = deepcopy(self.modules_copy)
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
    def __init__(self, network: Sequential, loss: Loss, eps: float):
        self.network = network
        self.loss = loss
        self.eps = eps

    def _create_batches(self, X, y, batch_size):
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
        # Forward pass
        y_hat = self.network.forward(batch_x)
        loss_value = self.loss.forward(batch_y, y_hat)

        # Backward pass
        loss_delta = self.loss.backward(batch_y, y_hat)
        self.network.zero_grad()
        self.network.backward(batch_x, loss_delta)
        self.network.update_parameters(self.eps)

        return loss_value

    def SGD(
        self,
        X,
        y,
        batch_size: int,
        epochs: int,
    ):

        losses = []
        for epoch in trange(epochs):
            loss_sum = 0

            for X_i, y_i in self._create_batches(X, y, batch_size):
                loss_sum += self.step(X_i, y_i).sum()

            losses.append(loss_sum / len(y))

            # print(f"Epoch [{epoch+1}], Loss = {losses[-1]:.4f}")

        return np.array(losses)

    def SGD_eval(
        self,
        X,
        y,
        batch_size: int,
        epochs: int,
        return_dataframe: bool = False,
    ):

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        # Sauvegarde pour éventuelle utilisation en dehors de la fonction
        self.X_train, self.X_test, self.y_train, self.y_test = (
            X_train,
            X_test,
            y_train,
            y_test,
        )

        # Training
        losses_train = []
        losses_test = []
        scores_train = []
        scores_test = []

        batch_progress = tqdm(desc="Batch", position=1, total=len(X_train) // batch_size, mininterval=0.3)
        epoch_progress = tqdm(range(epochs), desc="Epoch", position=0)
        for _ in epoch_progress:
            loss_sum = 0
            batch_iter = self._create_batches(X_train, y_train, batch_size)
            for X_i, y_i in batch_iter:
                loss_batch_vect = self.step(X_i, y_i)
                loss_sum += loss_batch_vect.sum()
                batch_progress.update()
            batch_progress.reset()  # Reset batch bar

            epoch_train_loss = loss_sum / len(y_train)
            losses_train.append(epoch_train_loss)
            epoch_train_score = self.score(X_train, y_train)
            scores_train.append(epoch_train_score)

            # Epoch evaluation
            y_hat = self.network.forward(X_test)
            epoch_test_loss = self.loss.forward(y_test, y_hat).mean()
            epoch_test_score = self.score(X_test, y_test)
            losses_test.append(epoch_test_loss)
            scores_test.append(epoch_test_score)

            # Update the epoch progress bar with the latest epoch loss value
            epoch_progress.set_postfix(
                {
                    "train_loss": epoch_train_loss,
                    "train_score": epoch_train_score,
                    "test_loss": epoch_test_loss,
                    "test_score": epoch_test_score,
                }
            )

        batch_progress.close()
        if return_dataframe:
            self.train_df = DataFrame(
                {
                    "epoch": np.arange(len(losses_train)),
                    "loss_train": losses_train,
                    "loss_test": losses_test,
                    "score_train": scores_train,
                    "score_test": scores_test,
                }
            )
        else:
            self.train_df = (
                np.array(losses_train),
                np.array(scores_train),
                np.array(losses_test),
                np.array(scores_test),
            )
        return self.train_df

    def score(self, X, y):
        assert X.shape[0] == y.shape[0], ValueError()
        if len(y.shape) != 1:  # eventual y with OneHot encoding
            y = y.argmax(axis=1)
        y_hat = np.argmax(self.network.forward(X), axis=1)
        return np.where(y == y_hat, 1, 0).mean()