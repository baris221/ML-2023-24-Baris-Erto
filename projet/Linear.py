from Module import *

class Linear(Module):
    def __init__(self,input_size,output_size):
        self.input_size = input_size
        self.output_size = output_size
        self._parameters = np.random.randn(self.input_size, self.output_size)

    def zero_grad(self):
        ## Annule gradient
        self._gradient= np.zeros((self.input_size, self.output_size))
        #pass

    def forward(self, X):
        """X@w=(batch_size,input_size)@(input_size,output_size)=(batch,output_size)"""
        ## Calcule la passe forward
        assert(X.shape[1]==self.input_size), "Le dimension de X doit être (batch_size,input_size)"
        self.output=np.dot(X, self._parameters)
        return self.output  
        #pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient

        
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass
