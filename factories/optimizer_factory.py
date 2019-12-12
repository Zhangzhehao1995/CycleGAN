from tensorflow.keras.optimizers import (Nadam, Adam, SGD)


# Create the optimizer
class Optimizer_Factory():
    def __init__(self):
        pass

    def make(self, optimizer, learning_rate_base):


        # Create the optimizer
        if optimizer == 'nadam':
            opt = Nadam(lr=learning_rate_base, beta_1 = 0.825, beta_2 = 0.99685)

        elif optimizer == 'adam':
            opt = Adam(lr=learning_rate_base, beta_1=0.5, beta_2=0.999, epsilon=1e-08)

        elif optimizer == 'sgd':
            opt = SGD(lr=learning_rate_base, momentum=0.9)
            #opt = SGD(lr=learning_rate_base, momentum=0.9, nesterov=True)

        else:
            raise ValueError("Unknown optimizer. Valid optimizer arguments are: 'nadam', 'adam' and 'sgd'.")

        # Return the optimizer
        return opt
