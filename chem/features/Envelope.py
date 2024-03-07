import numpy as np



class Envelope:
    """
    Envelope function that ensures a smooth cutoff
    """
    def __init__(self, exponent):
        self.exponent = exponent
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def call(self, inputs):

        # Envelope function divided by r
        env_val = 1 / inputs + self.a * inputs**(self.p - 1) + self.b * inputs**self.p + self.c * inputs**(self.p + 1)

        return np.where(inputs < 1, env_val, np.zeros_like(inputs))

