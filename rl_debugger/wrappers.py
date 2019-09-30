import numpy as np


class SaliencyWrapper:
    """ Wrapper for get_saliency_map 
        The wrapped model return policy digit and value estimation
    """

    def __init__(self, model, phi):
        self.model = model
        self.phi = phi

    def __call__(self, obs):
        pout, vout = self.model(self.phi(obs[None]))
        pout = np.squeeze(pout.logits.data)
        vout = np.squeeze(vout.data)
        return pout, vout
