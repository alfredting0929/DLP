import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])
        self.sos_token_id = self.mask_token_id + 1

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        latent, code_indices, q_loss = self.vqgan.encode(x)
        code_indices = code_indices.view(latent.size(0), -1)
        return latent, code_indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        _, z_indices = self.encode_to_z(x)
        # Obtain how many tokens to be mask out -> r = (gamma(uniform_ratio) * num_tokens)
        n = math.floor(self.gamma(np.random.uniform()) * z_indices.shape[1])
        # Get the indices to be mask out according to the top r tokens
        sample = torch.rand(z_indices.shape, device=z_indices.device).topk(n, dim=1).indices
        # All 0 mask
        mask = torch.zeros(z_indices.shape, dtype=torch.bool, device=z_indices.device)
        # Substitute False mask with sample indices to True
        mask.scatter_(dim=1, index=sample, value=True)

        masked_indices = self.mask_token_id * torch.ones_like(z_indices, device=z_indices.device)
        # Apply the mask to z_indices
        a_indices = mask * masked_indices + (~mask) * z_indices

        logits = self.transformer(a_indices)
        return logits, z_indices
        
##TODO3 step1-1: define one iteration decoding       
    @torch.no_grad()
    def inpainting(self, z_indices, mask_bc, ratio):
        # Mask the latent code
        masked_indices = self.mask_token_id * torch.ones_like(z_indices, device=z_indices.device)
        z_indices[mask_bc] = masked_indices[mask_bc]

        logits = self.transformer(z_indices)
        prob_dist = torch.softmax(logits, dim=-1)
        z_indices_predict_prob, z_indices_predict = torch.max(prob_dist, dim=-1)  

        # Compute the confidence
        g = torch.distributions.Gumbel(0, 1).sample(z_indices_predict_prob.shape).to(z_indices.device)
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g
        # For unmask tokens, set confidence to inf
        confidence[~mask_bc] = float('inf')

        # Determine the number of tokens from mask
        num_mask = torch.sum(mask_bc)
        n = math.floor(self.gamma(ratio) * num_mask)
        # Apply mask
        _, indices = torch.sort(confidence, dim=-1)
        new_mask = torch.zeros_like(mask_bc, dtype=torch.bool, device=mask_bc.device)
        new_mask.scatter_(dim=-1, index=indices[:, :n], value=True)
        new_mask[~mask_bc] = mask_bc[~mask_bc]
        
        return z_indices_predict, new_mask

    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
