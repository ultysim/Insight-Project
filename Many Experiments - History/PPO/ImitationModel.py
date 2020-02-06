import gym
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
#from model import Policy
import time


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(6000*2, 512), nn.ReLU(),
            nn.Linear(512, 2),
        )

    def state_to_tensor(self, I, opponent=False):
        """ prepro 210x160x3 uint8 frame into 6000 (75x80) 1D float vector """
        if I is None:
            return torch.zeros(1, 6000)
        if opponent:
            I = np.fliplr(I)
        I = I[35:185] # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
        I = I[::2,::2,0] # downsample by factor of 2.
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
        return torch.from_numpy(I.astype(np.float32).ravel()).unsqueeze(0)

    def pre_process(self, x, prev_x, opponent=False):
        #return self.state_to_tensor(x) - self.state_to_tensor(prev_x)
        return torch.cat([self.state_to_tensor(x, opponent), self.state_to_tensor(prev_x, opponent)], dim=1)

    def convert_action(self, action):
        return action + 2

    def forward(self, d_obs, deterministic=False):
        logits = self.layers(d_obs)
        if deterministic:
            action = int(torch.argmax(logits[0]).detach().cpu().numpy())
            action_prob = 1.0
        else:
            c = torch.distributions.Categorical(logits=logits)
            action = int(c.sample().cpu().numpy()[0])
            action_prob = float(c.probs[0, action].detach().cpu().numpy())
        return action, action_prob, logits
