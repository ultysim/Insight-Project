import gym
import numpy as np
import torch
from torch import nn

class Imitation(nn.Module):
    def __init__(self):
        super(Imitation, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(6000*2, 512), nn.ReLU(),
            nn.Linear(512, 3),
        )

    def forward(self, d_obs, deterministic=False):
        logits = self.layers(d_obs)
        if deterministic:
            action = torch.argmax(logits[0])
            action_prob = 1.0
        else:
            c = torch.distributions.Categorical(logits=logits)
            action = c.sample()
            action_prob = c.probs[0, action]
        return action, action_prob, logits

    def filter_image(self, I):
        """ prepro 210x160x3 uint8 frame into 6000 (75x80) 1D float vector """
        I = I[35:185]  # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
        I = I[::2, ::2, 0]  # downsample by factor of 2.
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
        return I

    def state_to_tensor(self, I):
        if I is None:
            return torch.zeros(1, 6000), torch.zeros(1, 6000)
        player_I = self.filter_image(I)
        opponent_I = np.fliplr(player_I)
        p_I = torch.from_numpy(player_I.astype(np.float32).ravel()).unsqueeze(0)
        o_I = torch.from_numpy(opponent_I.astype(np.float32).ravel()).unsqueeze(0)
        return p_I, o_I

    def pre_process(self, x, prev_x):
        player_x, opponent_x = self.state_to_tensor(x)
        player_prev, opponent_prev = self.state_to_tensor(prev_x)
        return torch.cat([player_x, player_prev], dim=1), torch.cat([opponent_x, opponent_prev], dim=1)

    def convert_action(self, action):
        return action + 1