import numpy as np
import torch
from torch.nn import functional as F
from torch import nn

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()

        self.gamma = 0.99
        self.eps_clip = 0.1

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
        return action, action_prob

    def filter_image(self, I):
        """ prepro 210x160x3 uint8 frame into 6000 (75x80) 1D float vector. Borrowed from Andrej Karpathy"""
        I = I[35:185]  # crop - remove 35px from start & 25px from end of image in x,
        I = I[::2, ::2, 0]  # downsample by factor of 2.
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1.
        return I

    def state_to_tensor(self, I, opponent=False):
        if I is None:
            return torch.zeros(1, 6000)
        I = self.filter_image(I)
        if opponent:
            I = np.fliplr(I)
        I = torch.from_numpy(I.astype(np.float32).ravel()).unsqueeze(0)
        return I

    def pre_process(self, x, prev_x, opponent=False):
        x = self.state_to_tensor(x, opponent)
        prev_x = self.state_to_tensor(prev_x, opponent)
        return torch.cat([x, prev_x], dim=1)

    def convert_action(self, action):
        #Actions are offset by one in Atari, 1:Still, 2:Up, 3:Down
        return action + 1

    def loss(self, obs, action, action_prob, advantage):
        # Shape the action for the softmax:
        vs = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        ts = torch.FloatTensor(vs[action.cpu().numpy()])

        # Compute logits here so that weights get updated
        # Tensors are annoying to index and batch, so this allows the action batches in training to work
        logits = self.layers(obs)

        # PPO ratio of probabilities, stabilizes training over traditional PG
        r = torch.sum(F.softmax(logits, dim=1) * ts, dim=1) / action_prob

        # Compute the two losses from PPO paper:
        loss1 = r * advantage
        loss2 = torch.clamp(r, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
        loss = -torch.min(loss1, loss2)
        loss = torch.mean(loss)

        return loss
