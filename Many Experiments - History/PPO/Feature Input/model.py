import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import random


class Policy(nn.Module):
    def __init__(self, n_actions):
        super(Policy, self).__init__()

        self.gamma = 0.99
        self.eps_clip = 0.1

        self.n_actions = n_actions

        self.layers = nn.Sequential(
            nn.Linear(6, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, n_actions),
        )

    def state_to_tensor(self, I,):
        """ prepro 210x160x3 uint8 frame into 6000 (75x80) 1D float vector """
        I = I[34:194]  # crop height
        I = I[:, :, 0]  # downsample by factor of 2.
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else
        return I
        # return torch.from_numpy(I.astype(np.float32).ravel()).unsqueeze(0)

    def get_opponent_col(self, obs):
        # return obs.cpu().numpy().reshape(75, 80)[:, 8:10]
        return obs[:, 16:20]

    def get_player_col(self, obs):
        # return obs.cpu().numpy().reshape(75, 80)[:, 70:72]
        return obs[:, 140:144]

    def get_paddle(self, col, player=True):
        '''Takes a processed image and finds the top of the paddle
        '''
        if player:
            look = np.where(self.get_player_col(col) == 1.)
        else:
            look = np.where(self.get_opponent_col(col) == 1.)
        if len(look) == 0:
            return -1
        if len(look[0]) == 0:
            return -1
        return look[0][0]

    def remove_paddles(self, obs):
        obs = np.array(obs)
        obs[:, 16:20] = 0.0
        obs[:, 140:144] = 0.0
        return obs

    def get_ball(self, obs):
        obs = self.remove_paddles(obs)
        locs = np.where(obs == 1.)
        if len(locs) == 0:
            return 0., 0.
        if len(locs[0]) == 0:
            ball_y = 0.
        elif len(locs[0]) == 1:
            ball_y = locs[0]
        elif len(locs[0]) > 1:
            ball_y = locs[0][0]

        else:
            ball_y = locs[0]
        if len(locs) == 2:
            if len(locs[1]) == 0:
                ball_x = 0.
            elif len(locs[1]) > 1:
                ball_x = locs[1][0]
            else:
                ball_x = locs[1]
        else:
            ball_x = 0.
        if type(ball_y) == list:
            ball_y = ball_y[0]
        if type(ball_x) == list:
            ball_x = ball_x[0]

        return ball_y, ball_x

    def get_velocity(self, time1, time0):
        vy = time1[0] - time0[0]
        vx = time1[1] - time0[1]
        return vy, vx

    def generate_features(self, obs, prev_pos):
        obs = self.state_to_tensor(obs)
        player_loc = self.get_paddle(obs)
        opponent_loc = self.get_paddle(obs, player=False)
        ball_y, ball_x = self.get_ball(obs)
        # Normalize features:
        v_y, v_x = self.get_velocity([ball_y, ball_x], prev_pos)
        return [player_loc, opponent_loc, ball_y, ball_x, v_y, v_x]

    def normalize_features(self, player_loc, opponent_loc, ball_y, ball_x):
        player_loc /= 160.
        opponent_loc /= 160.
        ball_y /= 160.
        ball_x /= 160.
        return player_loc, opponent_loc, ball_y, ball_x

    def convert_action(self, action):
        if self.n_actions == 2:
            return action + 2
        elif self.n_actions == 3:
            return action + 1
        else:
            # Raise an error
            pass

    def forward(self, d_obs, action=None, action_prob=None, advantage=None, deterministic=False):
        if action is None:
            with torch.no_grad():
                logits = self.layers(d_obs)
                if deterministic:
                    action = int(torch.argmax(logits[0]).detach().cpu().numpy())
                    action_prob = 1.0
                else:
                    c = torch.distributions.Categorical(logits=logits)
                    action = int(c.sample().cpu().numpy()[0])
                    action_prob = float(c.probs[0, action].detach().cpu().numpy())
                return action, action_prob


        # PPO
        vs = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        ts = torch.FloatTensor(vs[action.cpu().numpy()])
        
        logits = self.layers(d_obs)
        r = torch.sum(F.softmax(logits, dim=1) * ts, dim=1) / action_prob
        loss1 = r * advantage
        loss2 = torch.clamp(r, 1-self.eps_clip, 1+self.eps_clip) * advantage
        loss = -torch.min(loss1, loss2)
        loss = torch.mean(loss)

        return loss
