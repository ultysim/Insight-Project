import gym
import numpy as np
import torch
import random
from torch import nn
from matplotlib import pyplot as plt
from model import Policy
from Imitation_Model import Imitation
import time

MIXTURE_WEIGHT = 0.1

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]

#Initialize both agents

expert_policy = Policy()
expert_policy.load_state_dict(torch.load("3params.ckpt"))
expert_policy.eval()

imitation_policy = Imitation()
imitation_policy.eval()

def get_player_col(obs):
    return obs.cpu().numpy().reshape(75, 80)[:, 70:72]


def get_opponent_col(obs):
    return obs.cpu().numpy().reshape(75, 80)[:, 8:10]


def get_opponent_screen(obs):
    numpy_obs = np.fliplr(obs.cpu().numpy().reshape(75, 80))
    return torch.from_numpy(numpy_obs.astype(np.float32).ravel()).unsqueeze(0)


def get_opponent_action(x, prev_x):
    """Input: x, current screen; prev_x: previous screen
    Output: Returns opponent action. 0 for no action, 1 for up, 2 for down, -1 return for stability"""
    if prev_x is None:
        prev_x = x
    movement = x - prev_x
    op_window = movement[35:194, 16:20]
    #Remove 0s and see the action
    op_window = op_window[op_window != 0]
    if len(op_window) == 0:
        return 0
    if op_window[0] < 100:
        return 1
    elif op_window[0] >= 100:
        return 2
    return -1


env = gym.make('PongNoFrameskip-v4')
env.reset()


####################################
########## Metrics #################
####################################
timestep_hold = []
score_hold = []

for episode in range(1):
    prev_obs = None
    obs = env.reset()
    # Monitor score for mixture model
    score = 0

    for t in range(30000):
        if t < 10:
            time.sleep(2.0)
        env.render()
        e_obs, op_obs = imitation_policy.pre_process(obs, prev_obs)

        #Generate mixture:
        expert_logits = expert_policy.layers(e_obs)
        expert_probs = nn.Softmax(dim=0)(expert_logits.view(-1))
        imitation_logits = imitation_policy.layers(e_obs)
        imitation_probs = nn.Softmax(dim=0)(imitation_logits.view(-1))

        #Look at score delta:
        scale = 0
        if score > 0:
            scale = score * MIXTURE_WEIGHT

        prob_mixture = (1-scale)*expert_probs + scale*imitation_probs

        action = torch.distributions.Categorical(probs=prob_mixture).sample() + 1

        prev_obs = obs
        obs, reward, done, info = env.step(action)
        score += reward

        if done:
            print('Episode %d (%d timesteps) - Reward: %.2f' % (episode, t, score))
            timestep_hold.append(t)
            score_hold.append(score)
            break


env.close()
#torch.save(imitation_policy.state_dict(), 'BCloneparams.ckpt')
#plt.plot(op_action_hold)
#plt.show()
np.save('ScoreHold', score_hold)
np.save('TimestepHold', timestep_hold)
