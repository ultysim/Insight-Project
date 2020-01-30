import gym
import numpy as np
import torch
import random
from torch import nn
from matplotlib import pyplot as plt
from model import Policy
from Imitation_Model import Imitation

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
imitation_policy.train()

#Define loss criterion
criterion = nn.BCEWithLogitsLoss()
#Define the optimizer
optimizer = torch.optim.Adam(imitation_policy.parameters(), lr=0.001)#0.0001 stable


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


def get_opponent_action_2(obs):
    """Input: x, current screen; prev_x: previous screen
    Output: Returns opponent action. -2 if confused, 0 for up, 1 for down"""
    unstack = obs.view(2, 75, 80)
    obs = unstack[0] - unstack[1]
    opponent = get_opponent_col(obs)
    #Remove 0s and see the action
    opponent = opponent[opponent != 0]
    if len(opponent) == 0:
        return 0
    if opponent[0] > 0:
        return 1
    elif opponent[0] < 0:
        return 2
    return -1


def is_overlap(I):
    player = get_player_col(I)
    opponent = get_opponent_col(I)
    return np.sum(opponent[player == 1.]) >= 8

env = gym.make('PongNoFrameskip-v4')
env.reset()

for episode in range(1):
    prev_obs = None
    obs = env.reset()

    e_obs, op_obs = imitation_policy.pre_process(obs, prev_obs)

    op_action_pred, op_action_prob, op_action_logit = imitation_policy(op_obs)

    correct_hold = []
    op_action_hold = []
    op_action_logit_hold = torch.Tensor()

    overlap_hold = []
    det = True

    # Monitor score for mixture model
    score = 0

    for t in range(100):
        #Preprocess the images for more model and efficient state extraction:
        #e_obs = expert_policy.pre_process(obs, prev_obs)
        e_obs, op_obs = imitation_policy.pre_process(obs, prev_obs)



        op_action_real = get_opponent_action(obs, prev_obs)

        if op_action_real >= 0:
            op_action_hold.append(torch.tensor(op_action_real))
            op_action_logit_hold = torch.cat((op_action_logit_hold, op_action_logit))
            correct_hold.append(op_action_real == op_action_pred)

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

        op_action_pred, op_action_prob, op_action_logit = imitation_policy(op_obs, deterministic = det)

        prev_obs = obs
        obs, reward, done, info = env.step(action)
        score += reward

        if done:
            print('Episode %d (%d timesteps) - Reward: %.2f' % (episode, t, reward))
            break


    actions_stack = torch.stack(op_action_hold)
    print(op_action_logit_hold)
    for _ in range(10):
        n_batch = 100
        idxs = random.sample(range(len(actions_stack)), n_batch)
        action_batch = torch.LongTensor([actions_stack[idx] for idx in idxs])
        action_batch = one_hot_embedding(action_batch, 3)
        action_prob_batch = torch.FloatTensor([op_action_logit_hold[idx] for idx in idxs])
        #op_action_logit_hold, actions_stack = op_action_logit_hold, actions_stack
        # Clear the previous gradients
        optimizer.zero_grad()

        loss = criterion(op_action_logit_hold, actions_stack)
        # Compute gradients
        loss.backward()
        # Adjust weights
        optimizer.step()

    print("{}% correct guesses".format(np.mean(correct_hold)))
    print("{} Cross Entropy Loss".format(loss.item()))
env.close()
torch.save(imitation_policy.state_dict(), 'BCloneparams.ckpt')
#plt.plot(op_action_hold)
#plt.show()