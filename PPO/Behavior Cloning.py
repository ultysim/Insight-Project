import gym
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
#from model import Policy
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(6000*2, 512), nn.ReLU(),
            nn.Linear(512, 2),
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



policy = Policy()
policy.load_state_dict(torch.load("params.ckpt"))

#Define loss criterion
criterion = nn.BCEWithLogitsLoss()
#Define the optimizer
optimizer = torch.optim.Adam(policy.parameters(), lr=0.0001)#0.0001 stable

def get_opponent_screen(obs):
    numpy_obs = np.fliplr(obs.cpu().numpy().reshape(75, 80))
    return torch.from_numpy(numpy_obs.astype(np.float32).ravel()).unsqueeze(0)

def get_opponent_action(x, prev_x):
    """Input: x, current screen; prev_x: previous screen
    Output: Returns opponent action. -1 for no action, 0 for up, 1 for down"""
    if prev_x is None:
        prev_x = x
    movement = x - prev_x
    op_window = movement[35:194, 16:20]
    #Remove 0s and see the action
    op_window = op_window[op_window != 0]
    if len(op_window) == 0:
        return -1
    if op_window[0] < 100:
        return 0
    else:
        return 1


env = gym.make('PongNoFrameskip-v4')
env.reset()
env.render()
policy.train()


for episode in range(4):
    prev_obs = None
    op_action_pred = -1
    op_action_prob = 0
    op_action_logit = 0

    correct_hold = []
    op_action_hold = []
    op_action_prob_hold = []
    op_action_logit_hold = torch.Tensor()
    obs = env.reset()


    for t in range(1000):
        env.render()

        d_obs = policy.pre_process(obs, prev_obs)
        op_obs = policy.pre_process(obs, prev_obs, opponent=True)
        op_action_real = get_opponent_action(obs, prev_obs)
        if op_action_real != -1:
            op_action_hold.append(torch.tensor(op_action_real))
            op_action_prob_hold.append(op_action_prob)
            op_action_logit_hold = torch.cat((op_action_logit_hold,op_action_logit))
            correct_hold.append(op_action_real == op_action_pred)


        action, action_prob, _ = policy(d_obs)
        op_action_pred, op_action_prob, op_action_logit = policy(op_obs)

        prev_obs = obs
        obs, reward, done, info = env.step(policy.convert_action(action))

        if done:
            print('Episode %d (%d timesteps) - Reward: %.2f' % (episode, t, reward))
            break
    actions_stack = torch.stack(op_action_hold)
    actions_stack = one_hot_embedding(actions_stack, 2)
    #print(op_action_logit_hold.shape)
    #print(actions_stack.shape)

    op_action_logit_hold, actions_stack = op_action_logit_hold, actions_stack
    # Clear the previous gradients
    optimizer.zero_grad()

    loss = criterion(op_action_logit_hold, actions_stack)
    # Compute gradients
    loss.backward()
    # Adjust weights
    optimizer.step()

        # time.sleep(0.033)
    print("{}% correct guesses".format(np.mean(correct_hold)))
    print("{} Cross Entropy Loss".format(loss.item()))
env.close()
torch.save(policy.state_dict(), 'BCloneparams.ckpt')
