import gym
import numpy as np
import torch
import random
from torch import nn
from matplotlib import pyplot as plt
from model import Policy
from Imitation_Model import Imitation
from datetime import datetime
import os
from utils import *

MIXTURE_WEIGHT = 0.1
BATCH_SIZE = 7500
N_BATCHES = 10

time = datetime.now().strftime('%m-%d-%H-%M')
output_dir = time+'_'+str(MIXTURE_WEIGHT)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print("Directory ", output_dir,  " Created ")
else:
    print("Directory ", output_dir,  " already exists")

#Initialize both agents
expert_policy = Policy()
expert_policy.load_state_dict(torch.load("3params.ckpt"))
expert_policy.eval()

imitation_policy = Imitation()
imitation_policy.train()

#Define loss criterion
criterion = nn.CrossEntropyLoss()
#Define the optimizer
optimizer = torch.optim.Adam(imitation_policy.parameters(), lr=0.001)#0.0001 stable

env = gym.make('PongNoFrameskip-v4')
####################################
########## Metrics #################
####################################
timestep_hold = []
score_hold = []
prob_hold = []
crossloss_hold = []

#TODO Make this a cmd line arg
for episode in range(30):
    prev_obs = None
    op_prev_obs = None
    obs = env.reset()

    e_obs, op_obs = imitation_policy.pre_process(obs, prev_obs)

    op_action_pred, op_action_prob, op_action_logit = imitation_policy(op_obs)

    correct_hold = []
    op_action_hold = []
    op_obs_hold = []

    # Monitor score for mixture model
    score = 0
    op_state_flag = True
    #TODO Make this a cmd line arg
    for t in range(30):
        #Preprocess the images for more model and efficient state extraction:
        e_obs, _ = imitation_policy.pre_process(obs, prev_obs)
        if op_state_flag:
            op_state_flag = False
            _, op_obs = imitation_policy.pre_process(obs, op_prev_obs)
            op_action_real = get_opponent_action(obs, op_prev_obs)
            if op_action_real >= 0:
                op_action_hold.append(torch.tensor(op_action_real))
                correct_hold.append(op_action_real == op_action_pred)
                op_obs_hold.append(op_obs)
            op_action_pred, op_action_prob, op_action_logit = imitation_policy(op_obs, deterministic=True)
            op_prev_obs = obs
        else:
            op_state_flag = True

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

    batch_loss = []
    for b in range(N_BATCHES):
        print('Train batch {}/{}'.format(b, N_BATCHES))
        idxs = random.sample(range(len(op_action_hold)), BATCH_SIZE)
        action_batch = torch.stack([op_action_hold[idx] for idx in idxs])
        #actions_stack = torch.stack(op_action_hold)
        obs_batch = torch.cat([op_obs_hold[idx] for idx in idxs])
        #temp = torch.cat(op_obs_hold, 0)
        op_action_logit_hold = imitation_policy.layers(obs_batch)

        # Clear the previous gradients
        optimizer.zero_grad()
        loss = criterion(op_action_logit_hold, action_batch)
        # Compute gradients
        loss.backward()
        # Adjust weights
        optimizer.step()
        batch_loss.append(loss.item())

    run_loss = np.mean(batch_loss)
    print("{}% correct guesses".format(np.mean(correct_hold)))
    print("{} Cross Entropy Loss".format(run_loss))
    prob_hold.append(np.mean(correct_hold))
    crossloss_hold.append(run_loss)
env.close()
#torch.save(imitation_policy.state_dict(), 'BCloneparams.ckpt')
#plt.plot(op_action_hold)
#plt.show()
np.save(output_dir+'/ScoreHold',score_hold)
np.save(output_dir+'/TimestepHold', timestep_hold)
np.save(output_dir+'/ProbHold', prob_hold)
np.save(output_dir+'/CrosslossHold', crossloss_hold)