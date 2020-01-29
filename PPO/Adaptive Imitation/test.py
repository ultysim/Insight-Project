import gym
import numpy as np
import torch
from torch import nn
from model import Policy
import time

env = gym.make('PongNoFrameskip-v4')
#env = gym.wrappers.Monitor(env, './tmp', video_callable=lambda ep_id: True, force=True)
env.reset()

policy = Policy()
policy.load_state_dict(torch.load('3params.ckpt'))
policy.eval()


time_hold = []
score_hold = []
for episode in range(30):
    prev_obs = None
    obs = env.reset()
    score = 0
    for t in range(50000):
        #env.render()

        d_obs = policy.pre_process(obs, prev_obs)
        with torch.no_grad():
            action, action_prob = policy(d_obs, deterministic=False)
        
        prev_obs = obs
        obs, reward, done, info = env.step(policy.convert_action(action))
        score += reward
        if done:
            print('Episode %d (%d timesteps) - Reward: %.2f' % (episode, t, score))
            time_hold.append(t)
            score_hold.append(score)
            break


env.close()
print("{} pm {} time steps".format(np.mean(time_hold),np.std(time_hold)))

print("{} pm {} score delta".format(np.mean(score_hold),np.std(score_hold)))
