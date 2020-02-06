import gym
import random
import torch
from .agent import Agent
import os.path

def Train_Expert():
    env = gym.make('PongNoFrameskip-v4')
    env.reset()

    agent = Agent()

    if os.path.isfile('3params.ckpt'):
        print("Loading model weights.")
        agent.load_state_dict(torch.load("3params.ckpt"))
    else:
        print("Could not find model weights. Training from scratch.")


    opt = torch.optim.Adam(agent.parameters(), lr=0.001)

    reward_sum_running_avg = None
    for it in range(1):
        d_obs_history, action_history, action_prob_history, reward_history = [], [], [], []
        for ep in range(10):
            obs, prev_obs = env.reset(), None
            for t in range(50000):
                d_obs = agent.pre_process(obs, prev_obs)
                with torch.no_grad():
                    action, action_prob = agent(d_obs)

                prev_obs = obs
                obs, reward, done, info = env.step(agent.convert_action(action))

                d_obs_history.append(d_obs)
                action_history.append(action)
                action_prob_history.append(action_prob)
                reward_history.append(reward)

                if done:
                    reward_sum = sum(reward_history[-t:])
                    reward_sum_running_avg = 0.99*reward_sum_running_avg + 0.01*reward_sum if reward_sum_running_avg else reward_sum
                    print('Iteration %d, Episode %d (%d timesteps) - last_action: %d, last_action_prob: %.2f, reward_sum: %.2f, running_avg: %.2f' % (it, ep, t, action, action_prob, reward_sum, reward_sum_running_avg))
                    break

        # Compute advantage:
        R = 0
        discounted_rewards = []

        # Work backwards to discount the reward:
        for r in reward_history[::-1]:
            if r != 0: R = 0 # scored/lost a point in pong, so reset reward sum
            R = r + agent.gamma * R
            discounted_rewards.insert(0, R)

        # Normalize the discounted rewards:
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()

        # Batch update policy:
        for _ in range(5):
            n_batch = 25000
            idxs = random.sample(range(len(action_history)), n_batch)
            d_obs_batch = torch.cat([d_obs_history[idx] for idx in idxs], 0)
            action_batch = torch.LongTensor([action_history[idx] for idx in idxs])
            action_prob_batch = torch.FloatTensor([action_prob_history[idx] for idx in idxs])
            advantage_batch = torch.FloatTensor([discounted_rewards[idx] for idx in idxs])

            opt.zero_grad()
            loss = agent.loss(d_obs_batch, action_batch, action_prob_batch, advantage_batch)
            loss.backward()
            opt.step()

            print('Iteration %d -- Loss: %.3f' % (it, loss))
        if it % 5 == 0:
            torch.save(agent.state_dict(), '3params.ckpt')

    env.close()


if __name__ == '__main__':
    Train_Expert()