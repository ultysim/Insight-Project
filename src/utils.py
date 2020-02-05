import torch
import numpy as np

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

def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes)
    return y[labels]