import numpy as np
import MixtureModelPong as mmp



def test_opponent_action():
    op_initial = np.load('Opponent Initial.npy')
    op_up = np.load('Opponent Up.npy')
    op_down = np.load('Opponent Down.npy')
    assert 0 == mmp.get_opponent_action(op_initial, op_initial), "Failure Still"
    assert 1 == mmp.get_opponent_action(op_up, op_initial), "Failure Up"
    assert 2 == mmp.get_opponent_action(op_down, op_initial), "Failure Down"

if __name__ == '__main__':
    test_opponent_action()