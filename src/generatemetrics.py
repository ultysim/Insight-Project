import numpy as np
from matplotlib import pyplot as plt
import os
import sys


def Generate(data_dir):
    cwd = os.getcwd()

    def get_data(data_dir):
        #global cwd
        Cross_Loss = np.load(cwd+'\\'+data_dir+'\\CrosslossHold.npy')
        Prob = np.load(cwd+'\\'+data_dir+'\\ProbHold.npy')
        Score = np.load(cwd+'\\'+data_dir+'\\ScoreHold.npy')
        Timestep = np.load(cwd+'\\'+data_dir+'\\TimestepHold.npy')
        return Cross_Loss, Prob, Score, Timestep

    loss, prob, score, time = get_data(data_dir)

    loss_fig = plt.figure('Loss')
    plt.plot(loss)
    plt.xlabel('Game')
    plt.ylabel('Cross Entropy Loss')
    loss_fig.savefig(cwd+'\\'+data_dir+'\\Loss')

    prob_fig = plt.figure('Prob')
    plt.plot(prob)
    plt.xlabel('Game')
    plt.ylabel('Percent Accurate Action Predictions')
    prob_fig.savefig(cwd+'\\'+data_dir+'\\Pred')

    score_fig = plt.figure('Score')
    plt.plot(score)
    plt.xlabel('Game')
    plt.ylabel('Score Differential')
    score_fig.savefig(cwd+'\\'+data_dir+'\\Score')

    score_mean = np.mean(score)
    score_std = np.std(score)
    print("{} pm {} Average score over {} games".format(round(score_mean,2),round(score_std,2),len(score)))

    time_fig = plt.figure('Time')
    plt.plot(time)
    plt.xlabel('Game')
    plt.ylabel('Time Steps')
    time_fig.savefig(cwd+'\\'+data_dir+'\\Time')

    time_mean = np.mean(time)
    time_std = np.std(time)
    print("{} pm {} Average score over {} games".format(round(time_mean,2),round(time_std,2),len(time)))


if __name__ == '__main__':
    assert len(sys.argv) == 2, "Please specify one data directory"
    data_dir = str(sys.argv[1])
    Generate(data_dir)