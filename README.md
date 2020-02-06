# Insight-Project
Reinforcement learning project for Insight AI fellowship, completed in 4 weeks from project research and ideation to experimentation and completed product.

# Motivation:

Competitive games are best when the players are evenly matched in skill. The experience is quickly ruined if games feel unfair or unbalanced. Developers spend valuable time and resources balancing games and hard-coding various difficulty scaling metrics and parameters. Is there a way to use AI to automatically scale the difficulty of an opponent and dynamically adapt to the opponent?

# Approach:

It has been empirically proven that reinforcement learning algorithms can generalize and play any video game at a super human level. Deepmind was the first with a deep Q-network applied to a suite of Atari games, more recently Google Deepmind with AlphaStar and Open Ai with Open Five. But these algorithms are trained to win and have little use for the average player.

I choose Atari Pong as the environment for this experiment as it is the only two player game in the Open Ai gym suite. An example of the expert agent in action:

![Expert Pong Agent](https://github.com/ultysim/Insight-Project/blob/master/ExpertInAction.gif)

The idea of this project is to take a use a mixture model comprised of an expert agent and a actively learning imitation agent. The mixture ratio is a function of the positive score differential between the algorithm agent and the opponent, as the agent is ahead by more points, more of the imitation is mixed in. 

Mixing ratio:

![Mixing Ratio:](https://github.com/ultysim/Insight-Project/blob/master/mixtureparam.png)

Mixture model:

![Mixture Model](https://github.com/ultysim/Insight-Project/blob/master/mixturemodel.png)

# Results:

Game length increased by over 30% with the mixture model approach over baseline stand alone expert policy. The games were also more competitive. Base line expert policy wins with a score differential of +15, mixture model with no handicap wins with a differential of +1.8, and with a handicap can lose on average with a score differential of -0.5.

Results from various experiments can be analyzed with the Generate() function from generate_metrics.py. Data directory has associated plots and raw data. The naming convention is: EXPERIMENT_NAME + MIXTURE_RATIO, if handicap was used H# where # is the value of the handicap, if learning rate was adjusted LR# where # is the value of the learning rate.

Video example of the mixture model agent playing in real time:

https://www.youtube.com/watch?v=ORqwuvWxQUI
