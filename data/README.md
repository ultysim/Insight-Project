# Experimental Results

Results from various experiments can be analyzed with the Generate() function from generate_metrics.py. Data directory has associated plots and raw data. The naming convention is: EXPERIMENT_NAME + MIXTURE_RATIO, if handicap was used H# where # is the value of the handicap, if learning rate was adjusted LR# where # is the value of the learning rate.

For example: 

Smooth02. Smooth is an experiment tag where I used every other time step to compute the opponent's action. I used a mixture ratio of 0.2. 

Smooth01H2. Same experimental opponent action smoothing as above, mixture ratio of 0.1, handicap of +2. 

