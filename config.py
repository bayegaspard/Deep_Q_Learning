import  torch

# Hyper parameters
CONFIG = {
    "BUFFER_SIZE": int(2e6),     # replay buffer size
    "BATCH_SIZE": 512,           # minibatch size
    "GAMMA": 0.99,               # discount factor
    "MIN_REPLAY_SIZE" : 1000,    # How many experience or transitions we want in replay buffer before we can perform gradient decent and start training
    "EPSILOM_START": 1.0,        # starting value of epsilom
    "EPSILOM_DECAY": 10000,     # epsilom reduce after each 10000 episodes
    "EPSILOM_END": 0.02,        # final value of epsilom in epsilom greedy approach
    "TARGET_UPDATE_FREQ" : 1000, # Number of steps required to set target parameters equal to the online or policy parameters
    "TAU": 1e-3,  # for soft update of target parameters
    "LR_ACTOR": 5e-4,  # learning rate of the actor
    "LR_CRITIC": 1e-3,  # learning rate of the critic
    "WEIGHT_DECAY": 0,  # L2 weight decay
    "SIGMA": 0.05,  # std of noise
    "LEARN_STEP": 1,  # how often to learn (paper updates every 100 steps but 1 worked best here)
    "CLIP_GRADS": True,  # Whether to clip gradients
    "CLAMP_VALUE": 1,  # Clip value
    "FC1": 512,  # First linear layer size
    "FC2": 256,  # Second linear layer size
    "num_agents" : 1,
    "width": 0,   # width
    "height": 2000/20, # height
}

