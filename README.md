# RL-Robotic-Control

A preliminary comparison between Q-PROP and Proximal Policy Gradient in Hopper-v2 environment.

![placeholder](graph/plot.png)

Needs more rigorous test and polishing give out the full potential of the integrated on-policy ppo and off-policy ddpg training.

( Disclaimer..This figure might be deceptive, because Q-PROP as of now is not stable yet... :( )

Another comparison of the two algorithms in FetchReach-v0 environment.
![placeholder](graph/fr_plot.png)

### Some observation so far
    * Normalization of observation using a Scaler, as well as the GAE is crucial; should consider adding batchnormalization layers.
    * Scaling rewards is essential for DDPG learning; adding time steps as feature is necessary for tasks with short time spans.