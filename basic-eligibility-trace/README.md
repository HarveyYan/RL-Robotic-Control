# RL-Robotic-Control

This section only uses the most fundamental eligibility trace, to an actor-critic architecture which uses function approximators  to police and value function.

Some observations:
* Setting lambda to 1 and discount factor to near 1 could yields highest performance in the continuous mountain car problem.
* Need to bound the magnitude of policy update using KL divergence penalty or constraints, such as the TRPO or PPO methods.
* Batch methods may be more stable than updating policy and value function at each time step(hopeful noise in a batch would cancel each other out).
