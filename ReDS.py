from env import GridEnv, OBS_RANDOM, spec_from_string, WALL, START
import numpy as np
import pickle
import tabulate

import torch

from torch.distributions.categorical import Categorical

class Critic:
    def __init__(self, params):
        self.params = params
    
    def __call__(self, obs, actions):
        return self.params[obs, actions]

class Actor:
    def __init__(self, params):
        self.params = params
    
    def __call__(self, obs):
        return Categorical(self.params[obs])

def mse_loss(pred, target):
    return torch.mean((pred - target)**2)

# maze = spec_from_string(
#     "#SNNN#NNNNNNNN##\\"+
#     "###N####N####N##\\"+
#     "##NN###NN###NN##\\"+
#     "##N####N####N###\\"+
#     "#OOOO#OOOO#OOOO#\\"+
#     "#OOOO#OOOO#OOOO#\\"+
#     "#OOOO#OOOO#OOOO#\\"+
#     "#OOOO#OOOO#OOOO#\\"+
#     "#OOO##OOO##OOO##\\"+
#     "###N####N####N##\\"+
#     "##NN###NN###NN##\\"+
#     "##N####N####N###\\"+
#     "#OOOO#OOOO#OOOO#\\"+
#     "#OOOO#OOOO#OOOO#\\"+
#     "#OOOO#OOOO#OOOO#\\"+
#     "#OOOO#OOOO#OOOO#\\"+
#     "#OOO##OOO##OOO##\\"+
#     "###N####N####N##\\"+
#     "##NN###NN###NN##\\"+
#     "##N####N####N###\\"+
#     "#OOOOOOOOO#OOOO#\\"+
#     "#OOOOOOOOO#OOOO#\\"+
#     "#OOOOOOOOO#OOOO#\\"+
#     "#OOOOOOOOO#OOOR#\\"
# )
maze = spec_from_string("SNNO\\"+
                    "N##N\\"+
                    "OOOO\\"+
                    "N#RO\\"
                   )
env = GridEnv(maze, observation_type=OBS_RANDOM, dim_obs=8)

with open("behavior_policy_small.pkl", "rb") as f:
    behavior_policy = pickle.load(f)
dS = env.num_states
dA = env.num_actions
gamma = 0.9

critic_params = torch.ones((dS, dA))*1/(1-gamma) * 0.5
critic = Critic(critic_params)
target_critic_params = torch.ones((dS, dA))*1/(1-gamma) * 0.5
target_critic = Critic(target_critic_params)

actor_params = torch.ones((dS,dA))/dA
actor = Actor(actor_params)
rho_params = torch.ones((dS, dA))/dA
rho = Actor(rho_params)

no_wall = env.gs.find_non(WALL)

obs = np.random.choice(no_wall, 5)
actions = Categorical(torch.tensor(behavior_policy[obs])).sample().detach().numpy()

next_obs = np.zeros_like(obs)
rewards = np.zeros_like(obs)
dones = np.zeros_like(obs)

for i, sa in enumerate(zip(obs,actions)):
    s, a = sa
    ns, rw = env.step_stateless(s, a)
    next_obs[i] = ns
    rewards[i] = rw

batch = {
    "observations": torch.tensor(obs),
    "actions": torch.tensor(actions),
    "next_observations": torch.tensor(next_obs),
    "rewards": torch.tensor(rewards),
    "dones": torch.tensor(dones),
}

q_data = critic(batch["observations"], batch["actions"])

next_pi_dists = actor(batch["next_observations"])
next_pi_actions = next_pi_dists.sample()
target_qval = target_critic(batch["observations"], next_pi_actions)
target_qval = batch["rewards"] + gamma * (1 - batch["dones"]) * target_qval
td_loss = mse_loss(q_data, target_qval)

dist = actor(batch["observations"])
pi_actions = dist.sample()

rho_dist = actor(batch["observations"])
rho_actions = rho_dist.sample()

q_pi = critic(batch["observations"], pi_actions)
q_rho = critic(batch["observations"], rho_actions)
push_down_term_reds = 0.5*q_pi + 0.5*q_rho

loss = torch.mean(push_down_term_reds + q_data) + 0.5*td_loss


# done = False
# s = env.reset()
# traj = Trajectory()
# while not done:
#     a = np.random.choice(dA, p=behavior_policy[s])
#     ns, r, done, info = env.step(a)
#     # print(f"{s=}, {a=}, {r=}, {ns=}")
#     traj.add_state(State(s, a, r, ns))
#     s = ns
#     env.render()
# traj.backprop(gamma=0.99)
# traj.print_all()

