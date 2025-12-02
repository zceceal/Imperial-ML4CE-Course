import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import tqdm 

from common import PolicyNetwork
from utils import setup_model_saving

def your_optimization_alg(  env, *,  
                            max_episodes = 200 , 
                            max_time = 60. ,
                            ):
    """
    PPO-style Actor–Critic algorithm using the fixed PolicyNetwork architecture.
    """

    # Create file to store model weigths
    save_f_path = setup_model_saving(algorithm = "Your algorithm")
    
    # Initialize buffers to store data for plotting
    plot_data = {'reward_history': [],
                 'timesteps':[]
                 }
    
    # Initialize policy (actor)
    policy_net = PolicyNetwork(
        input_size=env.observation_space.shape[0], 
        output_size=env.action_space.shape[0],
    )

    # Critic network (value function) – allowed to design freely
    value_net = ValueNetwork(input_size=env.observation_space.shape[0])

    # Optimisers
    actor_lr  = 3e-4
    critic_lr = 1e-3
    optimizer_actor  = torch.optim.Adam(policy_net.parameters(), lr=actor_lr)
    optimizer_critic = torch.optim.Adam(value_net.parameters(), lr=critic_lr)

    # PPO / A2C hyperparameters
    gamma        = 0.99
    gae_lambda   = 0.95
    clip_eps     = 0.2
    entropy_coef = 1e-3
    value_coef   = 0.5
    max_grad_norm = 1.0
    ppo_epochs   = 4          # number of optimisation epochs per episode
    batch_size   = None       # whole episode as one batch (episodes are short)

    # Exploration parameters (Gaussian policy)
    action_std_init  = 0.5
    action_std_min   = 0.1
    action_std_decay = 0.995
    action_std       = action_std_init

    start_time     = time.time()
    best_reward    = -np.inf
    counter_steps  = 0  # total environment timesteps so far

    # -----------------------------------------------------------------------------------
    # PLEASE DO NOT MODIFY THE CODE ABOVE THIS LINE
    # -----------------------------------------------------------------------------------  

    for episode in tqdm.tqdm(range(int(max_episodes))):

        # Storage for one episode
        states      = []
        actions     = []
        logprobs    = []
        rewards     = []
        dones       = []
        values      = []

        # Reset environment and get initial state
        env.reset()
        state = env.state
        done  = False

        # Rollout one episode
        while not done:
            state_tensor = torch.from_numpy(state).float()

            # Sample action from current policy
            with torch.no_grad():
                action, logprob, _ = select_action(
                    state_tensor, policy_net, action_std
                )
                value = value_net(state_tensor).squeeze(-1)

            # Convert to numpy and clip into action bounds if Box space
            action_np = action.numpy()
            if hasattr(env.action_space, "high"):
                high = env.action_space.high
                low  = env.action_space.low
                action_np = np.clip(action_np, low, high)
            # Environment expects int16 actions
            next_state, reward, done, _ = env.step(action_np.astype(np.int16))

            # Store transition
            states.append(state_tensor)
            actions.append(torch.from_numpy(action_np).float())
            logprobs.append(logprob)
            rewards.append(float(reward))
            dones.append(done)
            values.append(value)

            state = next_state
            counter_steps += 1

        # Convert buffers to tensors
        states   = torch.stack(states)                  # (T, obs_dim)
        actions  = torch.stack(actions)                 # (T, act_dim)
        logprobs = torch.stack(logprobs)                # (T,)
        values   = torch.stack(values)                  # (T,)
        rewards  = torch.tensor(rewards, dtype=torch.float32)  # (T,)
        dones    = torch.tensor(dones, dtype=torch.float32)    # (T,)

        # Compute returns and advantages using GAE(λ)
        returns, advantages = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=gamma,
            lam=gae_lambda
        )

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO updates on this trajectory
        if batch_size is None:
            batch_size = len(rewards)

        for _ in range(ppo_epochs):
            # For this small problem, treat whole episode as a single batch
            batch_idx = torch.arange(len(rewards))

            b_states      = states[batch_idx]
            b_actions     = actions[batch_idx]
            b_old_logprob = logprobs[batch_idx].detach()
            b_returns     = returns[batch_idx].detach()
            b_advantages  = advantages[batch_idx].detach()

            # Recompute logprobs and values under current policy
            new_logprob, entropy = evaluate_actions(
                b_states, b_actions, policy_net, action_std
            )
            new_values = value_net(b_states).squeeze(-1)

            # PPO objective
            ratio = torch.exp(new_logprob - b_old_logprob)  # (T,)
            surr1 = ratio * b_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_advantages
            actor_loss = -torch.mean(torch.min(surr1, surr2)) - entropy_coef * entropy.mean()

            # Critic loss (value function)
            value_loss = F.mse_loss(new_values, b_returns)

            loss = actor_loss + value_coef * value_loss

            # Backprop
            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
            nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
            optimizer_actor.step()
            optimizer_critic.step()

        # Log episode return
        total_return = float(rewards.sum().item())
        plot_data['reward_history'].append(total_return)
        plot_data['timesteps'].append(counter_steps)

        # Save best policy so far
        if total_return > best_reward:
            best_reward = total_return
            torch.save(policy_net.state_dict(), save_f_path)

        # Decay exploration noise
        action_std = max(action_std * action_std_decay, action_std_min)

        # -----------------------------------------------------------------------------------
        # PLEASE DO NOT MODIFY THE CODE BELOW THIS LINE
        # -----------------------------------------------------------------------------------
        # Check time
        if (time.time()-start_time) > max_time:
            print("Timeout reached: the best policy found so far will be returned.")
            break

    print(f"Policy model weights saved in: {save_f_path}") 
    print(f"Best reward: {best_reward}")

    team_names = ["Student1","Student2"]
    cids = ["1234", "5678"]
    question = [0,0] # 1 if RL 0 else

    return save_f_path , plot_data


#################################
# Helper classes & functions
#################################
class ValueNetwork(nn.Module):
    """
    Critic network: estimates V(s). Architecture is similar to REINFORCE baseline.
    """
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def select_action(state_tensor, policy_net, action_std):
    """
    Sample action from a Multivariate Normal policy:
    mean from PolicyNetwork, fixed diagonal covariance from action_std.
    """
    with torch.no_grad():
        action_mean = policy_net(state_tensor)

    action_dim = action_mean.shape[-1]
    action_var = torch.full((action_dim,), action_std ** 2)
    cov_mat = torch.diag(action_var)

    dist = torch.distributions.MultivariateNormal(action_mean, covariance_matrix=cov_mat)
    action = dist.sample()
    logprob = dist.log_prob(action)
    entropy = dist.entropy()

    return action, logprob, entropy


def evaluate_actions(states, actions, policy_net, action_std):
    """
    Given states and actions, compute log-probabilities and entropy under the current policy.
    Used in PPO updates.
    """
    action_means = policy_net(states)
    action_dim = action_means.shape[-1]
    action_var = torch.full((action_dim,), action_std ** 2)
    cov_mat = torch.diag(action_var)

    dist = torch.distributions.MultivariateNormal(
        action_means, covariance_matrix=cov_mat
    )

    logprobs = dist.log_prob(actions)
    entropy  = dist.entropy()
    return logprobs, entropy


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE(λ)) and returns.
    rewards: (T,)
    values:  (T,)
    dones:   (T,)
    """
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32)
    gae = 0.0
    next_value = 0.0

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        next_value = values[t]

    returns = advantages + values
    return returns, advantages
