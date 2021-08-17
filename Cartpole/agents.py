"""
A set of agents for solving the OpenAI Cartpole Environment
"""
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.distributions import Categorical
from Cartpole.models import RBFRegressionModel, PolicyModel

from pdb import set_trace as bp

class QLearningAgent:
    """ A Q Learning Agent """
    def __init__(self, env, use_sklearn=False):
        """ Constructor 
        
        :param env: OpenAI gym cartpole environment
        :param use_sklearn: Use Scikit-Learn SGD regressor if True, use CustomSGDRegressor if False
        """

        # Initialize agent's environment and Q(s,a) model
        self.env = env
        self.model = RBFRegressionModel(env, use_sklearn=use_sklearn)

    def play(self, epsilon=0.3, gamma=0.9, max_steps=2000, render=False):
        """ 
        Play one episode of CartPole
        
        :param epsilon: Parameter for epsilon greedy algorithm which determines the explore-exploit selection
        :param gamma: Discount rate for future rewards
        :param max_steps: Maximum iterations for an episode. The original 200 max steps of OpenAI Gym is too small
        :param render: Flag for rendering if needed
        """

        # Initialize metrics and state for episode
        s = self.env.reset()
        done = False
        total_rewards = 0
        step = 0

        while not done and step < max_steps:
            action = self.select_action(s, epsilon)
            s_prime, reward, done, _ = self.env.step(action)

            # Give a huge negative reward for finishing early
            if done:
                reward = -200

            # Get Q(s_prime,a_prime) for all actions in action space
            Q_values = self.model.predict(s_prime)
            G = reward + gamma * np.max(Q_values)

            # Train one epoch of the function approximator model
            self.model.update(s, action, G)

            if reward == 1:
                # Add rewards only of the non-terminal steps
                total_rewards += reward

            if render:
                # Render to show play if needed
                plt.imshow(self.env.render('rgb_array'))

            # Don't forget to update state
            # Or agent won't learn
            s = s_prime
            step += 1

        return total_rewards

    def select_action(self, s, epsilon):
        """
        Bandit algorithm to solve the exploration-exploitation dilemma. 
        Epsilon is the probability of choosing a random action.

        :param s: State parameter to choose action from pi(a|s)
        :param epsilon: Probability to choose action randomly for epsilon-greedy algorithm
        """
        if np.random.random() < epsilon:
            # Select random action for exploration
            return self.env.action_space.sample()
        else:
            # Select action from policy pi(a|s)
            return np.argmax(self.model.predict(s))

class PolicyGradientAgent:
    """ Agent that learns using Policy Gradient Method """

    def __init__(self, env, policy_model=None, value_model=None, optimizer=None, lr=1e-3):
        """ Constructor 
        
        :param env: Environment object with same interface as OpenAI gym CartPole
        :param policy_model: Function approximation model that gives pi(a|s)
        :param value_model: Function apporximation model that gives Q(s, a)
        """

        # Attach or create policy model pi(a|s)
        if policy_model:
            self.policy_model = policy_model
        else:
            self.policy_model = PolicyModel(env)

        # Attach or create value model Q(s, a)
        if value_model:
            self.value_model = value_model
        else:
            self.value_model = RBFRegressionModel(env)

        # Attach or create the optimizer
        if optimizer:
            self.optim = optimizer
        else:
            self.optim = optim.Adam(self.policy_model.parameters(), lr=lr)

    def play(self, env, gamma=0.9, max_steps=2000, render=False):
        """ 
        Play one episode of CartPole. This is a Monte Carlo Method.
        
        :param env: Environment object with same interface as OpenAI gym CartPole
        :param gamma: Discount rate for future rewards
        :param max_steps: Maximum iterations for an episode. The original 200 max steps of OpenAI Gym is too small
        :param render: Flag for rendering if needed
        """

        # Initialize metrics and state for episode
        s = env.reset()
        done = False
        # total_rewards = 0
        step = 0

        # Keep track of (s, a, r) for an episode
        states = []
        actions = []
        rewards = []

        while not done and step < max_steps:
            action = self.select_action(s)
            s_prime, reward, done, _ = env.step(action)

            if render:
                # Render to show play if needed
                plt.imshow(env.render('rgb_array'))

            # Record the episode
            states.append(s)
            actions.append(action)
            rewards.append(reward)

            # Don't forget to update state
            # Or agent won't learn
            s = s_prime
            step += 1

        # Collect the returns and difference from value model
        G = 0
        returns = []
        for r in rewards[::-1]:
            G = r + gamma * G
            returns.append(G)
        returns.reverse()
        
        loss = self.update_policy(states, actions, returns)

        return loss, sum(rewards)


    def select_action(self, obs):
        """ Select the action to take based on the policy pi(a|s) """
        action_logits = self.policy_model(obs)

        with torch.no_grad():
            action_probs = nn.functional.softmax(action_logits, -1)
        
        action_probs = action_probs.numpy()

        return np.random.choice(len(action_probs), p=action_probs)

    def update_policy(self, obs, actions, G):
        """
        Backpropagation update for policy model pi(a|s)

        :param states: States of an episode
        :type states: numpy.ndarray
        :param actions: Actions of an episode
        :type actions: numpy.ndarray
        :param G: Either the list of returns or the list of advantages (G - V(s))
        :type G: numpy.ndarray (numpy.float64)
        """
        # Setup tensors
        action_tensors = nn.functional.one_hot(torch.as_tensor(actions))
        G_tensors = torch.as_tensor(G, dtype=torch.float32)

        # Get trained policy distribution model pi(a|s)
        action_logits = self.policy_model(obs)
        action_logprob = (nn.functional.log_softmax(action_logits, -1) * action_tensors).sum(1)

        # Calculate loss
        loss = -((action_logprob * G_tensors).mean())

        # Optimize
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss