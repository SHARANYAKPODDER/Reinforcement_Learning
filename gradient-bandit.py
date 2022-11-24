"""
Environment for the multi-armed bandit problem
"""
import numpy as np
import matplotlib.pyplot as plt	

class gradient_bandit_env:
    """
    Initialize the multi-arm bandit environment.

    :params:
    r_mean: takes a list of reward mean
    r_stddev: takes a list of reward standard deviation
    epsilon: takes a particular epsilon
    iters: takes number of iterations
    """

    def __init__(self, r_mean, r_stddev, iters, alpha):
        if len(r_mean) != len(r_stddev):
            raise ValueError(
                "Reward distribution parameters (mean and variance) must be of the same length"
            )

        if any(r <= 0 for r in r_stddev):
            raise ValueError("Standard deviation in rewards must all be greater than 0")

        self.n = len(r_mean)
        self.r_mean = r_mean
        self.r_stddev = r_stddev
        self.actions = np.arange(self.n)
        # Number of iterations
        self.iters = iters
        # Step count
        self.step = 1
        # Step count for each arm
        self.k_step = np.ones(self.n)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.zeros(self.n)
        # Initialize preferences
        self.H = np.zeros(self.n)
        # Learning rate
        self.alpha = alpha
    
    def softmax(self):
        self.prob_action = np.exp(self.H - np.max(self.H)) \
            / np.sum(np.exp(self.H - np.max(self.H)), axis=0)

    def pull(self, index_arm):
        """
        Performs the action of pulling the arm/lever of the selected bandit

        :inputs:
        index_arm: the index of the arm/level to be pulled

        :outputs:
        reward: the reward obtained by pulling tht arm (sampled from their corresponding Gaussian distribution)
        """
        reward = np.random.normal(self.r_mean[index_arm], self.r_stddev[index_arm])
        return reward

    def choose(self):
        """
        Performs the choosing of action
        """
        # Update probabilities
        self.softmax()
        # Select highest preference action
        a = np.random.choice(self.actions, p=self.prob_action)
        return a

    def update(self, index_arm):
        """
        Updates the reward, step and arm_index's step
        """
        # Update counts
        self.step += 1
        self.k_step[index_arm] += 1
        reward = self.pull(index_arm)
         
        # Update total
        self.mean_reward = self.mean_reward + (
            reward - self.mean_reward) / self.n
         
        # Update results for a_k
        self.k_reward[index_arm] = self.k_reward[index_arm] + (
            reward - self.k_reward[index_arm]) / self.k_step[index_arm]
         
        # Update preferences
        self.H[index_arm] = self.H[index_arm] + \
            self.alpha * (reward - self.mean_reward) * (1 -
                self.prob_action[index_arm])
        actions_not_taken = self.actions!=index_arm
        self.H[actions_not_taken] = self.H[actions_not_taken] - \
            self.alpha * (reward - self.mean_reward) * self.prob_action[actions_not_taken]

    def run(self):
        for i in range(self.iters):
            index_arm = self.choose()
            self.update(index_arm)
            self.reward[i] = self.mean_reward

    def reset(self, mean, std):
        # Resets results while keeping settings
        self.step = 0
        self.n = len(mean)
        self.k_step = np.zeros(self.n)
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        self.k_reward = np.zeros(self.n)
        self.H = np.zeros(self.n)
        self.r_mean = mean
        self.r_std = std

if __name__ == "__main__":
    # driver code
    true_means = [2.5, -3.5, 1.0, 5.0, -2.5]
    true_stds = [0.33, 1.0, 0.66, 1.98, 1.65]
    iters = 1000
    grad_rewards = np.zeros(iters)
    opt_grad=0
    episodes = 1000
    # Initialize bandits
    grad_bandit = gradient_bandit_env(true_means, true_stds, iters, 0.1)
    # Run experiments
    for i in range(episodes):
        # Reset counts and rewards
        grad_bandit.reset(true_means, true_stds)
        # Run experiments
        grad_bandit.run()
        # Update long-term averages
        grad_rewards = grad_rewards + (grad_bandit.reward - grad_rewards) / (i + 1)

    plt.figure(figsize=(12, 8))
    plt.plot(grad_rewards, label="$alpha=0.1$")
    plt.legend(loc="best")
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("Average $\epsilon-greedy$ Rewards after " + str(episodes) + " Episodes")
    plt.show()
