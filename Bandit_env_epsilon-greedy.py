"""

Environment for the 5-armed bandit problem

"""

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt

class bandit_env_epsilon_greedy():
    """
    
    Initialize the 5-arm bandit environment.

    :params:
    r_mean: takes a list of reward mean
    r_stddev: takes a list of reward standard deviation
    eps = value of the epsilon
    step = the number of the iterations

    """
    def __init__(self, k, r_mean, r_stddev, eps, step):
        if len(r_mean) != len(r_stddev):
            raise ValueError("Reward distribution parameters (mean and variance) must be of the same length")

        if any(r <= 0 for r in r_stddev):
            raise ValueError("Standard deviation in rewards must all be greater than 0")

        self.len = len(r_mean)
        self.k = k # The number of bandits/ lever
        self.r_mean = r_mean # the true reward values
        self.r_stddev = r_stddev # the deviation around the mean
        self.eps = eps # value of the epsilon
        self.step = step # Number of iterations
        self.n = 0 # Step count
        self.k_n = np.zeros(k) # Step count for each arm
        self.mean_reward = 0 # Total mean reward
        self.reward = np.zeros(step)
        self.k_reward = np.zeros(k) # Mean reward for each arm

    def selection(self):
        """
        
        This selects a process for the epsilon-greedy algorithm

        :outputs:
        a: the choosen action
        
        """
        p = np.random.rand()
        if self.eps == 0 and self.step == 0:
            a = np.random.choice(self.len)
        elif p <= self.eps:
            # Random selction of an action
            a = np.random.choice(self.len)
        else:
            # Selection of a greedy action
            a = np.argmax(self.k_reward)
        return a

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

    def update(self, index_arm):
        """
        
        Updates the count, mean reward, and result for a_k
        :inputs:
        index_arm: the index of the arm/level to be pulled
        
        """
        # Update counts
        self.n += 1
        self.k_n[index_arm] += 1
        reward = self.pull(index_arm)
        # Update total
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n
        
        # Update results for a_k
        self.k_reward[index_arm] = self.k_reward[index_arm] + (reward - self.k_reward[index_arm]) / self.k_n[index_arm]

    def run(self):
        for i in range(step):
            index_arm = self.selection()
            self.update(index_arm)
            self.reward[i] = self.mean_reward

if __name__ == "__main__":
    r_mean = [2.5, -3.5, 1.0, 5.0, -2.5]
    r_stddev = [0.33, 1.0, 0.66, 1.98, 1.65]
    k=5
    step = 1000
    eps_0_rewards = np.zeros(step)
    eps_01_rewards = np.zeros(step)
    eps_1_rewards = np.zeros(step)
    episodes = 1000

    # Run experiments
    for i in range(episodes):
        # Initialize bandits
        eps_0 = bandit_env_epsilon_greedy(k, r_mean, r_stddev, 0, step)
        eps_01 = bandit_env_epsilon_greedy(k, r_mean, r_stddev, 0.01, step)
        eps_1 = bandit_env_epsilon_greedy(k, r_mean, r_stddev, 0.1, step)
        
        # Run experiments
        eps_0.run()
        eps_01.run()
        eps_1.run()
        
        # Update long-term averages
        eps_0_rewards = eps_0_rewards + (eps_0.reward - eps_0_rewards) / (i + 1)

        eps_01_rewards = eps_01_rewards + (eps_01.reward - eps_01_rewards) / (i + 1)

        eps_1_rewards = eps_1_rewards + (eps_1.reward - eps_1_rewards) / (i + 1)
    
    plt.figure(figsize=(12,8))
    plt.plot(eps_0_rewards, label="$\epsilon=0$ (greedy)")
    plt.plot(eps_01_rewards, label="$\epsilon=0.01$")
    plt.plot(eps_1_rewards, label="$\epsilon=0.1$")
    plt.legend()
    plt.xlabel("TimeStep")
    plt.ylabel("Average Reward")
    plt.title("Average $\epsilon-greedy$ Rewards after " + str(episodes)+ " Episodes")
    plt.show()