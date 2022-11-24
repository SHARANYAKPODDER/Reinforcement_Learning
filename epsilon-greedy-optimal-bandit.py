"""
Environment for the multi-armed bandit problem
"""
import numpy as np
import matplotlib.pyplot as plt


class epsilon_bandit_env:
    """
    Initialize the multi-arm bandit environment.

    :params:
    r_mean: takes a list of reward mean
    r_stddev: takes a list of reward standard deviation
    epsilon: takes a particular epsilon
    iters: takes number of iterations
    """

    def __init__(self, r_mean, r_stddev, epsilon, iters):
        if len(r_mean) != len(r_stddev):
            raise ValueError(
                "Reward distribution parameters (mean and variance) must be of the same length"
            )

        if any(r <= 0 for r in r_stddev):
            raise ValueError("Standard deviation in rewards must all be greater than 0")

        self.n = len(r_mean)
        self.r_mean = r_mean
        self.r_stddev = r_stddev
        self.step = 0  # step count
        self.epsilon = epsilon
        self.iters = iters
        # Step count for each arm
        self.k_step = np.zeros(self.n)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.full(self.n, 5)

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
        p = np.random.uniform()
        if self.epsilon == 0 and self.step == 0:
            a = np.random.choice(self.n)
        elif p <= self.epsilon:
            # Randomly select an action
            a = np.random.choice(self.n)
        else:
            # Take greedy action
            a = np.argmax(self.k_reward)
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
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.step

        # Update results for index_arm
        self.k_reward[index_arm] = (
            self.k_reward[index_arm]
            + (reward - self.k_reward[index_arm]) / self.k_step[index_arm]
        )

    def run(self):
        for i in range(self.iters):
            index_arm = self.choose()
            self.update(index_arm)
            self.reward[i] = self.mean_reward


if __name__ == "__main__":
    # driver code
    true_means = [2.5, -3.5, 1.0, 5.0, -2.5]
    true_stds = [0.33, 1.0, 0.66, 1.98, 1.65]
    iters = 1000
    # Run experiments
    # Initialize bandits
    eps_01_bandit = epsilon_bandit_env(true_means, true_stds, 0.0, iters)
    # Run experiments
    eps_01_bandit.run()

    plt.figure(figsize=(12, 8))
    plt.plot(eps_01_bandit.reward, label="$\epsilon=0.01$")
    plt.legend(loc="best")
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("Average $\epsilon-greedy$ Rewards after " + str(iters) + " Ierations")
    plt.show()
