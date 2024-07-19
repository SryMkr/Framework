"""
放弃MAB
define observation space, action space, reward function
1：如果只考虑音标对应的所有字母的概率分布的平均KL散度，the larger the kl 散度, the larger the memory difference.
2: 结论好像是平均KL散度小的的先训练，反而能够快速提高准确度？
2：接下来要考虑的是正确字母和错误字母如何使用
3：第一个实验，只考虑错误的，计算错误字母的散度，纠正是全部纠正

# 只考虑一步，做一次决策，那就是不合并，只考虑当前
"""

import numpy as np
from scipy.stats import entropy
from collections import Counter


def normalize(values):
    min_val = np.min(values)
    max_val = np.max(values)
    normalized_values = (values - min_val) / (max_val - min_val)
    return normalized_values


class MultiArmBandit:
    """ we see each word as the bandit"""

    def __init__(self, n_arms, observation, accuracy_dict, MAB_history):
        self.n_arms = n_arms  # the number of history words
        self.arm_counts = np.zeros(n_arms)  # the selected numbers when training
        self.arm_values = np.zeros(n_arms)  # the contribution of each task
        self.observation = observation  # the legal action
        self.feedback = accuracy_dict  # the feedback
        self.MAB_history = MAB_history  # record history information
        self.reward_distribution = {}

    def select_arm(self, epsilon):
        """leverage the ε-epsilon
        :param epsilon: prob of exploration
        :return: the index of arms
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_arms)  # randomly select
        else:
            return np.argmax(self.arm_values)  # exploiting, choose the maximum value

    def construct_information(self, excellent_dataframe, forget_dataframe):
        # 对所有的单词计算KL散度，准确度，音标长度，字母长度,如何综合考虑比只考虑准确度高呢？
        # 考虑错误的单词拼写么?
        position_phonemes = []
        position_letters = []
        KL_list = []
        new_information = {}
        for task in self.observation:
            for index, phoneme in enumerate(task[0].split(' ')):
                position_phonemes.append(phoneme + '_' + str(index))
            for index, letter in enumerate(task[1].split(' ')):
                position_letters.append(letter + '_' + str(index))
            positioned_task = (position_phonemes, position_letters)
            excellent_prob = excellent_dataframe.loc[positioned_task].values
            forget_prob = forget_dataframe.loc[positioned_task].values
            kl_entropy = [entropy(excellent_prob[i], forget_prob[i], base=2) for i in range(excellent_prob.shape[0])]
            total_entropy = np.sum(kl_entropy)
            KL_list.append(total_entropy)
        normalized_entropy = normalize(KL_list)
        for index in range(len(self.observation)):
            contribution = 0.1 * normalized_entropy[index] + 0.8 * self.feedback[tuple(self.observation[index])]
            # [task, contribution]
            new_information[tuple(self.observation[index])] = round(contribution, 3)

        if not self.MAB_history:
            self.MAB_history = new_information
        else:
            # 合并新旧的贡献,表示为任务和奖励分布
            for task, contributions in self.MAB_history.items():
                contribution = new_information[task]
                contributions.append(contribution[0])
                self.MAB_history[task] = contributions

    def reward_function(self, arm):
        """the relative entropy of two prob distribution
            在这里选择任务奖励奖励，新旧合并一次就够了，循环的只是选择动作
           要考虑准确度 (最重要)，拼写长度，音标长度，
           标准用最简单的特征概括大多数的任务
        """
        selected_word = tuple(self.observation[arm])
        for key, values in self.MAB_history.items():
            count = Counter(values)
            total = sum(count.values())
            probabilities = {k: v / total for k, v in count.items()}
            self.reward_distribution[key] = probabilities
        reward_distribution = self.reward_distribution[selected_word]
        keys, probs = zip(*reward_distribution.items())
        reward = np.random.choice(keys, p=probs)
        return reward

    def update(self, chosen_arm, reward):
        """ update reward
        :param chosen_arm: the index of arms
        :param reward: reward
        """
        self.arm_counts[chosen_arm] += 1
        n = self.arm_counts[chosen_arm]
        value = self.arm_values[chosen_arm]
        new_value = value + (reward - value) / n
        self.arm_values[chosen_arm] = new_value

    def train_MAB(self, excellent_memory_df, forget_memory_df, epoch=300, exploration_rate=0.6):
        """ train the MAB model"""
        self.construct_information(excellent_memory_df, forget_memory_df)
        for _ in range(epoch):
            selected_arm = self.select_arm(exploration_rate)
            selected_arm_reward = self.reward_function(selected_arm)
            self.update(selected_arm, selected_arm_reward)
