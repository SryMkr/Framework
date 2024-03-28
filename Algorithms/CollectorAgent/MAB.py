"""
define observation space, action space, reward function
1：如果只考虑音标对应的所有字母的概率分布的平均KL散度，the larger the kl 散度, the larger the memory difference.
2: 结论好像是平均KL散度小的的先训练，反而能够快速提高准确度？
2：接下来要考虑的是正确字母和错误字母如何使用
3：第一个实验，只考虑错误的，计算错误字母的散度，纠正是全部纠正

# 第一步先使用最简单的办法，就是每一轮的feedback，把learn的读取出来然后计算KL散度，选择前10个让其学习
"""

import Levenshtein
import string
import numpy as np
from scipy.stats import entropy


class MultiArmBandit:
    """ we see each word as the bandit"""

    def __init__(self, n_arms, observation):
        self.n_arms = n_arms  # the number of history words
        self.arm_counts = np.zeros(n_arms)  # the selected numbers when training
        self.arm_values = np.zeros(n_arms)  # the value of each task
        self.observation = observation  # the history information

    def select_arm(self, epsilon):
        """leverage the ε-epsilon
        :param epsilon: prob of exploration
        :return: the index of arms
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_arms)  # randomly select
        else:
            return np.argmax(self.arm_values)  # exploiting, choose the maximum value

    def reward_function(self, arm, excellent_dataframe, forget_dataframe):
        """the relative entropy of two prob distribution
           要考虑准确度，完整度，拼写长度，音标长度，
           标准用最简单的特征概括大多数的任务
        """
        # method 1； whole KL entropy
        position_phonemes = []
        total_entropy = 0

        current_observation = self.observation[arm][0].split(' ')

        for position, phoneme in enumerate(current_observation):
            position_phonemes.append(phoneme + '_' + str(position))
        for position_phoneme in position_phonemes:
            excellent_prob = excellent_dataframe.loc[position_phoneme].values
            forget_prob = forget_dataframe.loc[position_phoneme].values
            total_entropy += entropy(excellent_prob, forget_prob, base=2)
        reward = total_entropy / len(position_phonemes)

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

    def train_MAB(self, excellent_memory_df, forget_memory_df, epoch=200, exploration_rate=0.5):
        """ train the MAB model"""
        for _ in range(epoch):
            selected_arm = self.select_arm(exploration_rate)
            selected_arm_reward = self.reward_function(selected_arm, excellent_memory_df, forget_memory_df)
            self.update(selected_arm, selected_arm_reward)


class evaluate_improvement:
    def __init__(self, memory, corpus):
        self.memory = memory
        self.corpus = corpus
        self.student_answer_pair = []
        self.accuracy = []
        self.completeness = []
        self.perfect = []
        self.avg_accuracy = []
        self.avg_completeness = []
        self.avg_perfect = []

    def generate_answer(self):
        """ generate answer based on the given phonemes,而且我要知道答案的长度，然后根据所有的音标对每一个位置选择最大值"""
        for phonemes, answer in self.corpus:
            phonemes = phonemes.split(' ')
            answer = answer.split(' ')
            spelling = []
            answer_length = len(answer)
            alphabet = string.ascii_lowercase
            for i in range(answer_length):
                # 将26个字母和位置结合起来，组成列索引
                if i == 0:
                    result_columns = [al + '_' + str(i) for al in alphabet]
                    possible_results = self.memory.loc[phonemes[0], result_columns]
                    letter = possible_results.idxmax()
                else:
                    result_columns = [al + '_' + str(i) for al in alphabet]
                    possible_results = self.memory.loc[phonemes, result_columns]
                    letters_prob = possible_results.sum(axis=0)  # 每一列相加,取概率最大值
                    letter = letters_prob.idxmax()
                spelling.append(letter)
            self.student_answer_pair.append([spelling, answer])

    def evaluation(self):
        for stu_answer, correct_answer in self.student_answer_pair:
            stu_answer = ''.join([i.split('_')[0] for i in stu_answer])
            correct_answer = ''.join([i.split('_')[0] for i in correct_answer])
            word_accuracy = round(Levenshtein.ratio(correct_answer, stu_answer), 2)
            word_completeness = round(1 - Levenshtein.distance(correct_answer, stu_answer) / len(correct_answer), 2)
            word_perfect = 0.0
            if stu_answer == correct_answer:
                word_perfect = 1.0
            self.accuracy.append(word_accuracy)
            self.completeness.append(word_completeness)
            self.perfect.append(word_perfect)
        self.avg_accuracy = sum(self.accuracy) / len(self.accuracy)
        self.avg_completeness = sum(self.completeness) / len(self.completeness)
        self.avg_perfect = sum(self.perfect) / len(self.perfect)
        return self.avg_accuracy, self.avg_completeness, self.avg_perfect


if __name__ == '__main__':

    # 输出每个臂被选择的次数和估计值
    for i in range(len(selected_items)):
        task_value[selected_items[i][0]] = (bandit.arm_values[i], bandit.accuracy[i])
    # 按照价值大小排序，则排好的顺序就是应该记忆的顺序,这种判定方式容易选择长的词
    sorted_dict = dict(sorted(task_value.items(), key=lambda item: item[1], reverse=True))
    print(sorted_dict)
