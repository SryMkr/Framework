"""
1: four agents instance

Tasks:
(1) simulate forgetting students
(2) student learn from feedback
(3) optimise all agents
2/21 implement the multi armed bandit into session collector player
step 1: calculate the KL divergence between wrong letter and excellent
step 2: sort by descending
step 3：the order will be the top 50 words
既然分组了，如何添加噪声，很简单，按照session的个数，把噪声分组，那么对应每一组搁的时间代表了那一组的遗忘，不是全部添加而是只添加那几个单词
新的单词，按照现在有方式进行软更新，看看最后的结果能不能抵抗遗忘？
至于老师推荐单词的话，那是后期的任务了，主要是为了一个平衡，对于每一个推荐的单词，准确度都要是当前组里最高的
"""
from itertools import chain

import os
import random
import string
import Levenshtein
from agents_interface import *
import pandas as pd
import torch
import numpy as np


# TaskCollector Agent
class SessionCollectorPlayer(SessionCollectorInterface):
    """select the prioritised words"""

    def __init__(self,
                 player_id,
                 player_name,
                 policy):
        super().__init__(player_id,
                         player_name,
                         policy)

    def step(self, time_step):
        """
                    :return: the words need to be reviewed
                    """
        action = []
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        review_words_number = time_step.observations["review_words_number"]
        history_information = time_step.observations["history_information"]
        student_excellent_df = time_step.observations["student_excellent_memory"]
        student_forget_df = time_step.observations["student_forget_memory"]

        if self._policy == 'random':  # randomly select the words to be reviewed
            action = random.sample(legal_actions, review_words_number)
            # 接下来我的问题是 如何讲这个历史信息和review信息结合，通过MAB选择一个最合适下一次练习的10个单词
            # 以及作图，搞出一个评价标准
        elif self._policy == 'MAB':  # Multi-Arm Bandit algorithm
            if history_information is None:
                action = random.sample(legal_actions, review_words_number)
            else:
                action = random.sample(legal_actions, review_words_number)
                # 如何将feedback的信息和review的信息结合？
                # print(history_information)
                # print(legal_actions)
                # print(student_forget_df)
                # print(student_excellent_df)
                # print(student_excellent_df)
            # history information will be used here
        return action


# PresentWord Agent
class PresentWordPlayer(PresentWordInterface):
    """"""

    def __init__(self,
                 player_id,
                 player_name,
                 policy):
        super().__init__(player_id,
                         player_name,
                         policy)

    def define_difficulty(self, time_step) -> Dict[tuple, int]:
        """tutor agent define the difficulty of each task
        difficulty definition: the length of word
        """
        task_difficulty = {}
        legal_actions: List[str] = time_step.observations["legal_actions"][self.player_id]
        for task in legal_actions:
            task_difficulty[tuple(task)] = len(''.join(task[1].split(' ')))
        return task_difficulty

    def action_policy(self, task_difficulty):
        action = ''  # action is empty string
        if self._policy == 'random':
            action = random.choice(list(task_difficulty.keys()))
        if self._policy == 'sequential':
            action = next(iter(task_difficulty.items()))[0]
        if self._policy == 'easy_to_hard':
            action = min(task_difficulty.items(), key=lambda x: x[1])[0]
        if self._policy == 'DDA':  # 可做可不做，这可以是第二篇文章
            pass
        return list(action)

    def step(self, time_step):
        task_difficulty = self.define_difficulty(time_step)
        action = self.action_policy(task_difficulty)
        return action


# student player




# student player
class StudentPlayer(StudentInterface):
    def __init__(self,
                 player_id,
                 player_name,
                 policy):
        super().__init__(player_id,
                         player_name,
                         policy)

        CURRENT_PATH = os.getcwd()  # get the current path
        self.policy = policy
        STU_MEMORY_PATH = os.path.join(CURRENT_PATH, 'agent_RL/excellent_memory.xlsx')
        self.stu_forget_df = pd.read_excel(STU_MEMORY_PATH, index_col=0, header=0)  # for forgetting

        self.stu_excellent_df = pd.read_excel(STU_MEMORY_PATH, index_col=0, header=0)  # excellent students memory
        self.stu_memory_tensor = torch.tensor(self.stu_excellent_df.values,
                                              dtype=torch.float32)  # the shape of distribution
        self.noise = torch.randn_like(self.stu_memory_tensor)  # generate the noise
        self.scaled_noise = (self.noise - self.noise.min()) / (self.noise.max() - self.noise.min())
        self.scaled_noise_df = pd.DataFrame(self.scaled_noise.numpy(), index=self.stu_excellent_df.index,
                                            columns=self.stu_excellent_df.columns)
        self.current_session_num = 0

    @staticmethod
    def forgetting_parameters(timing_steps, excel_start=0.9, excel_end=0.4, noise_start=0.1, noise_end=1, decay_rate=2):
        #the function aims to simulate the forgetting curve
        timing_points = np.linspace(0, 1, timing_steps)
        excel_list = (excel_start - excel_end) * np.exp(-decay_rate * timing_points) + excel_end
        noise_list = (noise_end - noise_start) * (1 - np.exp(-decay_rate * timing_points)) + noise_start
        return excel_list, noise_list

    @staticmethod
    def forget_process(unique_phonemes, excellent_dataframe, noise_dataframe, excel_ratio, random_ratio):
        #直接改变一个session的记忆
        excellent_dataframe_copy = excellent_dataframe.copy()
        for pho in unique_phonemes:
            excellent_dataframe_copy.loc[pho] = excel_ratio * excellent_dataframe.loc[pho] + random_ratio * \
                                                noise_dataframe.loc[pho]
        result_df = excellent_dataframe_copy.div(excellent_dataframe_copy.sum(axis=1), axis=0)  # normalize
        return result_df

    def stu_spell(self, time_step):
        #   three types of student agent: (1) random, (2) excellent, (3) forget :return: student spelling
        actions = []
        sessions_num = time_step.observations["sessions_number"]
        excel_list, noise_list = self.forgetting_parameters(sessions_num)
        condition = time_step.observations["condition"].split(' ')  # phonemes
        answer_length = time_step.observations["answer_length"]
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        alphabet = string.ascii_lowercase

        if (time_step.observations["current_session_num"] - self.current_session_num == 2) and self.policy == "forget":
            # 每一天所有的单词都会忘记一次，其他两个agent只是决定如何加强某个单词的
            history_words = time_step.observations["history_words"]
            # 要给每一个phoneme加上标签再分割
            split_phonemes = [corpus[0].split(' ') for corpus in history_words]
            position_phonemes = []
            for phonemes_list in split_phonemes:
                for index, value in enumerate(phonemes_list):
                    position_phonemes.append(value + '_' + str(index))

            unique_phonemes = set(position_phonemes)

            self.stu_forget_df = self.forget_process(list(unique_phonemes), self.stu_excellent_df, self.scaled_noise_df,
                                                     excel_list[self.current_session_num],
                                                     noise_list[self.current_session_num])
        self.current_session_num = time_step.observations["current_session_num"] - 1

        if self._policy == 'random':
            for letter_index in range(answer_length):
                selected_index = random.choice(legal_actions)
                actions.append(selected_index)

        elif self._policy == 'excellent':
            # use maximum expectation algorithm
            spelling = []  # store the letter_position
            self.position_condition = []  # empty is every time
            for position, phoneme in enumerate(condition):
                self.position_condition.append(phoneme + '_' + str(position))
            for i in range(answer_length):
                if i == 0:
                    result_columns = [al + '_' + str(i) for al in alphabet]
                    possible_results = self.stu_excellent_df.loc[self.position_condition[0], result_columns]
                    letter = possible_results.idxmax()
                else:
                    result_columns = [al + '_' + str(i) for al in alphabet]
                    possible_results = self.stu_excellent_df.loc[self.position_condition, result_columns]
                    letters_prob = possible_results.sum(axis=0)  # 每一列相加,取概率最大值
                    letter = letters_prob.idxmax()
                spelling.append(letter)

            for letter_position in spelling:
                actions.append(alphabet.index(letter_position.split('_')[0]))

        elif self._policy == 'forget':
            # 每一轮，按照历史记录添加噪声，代表每一轮之后都会进行遗忘操作
            spelling = []  # store the letter_position
            self.position_condition = []  # empty is every time
            for position, phoneme in enumerate(condition):
                self.position_condition.append(phoneme + '_' + str(position))

            for i in range(answer_length):
                if i == 0:
                    result_columns = [al + '_' + str(i) for al in alphabet]
                    possible_results = self.stu_forget_df.loc[self.position_condition[0], result_columns]
                    letter = possible_results.idxmax()
                else:
                    result_columns = [al + '_' + str(i) for al in alphabet]
                    possible_results = self.stu_forget_df.loc[self.position_condition, result_columns]
                    letters_prob = possible_results.sum(axis=0)  # 每一列相加,取概率最大值
                    letter = letters_prob.idxmax()
                spelling.append(letter)
            for letter_position in spelling:
                actions.append(alphabet.index(letter_position.split('_')[0]))

        return actions

    def stu_learn(self, time_step) -> None:
        # update the forgetting matrix by soft updates
        pass

    def step(self, time_step):
        # self.stu_learn(time_step)
        actions = self.stu_spell(time_step)
        return actions, self.stu_forget_df


class ExaminerPlayer(ExaminerInterface):
    def __init__(self,
                 player_id,
                 player_name):
        super().__init__(player_id,
                         player_name)
        self.accuracy = []

    def step(self, time_step):
        marks = []
        actions = {}
        answer = ''.join(time_step.observations["answer"].split(' '))  # 'b a t h' --> ['b', 'a', 't', 'h']
        student_spelling = ''.join(time_step.observations["student_spelling"])  # ['f', 'h', 'v', 'q']
        word_accuracy = round(Levenshtein.ratio(answer, student_spelling), 3)
        answer_length = time_step.observations["answer_length"]
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        for position in range(answer_length):
            if student_spelling[position] == answer[position]:
                marks.append(legal_actions[1])
            else:
                marks.append(legal_actions[0])
        self.accuracy.append(word_accuracy)
        # 在这里把位置加进去，然后和对错结合起来并组成一个元组
        for position, letter in enumerate(student_spelling):
            actions[letter + '_' + str(position)] = marks[position]
        return actions, word_accuracy

