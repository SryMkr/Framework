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
class CollectorPlayer(CollectorAgentInterface):
    """select the prioritised words"""

    def __init__(self,
                 player_id,
                 player_name,
                 policy):
        super().__init__(player_id,
                         player_name,
                         policy)

    def step(self, time_step):
        action = []
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        review_words_number = time_step.observations["review_words_number"]
        history_information = time_step.observations["history_information"]
        # student_excellent_df = time_step.observations["student_excellent_memory"]
        # student_forget_df = time_step.observations["student_forget_memory"]

        if self._policy == 'random':  # randomly select tasks per day
            action = random.sample(legal_actions, review_words_number)

        elif self._policy == 'MAB':  # Multi-Arm Bandit algorithm
            pass
            # if history_information is None:
            #     action = random.sample(legal_actions, review_words_number)
            # else:
            #     action = random.sample(legal_actions, review_words_number)
            #     # 如何将feedback的信息和review的信息结合？
            #     # print(history_information)
            #     # print(legal_actions)
            #     # print(student_forget_df)
            #     # print(student_excellent_df)
            #     # print(student_excellent_df)
            # # history information will be used here
        return action


# student player


# student player
class StudentPlayer(StudentAgentInterface):
    def __init__(self,
                 player_id,
                 player_name,
                 excellent_memory_dataframe,
                 policy):
        super().__init__(player_id,
                         player_name,
                         excellent_memory_dataframe,
                         policy)

        self.policy = policy
        self.current_session_num = 0

    @staticmethod
    def forgetting_parameters(timing_steps, excel_start=0.9, excel_end=0.4, noise_start=0.1, noise_end=1, decay_rate=2):
        # the function aims to simulate the forgetting curve
        timing_points = np.linspace(0, 1, timing_steps)
        excel_list = (excel_start - excel_end) * np.exp(-decay_rate * timing_points) + excel_end
        noise_list = (noise_end - noise_start) * (1 - np.exp(-decay_rate * timing_points)) + noise_start
        return excel_list, noise_list

    @staticmethod
    def forget_process(unique_phonemes, excellent_dataframe, noise_dataframe, excel_ratio, random_ratio):
        # 直接改变一个session的记忆
        excellent_dataframe_copy = excellent_dataframe.copy()
        for pho in unique_phonemes:
            excellent_dataframe_copy.loc[pho] = excel_ratio * excellent_dataframe.loc[pho] + random_ratio * \
                                                noise_dataframe.loc[pho]
        result_df = excellent_dataframe_copy.div(excellent_dataframe_copy.sum(axis=1), axis=0)  # normalize
        return result_df

    def step(self, time_step):

        return self._random_memory_df, self._excellent_memory_df


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
