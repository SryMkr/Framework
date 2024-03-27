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

    @staticmethod
    def forgetting_parameters(timing_steps, excel_start=0.9, excel_end=0.4, noise_start=0.1, noise_end=1, decay_rate=2):
        """create the weight pair of excellent memory and noise"""
        timing_points = np.linspace(0, 1, timing_steps)
        excel_list = (excel_start - excel_end) * np.exp(-decay_rate * timing_points) + excel_end
        excel_list = np.insert(excel_list, 0, 1)
        noise_list = (noise_end - noise_start) * (1 - np.exp(-decay_rate * timing_points)) + noise_start
        noise_list = np.insert(noise_list, 0, 0)
        return excel_list, noise_list

    @staticmethod
    def forget_process(unique_phonemes, excellent_dataframe, noise_dataframe, excel_ratio, random_ratio):
        """ add the noise to simulate forgetting"""
        excellent_dataframe_copy = excellent_dataframe.copy()
        for pho in unique_phonemes:
            excellent_dataframe_copy.loc[pho] = excel_ratio * excellent_dataframe.loc[pho] + random_ratio * \
                                                noise_dataframe.loc[pho]
        result_df = excellent_dataframe_copy.div(excellent_dataframe_copy.sum(axis=1), axis=0)  # normalize
        return result_df

    @staticmethod
    def learn_process(unique_phonemes, forget_dataframe, excellent_dataframe, learning_rate=0.01):
        """ this function aims to enhance memory, the larger the learning rate, the better the retention"""
        forget_dataframe_copy = forget_dataframe.copy()
        for pho in unique_phonemes:
            forget_dataframe_copy.loc[pho] = forget_dataframe.loc[pho] + learning_rate * excellent_dataframe.loc[pho]
        result_df = forget_dataframe_copy.div(forget_dataframe_copy.sum(axis=1), axis=0)
        return result_df

    @staticmethod
    def add_position(history_words: List[List[str]]):
        """ add the noise to simulate forgetting"""
        split_phonemes = [pair[0].split(' ') for pair in history_words]
        position_phonemes = []
        for phonemes_list in split_phonemes:
            for index, value in enumerate(phonemes_list):
                position_phonemes.append(value + '_' + str(index))
        unique_phonemes = set(position_phonemes)
        return unique_phonemes

    def step(self, time_step):

        current_session_number = time_step.observations["current_session_num"]
        history_words = time_step.observations["history_words"]
        sessions_number = time_step.observations["sessions_number"]
        history_unique_phonemes = self.add_position(history_words)
        excel_list, noise_list = self.forgetting_parameters(sessions_number)
        self._forget_memory_df = self.forget_process(list(history_unique_phonemes), self._excellent_memory_df,
                                                     self._random_memory_df,
                                                     excel_list[current_session_number],
                                                     noise_list[current_session_number])
        # for learn memory
        current_session_words = time_step.observations["current_session_words"]
        tasks_unique_phonemes = self.add_position(current_session_words)
        self._learn_memory_df = self.learn_process(tasks_unique_phonemes, self._forget_memory_df,
                                                   self._excellent_memory_df)
        return self._random_memory_df, self._excellent_memory_df, self._forget_memory_df, self._learn_memory_df


class ExaminerPlayer(ExaminerAgentInterface):
    def __init__(self,
                 player_id,
                 player_name):
        super().__init__(player_id,
                         player_name)

        self.alphabet = string.ascii_lowercase

    def spelling(self, memory, history_words):
        for pair in history_words:
            spelling = []
            split_phoneme = pair[0].split(' ')
            split_letters = pair[1].split(' ')
            position_condition = []  # empty is every time
            for position, phoneme in enumerate(split_phoneme):
                position_condition.append(phoneme + '_' + str(position))
            for i in range(len(split_letters)):
                if i == 0:
                    result_columns = [al + '_' + str(i) for al in self.alphabet]
                    possible_results = memory.loc[position_condition[0], result_columns]
                    letter = possible_results.idxmax()
                else:
                    result_columns = [al + '_' + str(i) for al in self.alphabet]
                    possible_results = memory.loc[position_condition, result_columns]
                    letters_prob = possible_results.sum(axis=0)
                    letter = letters_prob.idxmax()
                spelling.append(letter)
            # 明天起来做比较把
            print(spelling)

    def step(self, time_step):
        student_memories = time_step.observations["student_memories"]
        history_words = time_step.observations["history_words"]
        for memory in student_memories:
            self.spelling(memory, history_words)
        # marks = []
        # actions = {}
        # answer = ''.join(time_step.observations["answer"].split(' '))  # 'b a t h' --> ['b', 'a', 't', 'h']
        # student_spelling = ''.join(time_step.observations["student_spelling"])  # ['f', 'h', 'v', 'q']
        # word_accuracy = round(Levenshtein.ratio(answer, student_spelling), 3)
        # answer_length = time_step.observations["answer_length"]
        # legal_actions = time_step.observations["legal_actions"][self.player_id]
        # for position in range(answer_length):
        #     if student_spelling[position] == answer[position]:
        #         marks.append(legal_actions[1])
        #     else:
        #         marks.append(legal_actions[0])
        # self.accuracy.append(word_accuracy)
        # # 在这里把位置加进去，然后和对错结合起来并组成一个元组
        # for position, letter in enumerate(student_spelling):
        #     actions[letter + '_' + str(position)] = marks[position]
        # return actions, word_accuracy
