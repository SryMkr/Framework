"""
1: four agents instance
Tasks:
step 1: calculate the KL divergence between wrong letter and excellent
step 2: sort by descending
step 3：the order will be the top 50 words

"""

import random
import string
import Levenshtein
from agents_interface import *
import numpy as np
from Algorithms.CollectorAgent.MAB import MultiArmBandit
from operator import itemgetter


# TaskCollector Agent
class CollectorPlayer(CollectorAgentInterface):
    """select the prioritised words"""

    def __init__(self,
                 player_id,
                 player_name,
                 policies):
        super().__init__(player_id,
                         player_name,
                         policies)

    def step(self, time_step):

        legal_actions = time_step.observations["legal_actions"][self.player_id]
        review_words_number = time_step.observations["review_words_number"]
        history_information = time_step.observations["history_information"]

        for policy in self._policies:
            if policy == 'random_collector':  # randomly select tasks per day
                action = random.sample(legal_actions, review_words_number)
                self._actions[policy] = action
            elif policy == 'MAB':  # Multi-Arm Bandit algorithm
                if history_information is None:
                    action = random.sample(legal_actions, review_words_number)
                    self._actions[policy] = action
                else:
                    _, student_excellent_memory, _, student_learn_memory = time_step.observations["student_memories"]
                    bandit = MultiArmBandit(len(legal_actions), legal_actions)
                    bandit.train_MAB(student_excellent_memory['excellent'], student_learn_memory['MAB'])
                    words_value_pair = dict(zip(map(tuple, legal_actions), bandit.arm_values))
                    sorted_pair = sorted(words_value_pair.items(), key=itemgetter(1), reverse=True)
                    action = [list(item[0]) for item in sorted_pair[-review_words_number:]]
                    self._actions[policy] = action
                    # how to use the feedback form examiner？
        return self._actions


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
        self.history_unique_phonemes = ()
        self.excel_list = None
        self.noise_list = None

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
        if current_session_number == 0:
            history_words = time_step.observations["history_words"]
            sessions_number = time_step.observations["sessions_number"]
            self.history_unique_phonemes = self.add_position(history_words)
            self.excel_list, self.noise_list = self.forgetting_parameters(sessions_number)
        self._forget_memory_df['forget'] = self.forget_process(list(self.history_unique_phonemes),
                                                               self._excellent_memory_df['excellent'],
                                                               self._random_memory_df['random_memory'],
                                                               self.excel_list[current_session_number],
                                                               self.noise_list[current_session_number])

        # for learn memory
        current_session_words = time_step.observations["current_session_words"]
        for policy, tasks in current_session_words.items():
            tasks_unique_phonemes = self.add_position(tasks)
            self._learn_memory_df[policy] = self.learn_process(tasks_unique_phonemes, self._forget_memory_df['forget'],
                                                               self._excellent_memory_df['excellent'])
        return self._random_memory_df, self._excellent_memory_df, self._forget_memory_df, self._learn_memory_df


class ExaminerPlayer(ExaminerAgentInterface):
    def __init__(self,
                 player_id,
                 player_name):
        super().__init__(player_id,
                         player_name)

        self.alphabet = string.ascii_lowercase

    def evaluation(self, memory, history_words):
        word_accuracy_list = []
        pair_feedback = {}
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
                spelling.append(letter.split('_')[0])

            student_spelling = ''.join(spelling)
            correct_spelling = ''.join(split_letters)

            # calculate the similarity of pair
            word_accuracy = round(Levenshtein.ratio(correct_spelling, student_spelling), 2)
            # mark each letter
            letters_mark = [x + '_' + str(1) if x == y else x + '_' + str(0) for x, y in
                            zip(student_spelling, correct_spelling)]
            pair_feedback[tuple(pair)] = (letters_mark, word_accuracy)

            word_accuracy_list.append(word_accuracy)
        average_accuracy = round(sum(word_accuracy_list) / len(word_accuracy_list), 2)
        return pair_feedback, average_accuracy

    def step(self, time_step):
        self._examiner_feedback = dict()
        student_memories = time_step.observations["student_memories"]
        history_words = time_step.observations["history_words"]
        for memory in student_memories:
            for policy, mem_df in memory.items():
                examiner_feedback = self.evaluation(mem_df, history_words)
                self._examiner_feedback[policy] = examiner_feedback
        return self._examiner_feedback
