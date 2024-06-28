"""
1: four agents instance
Tasks:
step 1: calculate the KL divergence between wrong letter and excellent
step 2: sort by descending
step 3：the order will be the top 50 words

按照现在的设定是
 (1) 学习率越高曲线越高
 (2)  学习的单词越多，提升越高
 (3) 学的单词越多，不管什么方式表现出的结果都差不多

所以可以得出结论
即使使用任何算法来挑选单词，都无法展现出非常大的学习效果
那我之只能证明一点，即，MAB每一轮所选择的单词的准确度都比
"""

import random
import string
from collections import Counter
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
        current_session_number = time_step.observations["current_session_num"]
        for policy in self._policies:
            if policy == 'random_collector':  # randomly select tasks per day
                action = random.sample(legal_actions, review_words_number)
                self._actions[policy] = action
            elif policy == 'longest_collector':  # 该策略每次都选择最长的前N个单词
                sorted_desc = sorted(legal_actions, key=lambda x: len(''.join(x[0].split(' '))), reverse=True)
                action = sorted_desc[:review_words_number]
                self._actions[policy] = action
            elif policy == 'shortest_collector':  # 该策略每次都选择最短的前N个单词
                sorted_asc = sorted(legal_actions, key=lambda x: len(''.join(x[0].split(' '))))
                action = sorted_asc[:review_words_number]
                self._actions[policy] = action
            elif policy == 'MAB':  # Multi-Arm Bandit algorithm
                if history_information is None:
                    action = random.sample(legal_actions, review_words_number)
                    self._actions[policy] = action
                else:
                    current_examiner_feedback = history_information[current_session_number - 1]
                    forget_feedback = current_examiner_feedback['forget']
                    # 锁定每一个任务所影响的区域,计算每个单词对纠正的共享
                    task_impact = {}
                    wrong_area_counter = []
                    for task, feedback in forget_feedback[0].items():
                        wrong_area_list = []
                        word = ''.join(task[1].split(' '))
                        wrong_area = 0
                        for letter in feedback[0]:
                            l, m = letter.split('_')
                            if m == '0':
                                wrong_area_list.append(word[wrong_area] + '_' + str(wrong_area))
                                wrong_area_counter.append(word[wrong_area] + '_' + str(wrong_area))
                            wrong_area += 1
                        task_impact[task] = wrong_area_list

                    wrong_area_total_counter = Counter(wrong_area_counter)
                    print(wrong_area_total_counter)
                    word_correction_scores = {}

                    for words, positions in task_impact.items():
                        if len(positions) != 0:
                            total_correction_score = sum(wrong_area_total_counter[pos] for pos in positions)
                        else:
                            total_correction_score = 0
                        word_correction_scores[words] = total_correction_score
                    print(word_correction_scores)
                    sorted_word_correction_scores = sorted(word_correction_scores.items(), key=lambda item: item[1],
                                                           reverse=True)
                    # 如果某个单词已经被选择过了，那么其影响力应该随之下降一位
                    # print(wrong_area_total_counter)
                    print(sorted_word_correction_scores)
                    # action = [list(words) for words, score in sorted_word_correction_scores[:review_words_number]]
                    action = [list(words) for words, score in sorted_word_correction_scores[-review_words_number:]]
                    self._actions[policy] = action

                    # _, student_excellent_memory, student_forget_memory, student_learn_memory = time_step.observations["student_memories"]
                    # bandit = MultiArmBandit(len(legal_actions), legal_actions)
                    # # 根据当前的遗忘记忆，肯定是有学生对于遗忘记忆的单词的反馈
                    # bandit.train_MAB(student_excellent_memory['excellent'], student_forget_memory['forget'])
                    # words_value_pair = dict(zip(map(tuple, legal_actions), bandit.arm_values))
                    # sorted_pair = sorted(words_value_pair.items(), key=itemgetter(1), reverse=True)  # 降序
                    # action = [list(item[0]) for item in sorted_pair[-review_words_number:]]  # 取最小的
                    # # action = [list(item[0]) for item in sorted_pair[:review_words_number]]  # 取最大的
                    # self._actions[policy] = action
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
    def forgetting_parameters(timing_steps, excel_start=0.9, excel_end=0.05, noise_start=0.05, noise_end=1,
                              decay_rate=5):
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
    def learn_process(positioned_tasks, forget_dataframe, excellent_dataframe, learning_rate=0.5):
        """ this function aims to enhance memory, the larger the learning rate, the better the retention"""
        forget_dataframe_copy = forget_dataframe.copy()
        for positioned_task in positioned_tasks:
            forget_dataframe_copy.loc[positioned_task] = forget_dataframe.loc[positioned_task] + \
                                                         learning_rate * excellent_dataframe.loc[
                                                             positioned_task]
        result_df = forget_dataframe_copy.div(forget_dataframe_copy.sum(axis=1), axis=0)
        return result_df

    @staticmethod
    def add_position(history_words: List[List[str]]):
        """ add position to phonemes and tasks"""
        positioned_task = []
        for phonemes, letters in history_words:
            position_phonemes = []
            position_letters = []
            for index, phoneme in enumerate(phonemes.split(' ')):
                position_phonemes.append(phoneme + '_' + str(index))
            for index, letter in enumerate(letters.split(' ')):
                position_letters.append(letter + '_' + str(index))
            positioned_task.append((position_phonemes, position_letters))
        return positioned_task

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
