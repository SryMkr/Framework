from state_interface import StateInterface
import numpy as np


class State(StateInterface):
    def __init__(self, history_words, review_words_number, sessions_number):
        super().__init__(history_words, review_words_number, sessions_number)

    def legal_actions(self, player_ID):
        """get the legal action of one agent"""
        return self._legal_actions[player_ID]

    def spilt_task(self, action):
        """get the 'condition' and word_length for student, and answer for examiner """
        for task in self._current_session_words:
            if task == action:  # 如果两个任务完全相等
                self._current_corpus = tuple(task)  # the type of corpus is tuple('s ɛ n t ʌ n s', 's e n t e n c e')
                self._condition = task[0]
                self._answer = task[1]
                self._answer_length = len(''.join(task[1].split(' ')))
                break

    def apply_action(self, action):
        """ control the transition of state"""
        if self._current_player == 0:  # 0 means collector agent
            self._current_session_words = action   # the objective of the Collector Agent is selected the prioritised words

        elif self._current_player == 1:  # 1 is student agent
            self._student_memories = action

        self._current_player += 1

        if self._current_session_num == self.sessions_number:
            """ if the session has been finished and the session is the last session,then game over """
            self._game_over = True
        # elif self._current_player == 3:  # 3 is examiner agent
        #     self._examiner_feedback = action
        #
        #     # update the history information
        #     self._history_information[self._current_corpus] = [self._examiner_feedback]
        #     # store all information
        #     # if self._current_corpus in self._history_information:
        #     #     self._history_information[self._current_corpus].append(self._examiner_feedback)
        #     # else:
        #     #     self._history_information[self._current_corpus] = [self._examiner_feedback]
        #
        #     # if session task is empty, then select a new session, else continue to select new word from the session
        #     if len(self.legal_actions(1)) == 0:
        #         """如果当前的session为空则第一个玩家重新选择session"""
        #
        #         self._current_session_num += 1  # change the number of sessions
        #         # 如果要计算平均准确度，一定要计算根据当前的记忆，对历史单词的准确度
        #
        #         print(f"the session {self._current_session_num}current history information is", len(self._history_information))
        #     else:
        #         self._current_player = 1


    def reward_function(self, information):
        """the reward only for tutor agent, the information just accuracy
        奖励应该和准确度成正比，虽然和单词长度也有一定的关系"""
        scaled_accuracy = np.tanh(information * (np.pi / 2) - (np.pi / 4))
        return np.tan(scaled_accuracy)

