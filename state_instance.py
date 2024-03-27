from state_interface import StateInterface
import numpy as np


class State(StateInterface):
    def __init__(self, history_words, review_words_number, sessions_number):
        super().__init__(history_words, review_words_number, sessions_number)

    def legal_actions(self, player_ID):
        """get the legal action of one agent"""
        return self._legal_actions[player_ID]

    def apply_action(self, action):
        """ control the transition of state"""
        if self._current_player == 0:  # 0 means collector agent
            self._current_session_words = action  # the objective of the Collector Agent is selected the prioritised words

        elif self._current_player == 1:  # 1 is student agent
            self._student_memories = action

        elif self._current_player == 2:  # 2 is examiner agent
            self._examiner_feedback = action
            self._current_session_num += 1  # change the number of sessions

        self._current_player += 1
        if self._current_player == 3:
            self._current_player = 0

        if self._current_session_num == self.sessions_number:
            """ if the session has been finished and the session is the last session,then game over """
            self._game_over = True

