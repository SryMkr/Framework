"""
define the state of environment
state provide whole necessary information to help env construct the TimeStep

"""
import os
import abc
import pandas as pd
from typing import List, Tuple, Dict


class StateInterface(metaclass=abc.ABCMeta):
    """The state interface
    :args
        history_words (List[str]): The history words.
        review_words_number (int): The number of words to be reviewed.
        sessions_number (int): The number of sessions.
        current_session_words (int): The number of words to be reviewed in the current session.


        self._current_session_num: integer, the current session number
        self._game_over: if the game terminate
        self._current_player: the current player
        self._legal_actions: construct legal action for each agent

        self._condition: str = '', for student spelling
        self._answer: str = '', for examine
        self._answer_length: int = 0, control the answer length
        self._LETTERS: = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                                         'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

        self.stu_memory_df:  store student memory dataframe
        self._letter_feedback: List[int], student spelling feedback of per letter
        self._accuracy: student answer accuracy
        self._completeness: student answer feedback
    """

    @abc.abstractmethod
    def __init__(self, history_words: List[List[str]], review_words_number: int, sessions_number: int):
        self._history_words: List[List[str]] = history_words
        self._review_words_number: int = review_words_number
        self._sessions_number: int = sessions_number
        self._current_session_words: List[List[str]] = []

        self._legal_actions: List[any] = [self._history_words,
                                          self._current_session_words]

        self._current_session_num: int = 1
        self._game_over: bool = False
        self._current_player: int = 0
        self._student_memories: Tuple[pd.DataFrame, pd.DataFrame] = tuple()




        self._rewards: int = 0
        self._current_corpus = tuple()
        self._condition: str = ''
        self._answer: str = ''
        self._answer_length: int = 0

        self._LETTERS: List[str] = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                                    'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

        # load students memory dataframe

        self._examiner_feedback: Tuple[List[int], float, float] = tuple()
        self._history_information: Dict[tuple, list] = {}
        self._avg_accuracy = []  # store the accuracy of each round

    @abc.abstractmethod
    def reward_function(self, information) -> int:
        """
        apply action, and store necessary information
        :return: Returns the (player ID) of the next player.
        """
        return self._rewards

    @property
    def rewards(self) -> int:
        """
        :return:  the rewards .
        """
        return self._rewards

    @property
    def answer_length(self) -> int:
        """
        :return: Returns answer length
        """
        return self._answer_length

    @property
    def answer(self) -> str:
        """
        :return: Returns the correct answer
        """
        return self._answer

    @property
    def condition(self) -> str:
        """
        :return: Returns the condition for spelling
        """
        return self._condition

    @property
    def average_accuracy(self) -> List[float]:
        """
        :return: Returns the list of accuracy of each round
        """
        return self._avg_accuracy



    @property
    def examiner_feedback(self) -> Tuple[List[int], float, float]:
        """
        :return: feedback per letter
        """
        return self._examiner_feedback

    @property
    def history_information(self) -> Dict:
        """
        what kinds of observation should be record to help tutor make decision?
        [condition[phonemes], answer length, examiner feedback[letters], accuracy, completeness]
        """
        return self._history_information

    @property
    def current_player(self) -> int:
        """
        :return: Returns the current player index

        """
        return self._current_player

    @abc.abstractmethod
    def legal_actions(self, player_ID) -> List:
        """
        :return: Returns the legal action of the agent.
        """
        return self._legal_actions[player_ID]

    @abc.abstractmethod
    def apply_action(self, action) -> int:
        """
        apply action, and store necessary information
        :return: Returns the (player ID) of the next player.
        """

    @property
    def history_words(self) -> List[List[str]]:
        """
        Get the history words provided by the environment.

        :return: A list of history word pairs, where each pair contains two strings [phonemes, letters].
        """
        return self._history_words

    @property
    def review_words_number(self) -> int:
        """
        Get the number of review words in each session.

        :return: The number of review words.
        """
        return self._review_words_number

    @property
    def sessions_number(self) -> int:
        """
        Get the number of sessions representing the time units (days).

        :return: The number of sessions.
        """
        return self._sessions_number

    @property
    def current_session_words(self) -> List[List[str]]:
        """
        :return: Returns the current session tasks.
        """
        return self._current_session_words

    @property
    def current_session_num(self) -> int:
        """
        :return: Returns current session.
        """
        return self._current_session_num

    @property
    def is_terminal(self) -> bool:
        """
                :return: the game status .
                """
        return self._game_over

    @property
    def student_memories(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        :return: student memories
        """
        return self._student_memories
