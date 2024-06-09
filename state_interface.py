"""
define the state interface
state provide whole necessary information to help env construct the TimeStep

"""
import abc
import pandas as pd
from typing import List, Tuple, Dict


class StateInterface(metaclass=abc.ABCMeta):
    """The state interface
    :args
        history_words (List[str]): A list of [phonemes, letters] pairs.
        review_words_number (int): The number of words to be reviewed in each session.
        sessions_number (int): The number of sessions.

        current_session_words Dict[str, List[List[str]]]: means [policy name : The words to be reviewed in each policy.].
        self._legal_actions: construct legal action for each agent

        self._current_session_num: integer, the current session number
        self._game_over: if the game terminate
        self._current_player: the current player

        self._student_memories:  store student memory dataframe

        self._examiner_feedback: give the feedback based on different policy.
        self._history_information: Dict[session_number, examiner_feedback]
    """

    @abc.abstractmethod
    def __init__(self, history_words: List[List[str]], review_words_number: int, sessions_number: int):
        self._history_words: List[List[str]] = history_words
        self._review_words_number: int = review_words_number
        self._sessions_number: int = sessions_number

        self._current_session_words: Dict[str, List[List[str]]] = dict()

        self._legal_actions: List[any] = [self._history_words,
                                          self._current_session_words,
                                          [0, 1]]

        self._current_session_num: int = 0
        self._game_over: bool = False
        self._current_player: int = 0
        self._student_memories: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] = tuple()

        # load students memory dataframe
        self._rewards: int = 0
        self._examiner_feedback: Dict[str, List[Tuple[any]]] = dict()
        # time: examiner_feedback
        self._history_information: Dict[int, dict] = {}

    @abc.abstractmethod
    def legal_actions(self, player_ID) -> List:
        """
        :return: Returns the legal action of the agent.
        """
        return self._legal_actions[player_ID]

    @abc.abstractmethod
    def apply_action(self, action) -> int:
        """
        :return: Returns necessary information
        """

    @property
    def rewards(self) -> int:
        """
        Do not use it currently
        :return: Returns the list of accuracy of each round
        """
        return self._rewards

    @property
    def examiner_feedback(self) -> dict:
        """
        Dictionary format: (policy_name, feedback)
        :return: give the feedback based on different policy
        """
        return self._examiner_feedback

    @property
    def history_information(self) -> Dict:
        """
        Dict[policy_name, examiner_feedback]
        """
        return self._history_information

    @property
    def current_player(self) -> int:
        """
        :return: Returns the current player index

        """
        return self._current_player

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
    def current_session_words(self) -> Dict[str, List[List[str]]]:
        """
        :return: Returns the current session tasks in each policy.
        """
        return self._current_session_words

    @property
    def current_session_num(self) -> int:
        """
        :return: Returns current session number.
        """
        return self._current_session_num

    @property
    def is_terminal(self) -> bool:
        """
                :return: the game status .
                """
        return self._game_over

    @property
    def student_memories(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        :return: student memories: excellent, forget, random, interfere
        """
        return self._student_memories
