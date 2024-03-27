"""
Define: the environment interface
Return: TimeStep [observation, reward (uncertain?), discount(uncertain?), step_type]

The functions of the environment comprise three parts
(1) get the action from agents, change the old state to a new state
(2) summarize the information from new state, and give it to the agent as the observation
(3) other complementary functions

"""

import abc
from typing import List, Optional, Dict
from state_instance import State
import collections
from utils.choose_vocab_book import ReadVocabBook
import random
import enum


class StepType(enum.Enum):
    """Defines the status of a `TimeStep`."""

    FIRST = 0  # Denotes the initial `TimeStep`, start.
    MID = 1  # Denotes any `TimeStep` that is not FIRST or LAST.
    LAST = 2  # Denotes the last `TimeStep`, end.

    def first(self) -> bool:
        """
        Check if the step type is the first.

        :return: True if the step type is the first, False otherwise.
        """
        return self is StepType.FIRST

    def mid(self) -> bool:
        """
        Check if the step type is mid.

        :return: True if the step type is mid, False otherwise.
        """
        return self is StepType.MID

    def last(self) -> bool:
        """
        Check if the step type is the last.

        :return: True if the step type is the last, False otherwise.
        """
        return self is StepType.LAST


class TimeStep(collections.namedtuple("TimeStep", ["observations", "rewards", "discounts", "step_type"])):
    """
    Returned with every call to `step` and `reset`.
    """
    __slots__ = ()  # constrict the attributes of class

    observations: Dict[str, any]
    rewards: any
    discounts: float
    step_type: enum.Enum

    def first(self) -> bool:
        """
        Check if the step type is the first.

        :return: True if the step type is the first, False otherwise.
        """
        return self.step_type == StepType.FIRST

    def mid(self) -> bool:
        """
        Check if the step type is mid.

        :return: True if the step type is mid, False otherwise.
        """
        return self.step_type == StepType.MID

    def last(self) -> bool:
        """
        Check if the step type is the last.

        :return: True if the step type is the last, False otherwise.
        """
        return self.step_type == StepType.LAST

class EnvironmentInterface(metaclass=abc.ABCMeta):
    """ Environment Interface."""

    @abc.abstractmethod
    def __init__(self,
                 vocab_path: str,
                 vocab_book_name: str,
                 chinese_setting: bool,
                 phonetic_setting: bool,
                 POS_setting: bool,
                 english_setting: bool,
                 history_words_number: int,
                 review_words_number: int,
                 sessions_number: int,
                 discount: float = 1.0
                 ):
        """
        :args
                 vocab_path: the vocab data path for load vocabulary data
                 vocab_book_name: CET4, the book you want use
                 chinese_setting=True, do you want chinese?
                 phonetic_setting=True, do you want phonetic?
                 POS_setting=True, do you want POS?
                 english_setting=True, must be true

                 history_words_number: the number of learned words in history
                 review_words_number: the number of words per session
                 sessions_number:  the session numbers you want (days)

                 self._history_words: the environment is in charge of randomly choose the specific history words

                 self._state: read necessary information from state object
                 self._discount: the discount for the algorithm
                 self._should_reset: the timing to reset the game
                 self._player_num: the number of players in my game
                """

        self._vocab_book_name: str = vocab_book_name
        self._history_words_number: int = history_words_number
        self._review_words_number: int = review_words_number
        self._sessions_number: int = sessions_number

        # read the vocabulary book
        self._ReadVocabBook = ReadVocabBook(vocab_book_path=vocab_path,
                                            vocab_book_name=vocab_book_name,
                                            chinese_setting=chinese_setting,
                                            phonetic_setting=phonetic_setting,
                                            POS_setting=POS_setting,
                                            english_setting=english_setting)

        self._vocab_data: List[List[str]] = self._ReadVocabBook.read_vocab_book()
        # select the history words in accord with the history words number from vocab_data
        self._history_words: List[List[str]] = random.sample(self._vocab_data, self._history_words_number)

        self._state: Optional[State] = None
        self._discount: float = discount
        self._should_reset: bool = True
        self._player_num: int = 3

    @abc.abstractmethod
    def new_initial_state(self, history_words: List[List[str]], review_words_number: int,
                          sessions_number: int) -> State:
        """
        Initialize the state of the environment.

        :param history_words: A list of lists containing the history words provided by the environment.
        :param review_words_number: An integer indicating the number of review words in each session.
        :param sessions_number: An integer indicating the number of sessions representing the time units (days).

        :return: A new initial state of the environment.
        """

    @abc.abstractmethod
    def reset(self) -> TimeStep:
        """
        Reset the environment to its initial state.

        :return: A TimeStep representing the initial state of the game.
        """

    @abc.abstractmethod
    def get_time_step(self) -> TimeStep:
        """
        Construct the middle state of the environment.

        :return: A TimeStep representing the middle state of the environment.
        """

    @abc.abstractmethod
    def step(self, action) -> TimeStep:
        """
        Take a step in the environment.

        Args:
            action: The action to take in the environment.

        Returns:
            A TimeStep representing the result of taking the action in the environment.

        Steps:
            1. Apply the given action.
            2. Return a TimeStep representing the result.
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
