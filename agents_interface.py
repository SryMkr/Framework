"""
1: agent base abstract class, and per agent interface
2: initialize some parameters that will be regularly used in agents instance
# In a nutshell: An agent normally has
    (1) attributes: player_ID, player_Name, **agent_specific_kwargs, all implemented in __init__ function
    (2) step function: A: parameter: get the observation of environment time step
                       B: a policy get the observation and provide the action probabilities, then agent select an action based on probabilities
                       (Environment (observation, reward)-> agent ((policy function -> action probabilities)->action))
"""

import abc
from typing import List, Tuple, Dict
import pandas as pd
import torch


class AgentAbstractBaseClass(metaclass=abc.ABCMeta):
    """Agent abstract base class for all kinds of agents."""

    @abc.abstractmethod
    def __init__(self,
                 player_id: int,
                 player_name: str,
                 **agent_specific_kwargs):
        """
        Initializes agent abstract base

        Args:
            player_id (int): The zero-based integer for the agent's index.
            player_name (str): The name of the player.
            **agent_specific_kwargs: Optional extra arguments specific to the agent.
        """
        self._player_id: int = player_id
        self._player_name: str = player_name

    @abc.abstractmethod
    def step(self, time_step):
        """
        Agents should observe the `time_step` from the environment and extract the required part of the
        `time_step.observations` field and if applicable, the 'reward' field.

        Args:
            time_step: An instance of rl_environment.TimeStep.

        Returns:
            A `StepOutput` for the current `time_step`, containing the action or actions.
        """

    @property
    def player_id(self) -> int:
        """
        :return: the player ID
        """
        return self._player_id

    @property
    def player_name(self) -> str:
        """
        :return: the player name
        """
        return self._player_name


class CollectorAgentInterface(AgentAbstractBaseClass):
    """An interface for the collector agent.

    This agent is responsible for collecting words to be reviewed from history words.

    observation: extract the "history words", "review_words_number", "history_information" from rl_environment.TimeStep
    Policies: Designate the policies you want to compare ["random", "MAB"]
    """

    @abc.abstractmethod
    def __init__(self,
                 player_id: int,
                 player_name: str,
                 policies: List[str]):
        """
        Initializes the collector agent.

        Args:
            player_id (int): The ID of the player.
            player_name (str): The name of the player.
            policies List[str]: The policies used by the agent.
            self._actions Dict[str, List[List[str]]]: the first parameter is policy name, second is [[phonemes, letters]]
        """
        super().__init__(player_id, player_name)
        self._policies: List[str] = policies
        self._actions: Dict[str, List[List[str]]] = dict()

    @abc.abstractmethod
    def step(self, time_step) -> Dict[str, List[List[str]]]:
        """
        Take a step for the agent.

         Args:
            time_step: An instance of rl_environment.TimeStep.

        Returns:
            Dict[str, List[List[str]]]:  Dict[policy name : The words to be reviewed.]
        """
        return self._actions


class StudentAgentInterface(AgentAbstractBaseClass):
    """An interface for a student agent.

    The agent is responsible for reviewing selected words from the CollectorAgent each day (Learn).

    Observation：rl_environment.TimeStep[observation]["current_session_words"]
                 rl_environment.TimeStep[observation]["history_words"]
                 rl_environment.TimeStep[observation]["sessions_number"]
                 rl_environment.TimeStep[observation]["current_session_words"]
    """

    @abc.abstractmethod
    def __init__(self,
                 player_id: int,
                 player_name: str,
                 excellent_memory_dataframe: pd.DataFrame,
                 policy: str):
        """
        Initializes the student agent.

        Args:
            player_id (int): The ID of the player.
            player_name (str): The name of the player.
            excellent_memory_dataframe(pd.DataFrame): the student excellent memory.
            policy (str): The policy used by the agent.

        """
        super().__init__(player_id, player_name)
        self._policy: str = policy

        # excellent memory dataframe
        self._excellent_memory_df: Dict[str, pd.DataFrame] = {'excellent': excellent_memory_dataframe}

        # random memory dataframe and use it as noise
        stu_memory_tensor = torch.tensor(self._excellent_memory_df['excellent'].values,
                                         dtype=torch.float32)  # the shape of distribution
        noise = torch.randn_like(stu_memory_tensor)  # generate the noise
        scaled_noise = (noise - noise.min()) / (noise.max() - noise.min())
        random_memory = pd.DataFrame(scaled_noise.numpy(), index=self._excellent_memory_df['excellent'].index,
                                     columns=self._excellent_memory_df['excellent'].columns)
        result_df = random_memory.div(random_memory.sum(axis=1), axis=0)
        self._random_memory_df: Dict[str, pd.DataFrame] = {'random_memory': result_df}
        # forget memory dataframe
        self._forget_memory_df: Dict[str, pd.DataFrame] = dict()
        # learn memory dataframe
        self._learn_memory_df: Dict[str, pd.DataFrame] = dict()

    @abc.abstractmethod
    def step(self, time_step) -> Tuple[Dict[str, pd.DataFrame]]:
        """
        Executes a step for the agent.

        Args:
            time_step: An instance of rl_environment.TimeStep.

        Returns:
            Tuple[Dict[str, pd.DataFrame]]: The random (do not learn)
                                            excellent (do not forget and do not need to learn)
                                            forgetting (forget)
                                            learn (forget then learn).
        """
        pass


class ExaminerAgentInterface(AgentAbstractBaseClass):
    """ An interface for a student agent.
        The agent is responsible for giving the feedback of each memory each day.

        Observation：rl_environment.TimeStep[observation]["history_words"]
                    rl_environment.TimeStep[observation]["student_memories"]
    """

    def __init__(self,
                 player_id: int,
                 player_name: str):
        super().__init__(player_id,
                         player_name)

        self._examiner_feedback: Dict[str, List[Tuple[any]]] = dict()

    @abc.abstractmethod
    def step(self, time_step) -> Dict[str, List[Tuple[any]]]:
        """
        Executes a step for the agent.

        Args:
            time_step: An instance of rl_environment.TimeStep.

        Returns:
           List[Tuple[Dict[any]]]: ({('d w ɛ l ɪ ŋ', 'd w e l l i n g'): (['d_1', 'w_1', 'e_1', 'l_1', 'i_0', 'n_0', 'n_1', 'g_1'], 0.88))
        """
        pass
