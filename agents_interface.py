"""
1: agent base abstract class, and per agent interface
2: initialize some parameters that will be regularly used in agents instance
# In a nutshell: An agent normally has
    (1) attributes: player_ID, player_Name, **agent_specific_kwargs, all implemented in __init__ function
    (2) step function: A: parameter: get the observation of environment time step
                       B: a policy get the observation and provide the action probabilities, then agent select an action based on probabilities
                       (Environment (observation, reward)-> agent ((policy function -> action probabilities)->action))
    compared with the (uniform_random) agent that have different policy

"""

import abc
from typing import List, Tuple, Dict
import pandas as pd
import torch
import numpy as np


class AgentAbstractBaseClass(metaclass=abc.ABCMeta):
    """Abstract base class for all agents."""

    @abc.abstractmethod
    def __init__(self,
                 player_id: int,
                 player_name: str,
                 **agent_specific_kwargs):
        """
        Initializes the agent.

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
        `time_step.observations` field and 'reward' field.

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
    """An interface for a collector agent.

    This agent is responsible for collecting words to be reviewed from history words.

    observation: extract the "history words" from  rl_environment.TimeStep
    Policy: there are two optional policy including "random", "AMB"

    """

    @abc.abstractmethod
    def __init__(self,
                 player_id: int,
                 player_name: str,
                 policy: str):
        """
        Initializes the collector agent.

        Args:
            player_id (int): The ID of the player.
            player_name (str): The name of the player.
            policy (str): The policy used by the agent.
        """
        super().__init__(player_id, player_name)
        self._policy: str = policy

    @abc.abstractmethod
    def step(self, time_step) -> List[List[str]]:
        """
        Executes a step for the agent.

         Args:
            time_step: An instance of rl_environment.TimeStep.




        Returns:
            List[List[str]]: The words to be reviewed.
        """
        pass


class StudentAgentInterface(AgentAbstractBaseClass):
    """An interface for a student agent.

    The agent is responsible for reviewing selected words from the CollectorAgent each day (Learn).

    Observation：rl_environment.TimeStep[observation]["current_session_words"]

    Policy： random (do not learn)
            excellent (do not forget and do not need to learn)
            forgetting (forget)
            learn (forget and learn)

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
        self._policy = policy

        self._excellent_memory_df = excellent_memory_dataframe  # excellent memory dataframe

        normal_noise = np.random.normal(0, 1, self._excellent_memory_df.shape)
        # normalized_noise = (noise - noise.min(axis=1, initial=None, keepdims=True)) / (noise.max(axis=1, initial=None, keepdims=True) - noise.min(axis=1, initial=None, keepdims=True))
        # random memory dataframe
        # 将所有负数截断为零
        normal_noise = np.clip(normal_noise, 0, None)
        normalized_noise = normal_noise / normal_noise.sum(axis=1, keepdims=True)
        self._random_memory_df = pd.DataFrame(normalized_noise, columns=self._excellent_memory_df.columns,
                                              index=self._excellent_memory_df.index)
        print(self._random_memory_df)
        # 生成遗忘矩阵
        # 生成学习矩阵

    @abc.abstractmethod
    def step(self, time_step) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Executes a step for the agent.

        Args:
            time_step: An instance of rl_environment.TimeStep.

        Returns:
            Tuple[pd.DataFrame]: The random (do not learn)
                                 excellent (do not forget and do not need to learn)
                                 forgetting (forget)
                                 learn (forget and learn)  memory dataframe.
        """
        pass


class ExaminerInterface(AgentAbstractBaseClass):
    """Examiner Interface： provide feedback based on the [student spelling, correct answer]
        Legal actions: [0，1], where 0 presents wrong, 1 denotes right
        Observation：TimeStep [student_spelling，answer]
        Policy：no policy
        Output：actions: for example, ([0,1,1,0,1,1,1], similarity)
        State: calculate the similarity
    """

    def __init__(self,
                 player_id: int,
                 player_name: str):
        super().__init__(player_id,
                         player_name)
        """Initializes examiner agent"""

    @abc.abstractmethod
    def step(self, time_step) -> Tuple[Dict[str, int], float]:
        """
        :returns the ({letter_position: mark}, accuracy)
        """
        pass
