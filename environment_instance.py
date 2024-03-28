"""
environment instance consists of four fundamental functions: new_initial_state, reset, step, get_time_step.

new_initial_state: initialize the start of the game
reset: reset the environment to initial state
step： apply the action receiving from agent to change the current state of environment
get_time_step: read information from the new state as the agents information

"""
from environment_interface import EnvironmentInterface, TimeStep, StepType
from state_instance import State


class VocabSpellGame(EnvironmentInterface):
    """ create the interactive environment"""

    def __init__(self,
                 vocabulary_book_path,
                 vocabulary_book_name,
                 chinese_setting,
                 phonetic_setting,
                 POS_setting,
                 english_setting,
                 history_words_number,
                 review_words_number,
                 sessions_number,
                 ):
        super().__init__(vocabulary_book_path,
                         vocabulary_book_name,
                         chinese_setting,
                         phonetic_setting,
                         POS_setting,
                         english_setting,
                         history_words_number,
                         review_words_number,
                         sessions_number,
                         )

    def new_initial_state(self, history_words, review_words_number, sessions_number):
        """ calling the state, and pass the history words, review_words_number, sessions_number """
        return State(history_words, review_words_number, sessions_number)

    def reset(self):
        """ initialize the state of environment"""
        self._state = self.new_initial_state(self.history_words, self.review_words_number, self.sessions_number)
        self._should_reset = False
        # initialize the observations, and read from state object
        observations = {"history_words": self._state.history_words,
                        "review_words_number": self._state.review_words_number,
                        "sessions_number": self._state.sessions_number,
                        "current_session_words": None,
                        "legal_actions": [],
                        "current_player": self._state.current_player,
                        "student_memories": None,
                        "current_session_num": None,
                        "examiner_feedback": None,
                        "history_information": None}

        # add the legal action of each player
        for player_ID in range(self._player_num):
            observations["legal_actions"].append(self._state.legal_actions(player_ID))

        return TimeStep(
            observations=observations,
            rewards=None,
            discounts=None,
            step_type=StepType.FIRST)

    def get_time_step(self):
        observations = {"history_words": self._state.history_words,
                        "review_words_number": self._state.review_words_number,
                        "sessions_number": self._state.sessions_number,
                        "current_session_words": self._state.current_session_words,
                        "legal_actions": [],
                        "current_player": self._state.current_player,
                        "current_session_num": self._state.current_session_num,
                        "student_memories": self._state.student_memories,
                        "examiner_feedback": self._state.examiner_feedback,
                        "history_information": self._state.history_information
                        }

        # add the legal action of each player
        for player_ID in range(self._player_num):
            observations["legal_actions"].append(self._state.legal_actions(player_ID))

        rewards = self._state.rewards  # how to define the rewards?!!!!!!!!!
        discounts = self._discount
        step_type = StepType.LAST if self._state.is_terminal else StepType.MID  # indicate the stage of environment
        self._should_reset = step_type == StepType.LAST  # True, if game terminate

        if step_type == StepType.LAST:
            # what to do if the game terminate !!!!!!!!!!!!!!!!!
            pass

        return TimeStep(
            observations=observations,
            rewards=rewards,
            discounts=discounts,
            step_type=step_type)

    def step(self, action):
        if self._should_reset:
            return self.reset()

        self._state.apply_action(action)  # (1) apply action/actions
        # (2) construct new TimeStep
        return self.get_time_step()
