"""
the interaction of env and agents
"""

import os
from typing import Dict
import matplotlib.pyplot as plt
from environment_instance import VocabSpellGame
from agents_instance import CollectorPlayer, StudentPlayer, ExaminerPlayer
import pandas as pd


class draw_graph:
    def __init__(self, history_information: Dict[int, list]):
        self.history_information = history_information

    def draw_average_accuracy(self):
        random_accuracy = []
        excellent_accuracy = []
        forget_accuracy = []
        learn_accuracy = []
        for session_number, information in self.history_information.items():
            random_accuracy.append(information[0][1])
            excellent_accuracy.append(information[1][1])
            forget_accuracy.append(information[2][1])
            learn_accuracy.append(information[3][1])

        plt.figure(figsize=(15, 6))
        x_points = list(self.history_information.keys())
        # draw graph
        plt.plot(x_points, random_accuracy, color='blue', linestyle='-', label='Random')
        plt.plot(x_points, excellent_accuracy, color='red', linestyle='-', label='Excellent')
        plt.plot(x_points, forget_accuracy, color='green', linestyle='-.', label='Forget')
        plt.plot(x_points, learn_accuracy, color='orange', linestyle='--', label='Learn')

        for i in range(len(x_points)):
            plt.text(x_points[i], random_accuracy[i], str(random_accuracy[i]), ha='center', va='bottom')
            plt.text(x_points[i], excellent_accuracy[i], str(excellent_accuracy[i]), ha='center', va='bottom')
            plt.text(x_points[i], forget_accuracy[i], str(forget_accuracy[i]), ha='center', va='bottom')
            plt.text(x_points[i], learn_accuracy[i], str(learn_accuracy[i]), ha='center', va='bottom')

        # 添加标题和标签
        plt.title('Average Accuracy Each Session ')
        plt.xlabel('Days')
        plt.ylabel('Average Accuracy')
        plt.xticks(x_points)
        # 添加图例
        plt.legend()

        # 显示图形
        plt.show()


if __name__ == "__main__":
    current_path = os.getcwd()  # get the current path
    vocabulary_absolute_path = os.path.join(current_path, 'VocabularyBook', 'CET4',
                                            'vocabulary.json')  # get the vocab data path

    env = VocabSpellGame(vocabulary_book_path=vocabulary_absolute_path,
                         vocabulary_book_name='CET4',
                         chinese_setting=False,
                         phonetic_setting=True,
                         POS_setting=False,
                         english_setting=True,
                         history_words_number=50,
                         review_words_number=10,
                         sessions_number=30,
                         )  # initialize game environment

    # instance agents
    student_excellent_memory_path = os.path.join(current_path, 'StudentMemory/excellent_memory.xlsx')
    excellent_memory_df = pd.read_excel(student_excellent_memory_path, index_col=0, header=0)

    agents = [CollectorPlayer(0, 'CollectorPlayer', 'MAB'),
              StudentPlayer(1, 'StudentPlayer', excellent_memory_df, 'None'),
              ExaminerPlayer(2, 'ExaminerPlayer')]

    time_step = env.reset()  # initialize state

    while not time_step.last():  # not terminate
        player_id = time_step.observations["current_player"]  # current player
        agent_output = agents[player_id].step(time_step)  # action
        time_step = env.step(agent_output)  # current TimeStep
    draw_accuracy = draw_graph(time_step.observations["history_information"])
    draw_accuracy.draw_average_accuracy()

