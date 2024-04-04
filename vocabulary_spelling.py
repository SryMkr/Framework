"""
the interaction of env and agents
"""

import os
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from environment_instance import VocabSpellGame
from agents_instance import CollectorPlayer, StudentPlayer, ExaminerPlayer
import pandas as pd


class draw_graph:
    def __init__(self, history_information):
        self.history_information = history_information

    def draw_average_accuracy(self):
        random_accuracy = []
        excellent_accuracy = []
        forget_accuracy = []
        random_collector_accuracy = []
        MAB_collector_accuracy = []
        for session_number, information in self.history_information.items():
            random_accuracy.append(information["random_memory"][1])
            excellent_accuracy.append(information["excellent"][1])
            forget_accuracy.append(information["forget"][1])
            random_collector_accuracy.append(information["random_collector"][1])
            MAB_collector_accuracy.append(information["MAB"][1])

        plt.figure(figsize=(15, 6))
        x_points = list(self.history_information.keys())
        # draw graph
        plt.plot(x_points, random_accuracy, color='blue', linestyle='-', label='Random')
        plt.plot(x_points, excellent_accuracy, color='red', linestyle='-', label='Excellent')
        plt.plot(x_points, forget_accuracy, color='green', linestyle='-.', label='Forget')
        plt.plot(x_points, random_collector_accuracy, color='orange', linestyle='--', label='random_collector')
        plt.plot(x_points, MAB_collector_accuracy, color='k', linestyle='--', label='MAB_collector')
        for i in range(len(x_points)):
            plt.text(x_points[i], random_accuracy[i], str(random_accuracy[i]), ha='center', va='bottom')
            plt.text(x_points[i], excellent_accuracy[i], str(excellent_accuracy[i]), ha='center', va='bottom')
            plt.text(x_points[i], forget_accuracy[i], str(forget_accuracy[i]), ha='center', va='bottom')
            plt.text(x_points[i], random_collector_accuracy[i], str(random_collector_accuracy[i]), ha='center',
                     va='bottom')
            plt.text(x_points[i], MAB_collector_accuracy[i], str(MAB_collector_accuracy[i]), ha='center',
                     va='bottom')
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

    agents = [CollectorPlayer(0, 'CollectorPlayer', ['random_collector', 'MAB']),
              StudentPlayer(1, 'StudentPlayer', excellent_memory_df, 'None'),
              ExaminerPlayer(2, 'ExaminerPlayer')]

    time_step = env.reset()  # initialize state

    while not time_step.last():  # not terminate
        player_id = time_step.observations["current_player"]  # current player
        agent_output = agents[player_id].step(time_step)  # action
        # print(agent_output)
        time_step = env.step(agent_output)  # current TimeStep
    draw_accuracy = draw_graph(time_step.observations["history_information"])
    draw_accuracy.draw_average_accuracy()
