"""
the interaction of env and agents
"""

import os
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
        topsis_collector_accuracy = []
        longest_collector_accuracy = []
        shortest_collector_accuracy = []
        for session_number, information in self.history_information.items():
            random_accuracy.append(information["random_memory"][1])
            excellent_accuracy.append(information["excellent"][1])
            forget_accuracy.append(information["forget"][1])
            random_collector_accuracy.append(information["random_collector"][1])
            topsis_collector_accuracy.append(information["TOPSIS"][1])
            longest_collector_accuracy.append(information["longest_collector"][1])
            shortest_collector_accuracy.append(information["shortest_collector"][1])

        plt.figure(figsize=(15, 6))
        x_points = list(self.history_information.keys())
        # draw graph
        # plt.plot(x_points, random_accuracy, color='blue', linestyle='-', label='Random')
        # plt.plot(x_points, excellent_accuracy, color='red', linestyle='-', label='Excellent')
        # plt.plot(x_points, forget_accuracy, color='green', linestyle='-.', label='Forget')
        plt.plot(x_points, random_collector_accuracy, color='orange', linestyle='--', label='random_collector')
        plt.plot(x_points, topsis_collector_accuracy, color='k', linestyle=':', label='topsis_collector')
        plt.plot(x_points, longest_collector_accuracy, color='purple', linestyle='--', label='longest_collector')
        plt.plot(x_points, shortest_collector_accuracy, color='cyan', linestyle='--', label='shortest_collector')

        for i in range(len(x_points)):
            # plt.text(x_points[i], random_accuracy[i], str(random_accuracy[i]), ha='center', va='bottom')
            # plt.text(x_points[i], excellent_accuracy[i], str(excellent_accuracy[i]), ha='center', va='bottom')
            # plt.text(x_points[i], forget_accuracy[i], str(forget_accuracy[i]), ha='center', va='bottom')
            plt.text(x_points[i], random_collector_accuracy[i], str(random_collector_accuracy[i]), ha='center',
                     va='bottom')
            plt.text(x_points[i], topsis_collector_accuracy[i], str(topsis_collector_accuracy[i]), ha='center',
                     va='bottom')
            plt.text(x_points[i], longest_collector_accuracy[i], str(longest_collector_accuracy[i]), ha='center',
                     va='bottom')
            plt.text(x_points[i], shortest_collector_accuracy[i], str(shortest_collector_accuracy[i]), ha='center',
                     va='bottom')
        # add title and label
        plt.title('Average Accuracy Each Session ')
        plt.xlabel('Days')
        plt.ylabel('Average Accuracy')
        plt.xticks(x_points)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    current_path = os.getcwd()  # get the current path
    vocabulary_absolute_path = os.path.join(current_path, 'VocabularyBook', 'CET4',
                                            'vocabulary.json')  # get the vocab data path

    # instance agents
    student_excellent_memory_path = os.path.join(current_path, 'StudentMemory/excellent_memory.xlsx')
    student_random_memory_path = os.path.join(current_path, 'StudentMemory/random_memory.xlsx')
    excellent_memory_df = pd.read_excel(student_excellent_memory_path, index_col=0, header=0)
    random_memory_df = pd.read_excel(student_random_memory_path, index_col=0, header=0)
    agents = [
        CollectorPlayer(0, 'CollectorPlayer',
                        ['random_collector', 'TOPSIS', 'longest_collector', 'shortest_collector']),
        StudentPlayer(1, 'StudentPlayer', excellent_memory_df, random_memory_df, 'None'),
        ExaminerPlayer(2, 'ExaminerPlayer')]

    index = [f'{i}' for i in range(30)]
    columns = ["random_memory", "excellent", "forget", "random", "topsis", "longest", "shortest"]
    original_df = pd.DataFrame(0.0, index=index, columns=columns)
    epoch = 1

    for i in range(epoch):
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

    # time_step = env.reset()  # initialize state
    # while not time_step.last():  # not terminate
    #     player_id = time_step.observations["current_player"]  # current player
    #     agent_output = agents[player_id].step(time_step)  # action
    #     time_step = env.step(agent_output)  # current TimeStep
    #     # 在这统计结果就可以
    #
    # draw_accuracy = draw_graph(time_step.observations["history_information"])
    # draw_accuracy.draw_average_accuracy()
        time_step = env.reset()  # initialize state
        while not time_step.last():  # not terminate
            player_id = time_step.observations["current_player"]  # current player
            agent_output = agents[player_id].step(time_step)  # action
            time_step = env.step(agent_output)  # current TimeStep
        history_information = time_step.observations["history_information"]

        random_accuracy = []
        excellent_accuracy = []
        forget_accuracy = []
        random_collector_accuracy = []
        topsis_collector_accuracy = []
        longest_collector_accuracy = []
        shortest_collector_accuracy = []

        for session_number, information in history_information.items():
            random_accuracy.append(information["random_memory"][1])
            excellent_accuracy.append(information["excellent"][1])
            forget_accuracy.append(information["forget"][1])
            random_collector_accuracy.append(information["random_collector"][1])
            topsis_collector_accuracy.append(information["TOPSIS"][1])
            longest_collector_accuracy.append(information["longest_collector"][1])
            shortest_collector_accuracy.append(information["shortest_collector"][1])

        new_df = pd.DataFrame({
            'random_memory': random_accuracy,
            'excellent': excellent_accuracy,
            'forget': forget_accuracy,
            'random': random_collector_accuracy,
            'topsis': topsis_collector_accuracy,
            'longest': longest_collector_accuracy,
            'shortest': shortest_collector_accuracy,
        }, index=index)

        original_df = original_df + new_df
        original_df.to_excel("Evaluation_Results/excellent20.xlsx", index=True)
        print(original_df)
        # print(f"当前第{i}轮已经结束")
    # # 将后四列的元素减去前4列的元素
    # improvement_df = original_df.iloc[:, 1:].subtract(original_df['forget'], axis=0)
    # # 再除以循环次数
    # improvement_df = improvement_df/epoch
    # # 再保存到文件中
    # performance_df = improvement_df[["random", "topsis", "longest", "shortest"]]
    # # 保存到Excel文件
    # performance_df.to_excel("Evaluation_Results/performance20.xlsx", index=True)


