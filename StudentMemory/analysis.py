# 我要读取这个Json数据 观察为什么随机的会比其他的高，在音标和字母上到底有什么差别？
"""
1：准确度和一组单词的音标的平均长度没多大关系
2：准确度和一组单词的字母的平均长度没大多关系
3：准确度和一组单词字母的多样性没多大关系
4：准确度和一组单词音标的多样性没多大关系
5: 对于每一天的记忆都有其最优解，而不是固定记忆哪些单词
6: 记忆前期长音标具有一定的优势，但是记忆后期，短音标表现出一定的优势
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def add_position(history_words):
    """ add position to phonemes and tasks"""
    positioned_task = []
    for phonemes, letters in history_words:
        position_phonemes = []
        position_letters = []
        for index, phoneme in enumerate(phonemes.split(' ')):
            position_phonemes.append(phoneme + '_' + str(index))
        for index, letter in enumerate(letters.split(' ')):
            position_letters.append(letter + '_' + str(index))
        positioned_task.append((position_phonemes, position_letters))
    return positioned_task


# 找到每一组平均概率最大的任务组
file_path = 'info.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

counter = Counter()

for a, b in data.items():  # 按照每轮循环
    accuracy_dic = b['accuracy']
    words = b['words']
    # 找到概率最大的那个策略
    max_key = max(accuracy_dic, key=accuracy_dic.get)
    # 根据策略找到策略选择的单词组
    task = words[max_key]
    print(task)
'''


r_length_list = []
r_accuracy = []
m_length_list = []
m_accuracy = []
l_length_list = []
l_accuracy = []
s_length_list = []
s_accuracy = []

task_number = 10
for a, b in data.items():  # 按照每轮循环
    accuracy_dic = b['accuracy']
    words = b['words']
    # 'random_collector'  'MAB'   'longest_collector' 'shortest_collector'
    # 每一轮，音标的平均长度和准确度之间的关系
    r_length = [len(phonemes.split(' ')) for phonemes, _ in words['random_collector']]
    r_length_list.append(np.sum(r_length) / 10)
    m_length = [len((phonemes.split(' '))) for phonemes, _ in words['MAB']]
    m_length_list.append(np.sum(m_length) / 10)
    l_length = [len((phonemes.split(' '))) for phonemes, _ in words['longest_collector']]
    l_length_list.append(np.sum(l_length) / 10)
    s_length = [len((phonemes.split(' '))) for phonemes, _ in words['shortest_collector']]
    s_length_list.append(np.sum(s_length) / 10)

    # 每一轮，单词的平均长度和准确度之间的关系
    # r_length = [len(letters.split(' ')) for _, letters in words['random_collector']]
    # r_length_list.append(np.sum(r_length) / 10)
    # m_length = [len((letters.split(' '))) for _, letters in words['MAB']]
    # m_length_list.append(np.sum(m_length) / 10)
    # l_length = [len((letters.split(' '))) for _, letters in words['longest_collector']]
    # l_length_list.append(np.sum(l_length) / 10)
    # s_length = [len((letters.split(' '))) for _, letters in words['shortest_collector']]
    # s_length_list.append(np.sum(s_length) / 10)

    # 每一轮，音标的多样性和准确度之间的关系，要添加位置信息

    # 每一轮，字母的多样性和准确度之间的关系
    # random_position_words = add_position(words['random_collector'])
    # phonemes_list = []
    # letters_list = []
    # for pair in random_position_words:
    #     phonemes = pair[0]
    #     letters = pair[1]
    #     phonemes_list.append(phonemes)
    #     letters_list.append(letters)
    # unique_phonemes = set(sum(phonemes_list, []))
    # unique_letters = set(sum(letters_list, []))
    # r_length_list.append(len(unique_letters))
    #
    # MAB_position_words = add_position(words['MAB'])
    # phonemes_list = []
    # letters_list = []
    # for pair in MAB_position_words:
    #     phonemes = pair[0]
    #     letters = pair[1]
    #     phonemes_list.append(phonemes)
    #     letters_list.append(letters)
    # unique_phonemes = set(sum(phonemes_list, []))
    # unique_letters = set(sum(letters_list, []))
    # m_length_list.append(len(unique_letters))
    #
    # long_position_words = add_position(words['longest_collector'])
    # phonemes_list = []
    # letters_list = []
    # for pair in long_position_words:
    #     phonemes = pair[0]
    #     letters = pair[1]
    #     phonemes_list.append(phonemes)
    #     letters_list.append(letters)
    # unique_phonemes = set(sum(phonemes_list, []))
    # unique_letters = set(sum(letters_list, []))
    # l_length_list.append(len(unique_letters))
    #
    # short_position_words = add_position(words['shortest_collector'])
    # phonemes_list = []
    # letters_list = []
    # for pair in short_position_words:
    #     phonemes = pair[0]
    #     letters = pair[1]
    #     phonemes_list.append(phonemes)
    #     letters_list.append(letters)
    # unique_phonemes = set(sum(phonemes_list, []))
    # unique_letters = set(sum(letters_list, []))
    # s_length_list.append(len(unique_letters))
    # 那我只能具体案例具体分析了，究竟每一轮随机选择的比MAB好在哪？
    # 统计音标的总数，总结规律
    # random_position_words = add_position(words['random_collector'])
    # phonemes_list = []
    # letters_list = []
    # for pair in random_position_words:
    #     phonemes = pair[0]
    #     letters = pair[1]

        # phonemes_list.append(phonemes)
        # letters_list.append(letters)
    # phonemes = sum(phonemes_list, [])
    # letters = sum(letters_list, [])
    # r_phonemes_Counter = Counter(phonemes)
    # print(r_phonemes_Counter)
    # r_letters_Counter = Counter(letters)
    # print(r_letters_Counter)
    # r_length_list.append(len(unique_letters))

    # MAB_position_words = add_position(words['MAB'])
    # phonemes_list = []
    # letters_list = []
    # for pair in MAB_position_words:
    #     phonemes = pair[0]
    #     letters = pair[1]
    #     phonemes_list.append(phonemes)
    #     letters_list.append(letters)
    # phonemes = sum(phonemes_list, [])
    # letters = sum(letters_list, [])
    # m_phonemes_Counter = Counter(phonemes)
    # # print(m_phonemes_Counter)
    # m_letters_Counter = Counter(letters)
    # print(m_letters_Counter)

    # long_position_words = add_position(words['longest_collector'])
    # phonemes_list = []
    # letters_list = []
    # for pair in long_position_words:
    #     phonemes = pair[0]
    #     letters = pair[1]
    #     phonemes_list.append(phonemes)
    #     letters_list.append(letters)
    # unique_phonemes = set(sum(phonemes_list, []))
    # unique_letters = set(sum(letters_list, []))
    # l_length_list.append(len(unique_letters))
    #
    # short_position_words = add_position(words['shortest_collector'])
    # phonemes_list = []
    # letters_list = []
    # for pair in short_position_words:
    #     phonemes = pair[0]
    #     letters = pair[1]

    #     phonemes_list.append(phonemes)
    #     letters_list.append(letters)
    # phonemes = sum(phonemes_list, [])
    # letters = sum(letters_list, [])
    # s_phonemes_Counter = Counter(phonemes)
    # print(r_phonemes_Counter)
    # s_letters_Counter = Counter(letters)
    # print(s_letters_Counter)

    # 统计字母的总数，总结规律
    ra = accuracy_dic['random_collector']
    r_accuracy.append(ra)
    ma = accuracy_dic['MAB']
    m_accuracy.append(ma)
    la = accuracy_dic['longest_collector']
    l_accuracy.append(la)
    sa = accuracy_dic['shortest_collector']
    s_accuracy.append(sa)

x1 = list(range(len(r_accuracy)))
fig, ax1 = plt.subplots()
#
ax1.plot(x1, r_length_list, marker='o', label='random', color='blue')
ax1.plot(x1, m_length_list, marker='o', label='mab', color='red')
ax1.plot(x1, l_length_list, marker='o', label='long', color='green')
ax1.plot(x1, s_length_list, marker='o', label='short', color='orange')
ax1.set_xlabel('session')
ax1.set_ylabel('length', color='b')
#
ax2 = ax1.twinx()
ax2.plot(x1, r_accuracy, marker='x', linestyle='--', label='random_accuracy', color='blue')
ax2.plot(x1, m_accuracy, marker='x', linestyle='--', label='mab_accuracy', color='red')
ax2.plot(x1, l_accuracy, marker='x', linestyle='--', label='long_accuracy', color='green')
ax2.plot(x1, s_accuracy, marker='x', linestyle='--', label='short_accuracy', color='orange')
ax2.set_ylabel('Average Length', color='r')

plt.plot(r_length_list, r_accuracy, color='blue')
plt.plot(m_length_list, m_accuracy, color='red')
plt.plot(l_length_list, l_accuracy, color='green')
plt.plot(s_length_list, s_accuracy, color='orange')

# 添加标题和标签
plt.title('Scatter Plot of Two Variables')
fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
# 显示网格
plt.grid(True)

# 显示图形
plt.show()

'''
