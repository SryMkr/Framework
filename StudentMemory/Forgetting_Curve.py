"""
如果我对优秀记忆加一个正态分布的噪声，是不是直接把准确率提高到了100%
"""

import string
import numpy as np
import random
import Levenshtein
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from utils.choose_vocab_book import ReadVocabBook

CURRENT_PATH = os.getcwd()  # get the current path
VOCAB_PATH: str = os.path.join(CURRENT_PATH, '../VocabularyBook', 'CET4', 'vocabulary.json')  # get the vocab data path
corpus_instance = ReadVocabBook(vocab_book_path=VOCAB_PATH,
                                vocab_book_name='CET4',
                                chinese_setting=False,
                                phonetic_setting=True,
                                POS_setting=False,
                                english_setting=True)
original_corpus = corpus_instance.read_vocab_book()
random.shuffle(original_corpus)

def add_position(corpus):
    """add position for each corpus"""
    corpus_with_position = []
    for pair in corpus:
        phonemes_position = ''
        letters_position = ''
        pair_position = []
        phonemes_list = pair[0].split(' ')
        for index, phoneme in enumerate(phonemes_list):
            phoneme_index = phoneme + '_' + str(index)
            phonemes_position = phonemes_position + phoneme_index + ' '
        letters_list = pair[1].split(' ')

        for index, letter in enumerate(letters_list):
            letter_index = letter + '_' + str(index)
            letters_position = letters_position + letter_index + ' '
        pair_position.append(phonemes_position.strip())
        pair_position.append(letters_position.strip())
        corpus_with_position.append(pair_position)
    return corpus_with_position



pos_corpus = add_position(original_corpus)  # get the training data [phonemes, word]

hhh_corpus = random.sample(pos_corpus, 10)

def forward_process_dataframe(corpus, dataframe, alphas_bar_sqrt, one_minus_alphas_bar_sqrt):
    dataframe_tensor = torch.tensor(dataframe.values, dtype=torch.float32)
    noise = torch.randn_like(dataframe_tensor)
    scaled_noise = (noise - noise.min()) / (noise.max() - noise.min())

    random_memory_df = pd.DataFrame(scaled_noise.numpy(), index=dataframe.index, columns=dataframe.columns)

    excellent_dataframe_copy = dataframe.copy()

    positioned_task = []
    for phonemes, letters in corpus:
        position_phonemes = phonemes.split(' ')
        position_letters = letters.split(' ')
        positioned_task.append((position_phonemes, position_letters))

    for pho in positioned_task:
        # excellent_dataframe_copy.loc[pho[0]] = alphas_bar_sqrt * dataframe.loc[pho[0]] + one_minus_alphas_bar_sqrt * \
        #                                     random_memory_df.loc[pho[0]]
        excellent_dataframe_copy.loc[pho] = alphas_bar_sqrt * dataframe.loc[pho] + one_minus_alphas_bar_sqrt * \
                                               random_memory_df.loc[pho]
    result_df = excellent_dataframe_copy.div(excellent_dataframe_copy.sum(axis=1), axis=0)  # normalize
    return result_df


def generate_answer(memory_df, test_corpus):
    """ generate answer based on the given phonemes,而且我要知道答案的长度，然后根据所有的音标对每一个位置选择最大值"""
    student_answer_pair = []
    random_student_answer_pair = []  # 记录随机拼写的准确度
    test_corpus = [[item.split() for item in sublist] for sublist in test_corpus]
    for phonemes, answer in test_corpus:
        random_spelling = []
        spelling = []
        answer_length = len(answer)
        alphabet = string.ascii_lowercase
        for i in range(answer_length):
            result_columns = [al + '_' + str(i) for al in alphabet]
            possible_results = memory_df.loc[phonemes, result_columns]
            letters_prob = possible_results.sum(axis=0)  # 每一列相加,取概率最大值
            letter = letters_prob.idxmax()
            random_letter = random.choice(string.ascii_lowercase) + '_' + str(i)
            spelling.append(letter)
            random_spelling.append(random_letter)
        random_student_answer_pair.append([random_spelling, answer])
        student_answer_pair.append([spelling, answer])

    return student_answer_pair, random_student_answer_pair


def evaluation(answer_pair):
    accuracy = []
    for stu_answer, correct_answer in answer_pair:
        stu_answer = ''.join([i.split('_')[0] for i in stu_answer])
        correct_answer = ''.join([i.split('_')[0] for i in correct_answer])
        word_accuracy = round(Levenshtein.ratio(correct_answer, stu_answer), 2)
        accuracy.append(word_accuracy)
    avg_accuracy = sum(accuracy) / len(accuracy)
    return avg_accuracy

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    steps = 30  # means 30 days
    scaled_steps = np.linspace(0, 1, steps)
    alphas_bar_sqrt_ = 0.5 * np.exp(-2.5 * scaled_steps) + 0.4
    one_minus_alphas_bar_sqrt_ = 0.9 * (1 - np.exp(-2 * scaled_steps)) + 0.1
    avg_acc_list = [0.77]
    random_avg_acc_list = [0.15]

    current_path = os.getcwd()
    student_excellent_memory_path = os.path.join(current_path, 'excellent_memory.xlsx')
    excellent_memory_df = pd.read_excel(student_excellent_memory_path, index_col=0, header=0)

    for t in range(1, steps + 1):
        forgetting_student_memory = forward_process_dataframe(hhh_corpus, excellent_memory_df, alphas_bar_sqrt_[t-1], one_minus_alphas_bar_sqrt_[t-1])
        spelling_answer_pair, random_answer_pair = generate_answer(forgetting_student_memory, hhh_corpus)
        avg_acc = evaluation(spelling_answer_pair)
        random_avg_acc = evaluation(random_answer_pair)
        print(f'epoch: {t}: average accuracy is {avg_acc}........the random accuracy is {random_avg_acc}')
        avg_acc_list.append(avg_acc)
        random_avg_acc_list.append(random_avg_acc)
    plt.figure()
    plt.plot([i for i in range(steps + 1)], avg_acc_list, label='forgetting curve')
    plt.plot([i for i in range(steps + 1)], random_avg_acc_list, label='random')
    plt.xlabel('days')
    plt.ylabel('accuracy')
    plt.show()
