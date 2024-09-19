"""
看看标准的IBM model给出的结果
"""

import random
import Levenshtein
import os
from utils.choose_vocab_book import ReadVocabBook
from typing import List, Tuple
import pandas as pd
import string


class PositionPhoLetStudent:
    def __init__(self, train_corpus: List[List[str]], initial_prob, p_index, l_columns):
        self.train_corpus = [[item.split() for item in sublist] for sublist in train_corpus]
        self.letters: List[str] = l_columns
        self.phonemes: List[str] = p_index
        self.phoneme_letter_prob = initial_prob
        self.phoneme_letter_df = pd.DataFrame()
        self.student_answer_pair = []
        self.accuracy = []
        self.avg_accuracy = 0.0

    def train_model(self, num_epochs=1):
        """ train the model"""
        s_total = {}
        for epoch in range(num_epochs):
            phoneme_letter_counts = {}
            total = {}
            # initialize the counts of fw-ew pair
            for fw in self.phonemes:
                total[fw] = 0.00001
                for ew in self.letters:
                    if fw not in phoneme_letter_counts:
                        phoneme_letter_counts[fw] = {}
                    phoneme_letter_counts[fw][ew] = 0
            # print('fw-ew pairs: ', phoneme_letter_counts)

            for sp in self.train_corpus:
                # 对于每一个corpus，将所有的外文单词对应到每一个英文单词上面的概率值相加，相当于不同的对齐方式
                for ew in sp[1]:  # 循环英文字母['n aɪ n t i n', 'n i n e t e e n']
                    s_total[ew] = 0.0
                    for fw in sp[0]:
                        s_total[ew] += self.phoneme_letter_prob.loc[fw, ew]

                for ew in sp[1]:
                    # 将每一种对齐关系中的概率除于总的概率（所有对齐关系）近似于求出每一种对齐方式的概率，相加就是对应关系的期望数量
                    for fw in sp[0]:
                        # 对于任何一种对齐方式  求出其期望数量
                        phoneme_letter_counts[fw][ew] += self.phoneme_letter_prob.loc[fw, ew] / s_total[ew]
                        total[fw] += self.phoneme_letter_prob.loc[fw, ew] / s_total[ew]  # 求出不同外文单词的概率
            # normalization
            for fw in self.phonemes:
                for ew in self.letters:
                    self.phoneme_letter_df.loc[fw, ew] = phoneme_letter_counts[fw][ew] / total[fw]
        self.phoneme_letter_df.replace(0, 0.0001, inplace=True)  # replace zero with constant

    def generate_answer(self):
        """ 只能通过词对词的翻译来进行"""
        for phonemes, answer in self.train_corpus:
            spelling = []
            answer_length = len(answer)
            alphabet = string.ascii_lowercase
            for i in range(answer_length):
                # 将26个字母和位置结合起来，组成列索引
                result_columns = [al + '_' + str(i) for al in alphabet]
                possible_results = self.phoneme_letter_df.loc[phonemes, result_columns]
                letters_prob = possible_results.sum(axis=0)  # 每一列相加,取概率最大值
                letter = letters_prob.idxmax()
                spelling.append(letter)
            self.student_answer_pair.append([spelling, answer])

    def evaluation(self) -> Tuple[float, float, float]:
        for stu_answer, correct_answer in self.student_answer_pair:
            stu_answer = ''.join([i.split('_')[0] for i in stu_answer])
            correct_answer = ''.join([i.split('_')[0] for i in correct_answer])
            word_accuracy = round(Levenshtein.ratio(correct_answer, stu_answer), 2)
            self.accuracy.append(word_accuracy)
        self.avg_accuracy = sum(self.accuracy) / len(self.accuracy)

        return self.avg_accuracy


if __name__ == "__main__":
    CURRENT_PATH = os.getcwd()  # get the current path
    VOCAB_PATH: str = os.path.join(CURRENT_PATH, 'VocabularyBook', 'CET4', 'vocabulary.json')  # get the vocab data path
    corpus_instance = ReadVocabBook(vocab_book_path=VOCAB_PATH,
                                    vocab_book_name='CET4',
                                    chinese_setting=False,
                                    phonetic_setting=True,
                                    POS_setting=False,
                                    english_setting=True)
    original_corpus = corpus_instance.read_vocab_book()
    # 3206
    random.shuffle(
        original_corpus)  # [['p ɑ p j ʌ l eɪ ʃ ʌ n', 'p o p u l a t i o n'], ['n aɪ n t i n', 'n i n e t e e n']

    corpus_1 = [[item.split() for item in sublist] for sublist in original_corpus]
    letters_1 = []
    phonemes_1 = []
    for sp in corpus_1:
        for fw in sp[0]:  # phoneme
            phonemes_1.append(fw)
        for ew in sp[1]:  # letters
            letters_1.append(ew)
    # covert into lower letter, and omit the duplicated word
    df_column = sorted(list(set(letters_1)), key=lambda s: s.lower())  # 26
    df_index = sorted(list(set(phonemes_1)), key=lambda s: s.lower())  # 39

    """ initialize all prob, all possible included"""
    init_prob = 1 / len(df_column)
    phoneme_letter_prob = pd.DataFrame(init_prob, index=df_index, columns=df_column)

    # phoneme letter pair student
    position_phoLet_student = PositionPhoLetStudent(original_corpus, phoneme_letter_prob, df_index, df_column)
    position_phoLet_student.train_model()
    # 得到的每一行的结果确实是1
    print(position_phoLet_student.phoneme_letter_df)
    position_phoLet_student.phoneme_letter_df.to_excel("standard_IBM_Model_1.xlsx")
    # position_phoLet_student.generate_answer()
    # position_phoLet_accuracy, position_phoLet_completeness, position_phoLet_perfect = position_phoLet_student.evaluation()
    # print(f'position phoLet students accuracy is: {position_phoLet_accuracy}, completeness is: {position_phoLet_completeness}, '
    #       f'perfect is: {position_phoLet_perfect}')
