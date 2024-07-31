"""
1: 无论是从音标长度，单词长度，错误字母数，正确字母数，字母出现的频率，全部KL散度，还是部分KL散度都无法有明显的规律
2：相反莱温斯特比率在绝大多数情况下都可以有一个比较好的结果，但是不是最优的
3：因此如何在莱温斯特比例上做好文章比较重要，可惜的是一直没有定位字母
4：通过这几天的实验可以得出，一定是某些字母对整体的记忆提高比较好
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def entropy_weight(matrix):
    normalized_matrix = matrix / matrix.sum(axis=0)
    entropy = -np.nansum(normalized_matrix * np.log(normalized_matrix + 1e-10), axis=0) / np.log(matrix.shape[0])
    original_weights = (1 - entropy) / (1 - entropy).sum()
    # 前期单词长度长有优势，后期单词长度短有优势
    adjusted_weights = original_weights.copy()
    adjusted_weights[2] = 0.9
    adjusted_weights = adjusted_weights * (1 / adjusted_weights.sum())  # 重新归一化权重
    return adjusted_weights


def calculate_topsis_weight(matrix, weight):
    weighted_matrix = matrix * weight
    ideal_solution = np.max(weighted_matrix, axis=0)
    negative_ideal_solution = np.min(weighted_matrix, axis=0)
    distance_to_ideal = np.sqrt(np.sum((weighted_matrix - ideal_solution) ** 2, axis=1))
    distance_to_negative_ideal = np.sqrt(np.sum((weighted_matrix - negative_ideal_solution) ** 2, axis=1))
    relative_closeness = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)
    return relative_closeness


def topsis(new_information):
    # 经过测试MinMax效果最好
    scale = MinMaxScaler()
    df = pd.DataFrame.from_dict(new_information, orient='index',
                                columns=['wrong_ratio', 'phoneme_length', 'accuracy', 'word_length'])

    df['wrong_ratio_inverse'] = df['wrong_ratio'].max() - df['wrong_ratio']
    df['phoneme_length_inverse'] = df['phoneme_length'].max() - df['phoneme_length']
    df['word_length_inverse'] = df['word_length'].max() - df['word_length']
    df['wrong_ratio_inverse'] = df['wrong_ratio'].max() - df['wrong_ratio']
    # 错误率有好，错误率取小优，取平均稳，最后决定取平均。
    # 相似度有好，越低越好是确定的
    # 音标长度有好，无法确定，取平均最优
    # 单词长度有好，越长越好，取平均也可取
    features = ['wrong_ratio', 'phoneme_length', 'accuracy', 'word_length']
    features_inverse = ['wrong_ratio_inverse', 'phoneme_length_inverse', 'accuracy', 'word_length']

    new_df = df[features]
    new_df_inverse = df[features_inverse]

    X_selected = scale.fit_transform(new_df)
    X_selected_inverse = scale.fit_transform(new_df_inverse)

    weight_normal = entropy_weight(X_selected)
    weight_inverse = entropy_weight(X_selected_inverse)
    relative_closeness = calculate_topsis_weight(X_selected, weight_normal)
    relative_closeness_inverse = calculate_topsis_weight(X_selected_inverse, weight_inverse)

    # 将相对接近度添加到DataFrame
    df['relative_closeness_normal'] = relative_closeness
    df['relative_closeness_inverse'] = relative_closeness_inverse
    # 经过测试mean是最好的
    df['relative_closeness'] = df[['relative_closeness_normal', 'relative_closeness_inverse']].mean(axis=1)
    # 按相对接近度排序
    top_pairs = df.sort_values(by='relative_closeness', ascending=False)
    # 获取选择的索引和相对接近度列，并转换为字典
    word_contribution = top_pairs['relative_closeness'].to_dict()
    return word_contribution


class TopsisAlogorithm:
    """ KL divergence"""

    def __init__(self, observation, feedback, review_words_number):
        self.observation = observation  # the legal action
        self.feedback = feedback  # the feedback
        self.acc_data = {}  # the accuracy
        self.letter_inf = {}  # the letters information
        self.wrong_ratio_list = []
        self.review_words_number = review_words_number

    def extract_feedback(self):
        """extract the accuracy information"""
        for key, values in self.feedback[0].items():
            self.acc_data[key] = values[1]
            self.letter_inf[key] = values[0]

    def extract_wrong_letter(self, task):
        """extract the wrong letters
            只关心错误的字母数，正确的字母数，还有错误字母的
        """
        letter_inf = self.letter_inf[task]
        wrong_num = 0
        right_num = 0
        wrong_letters = []
        correct_letters = []
        word = ''.join(task[1].split(' '))
        current_position = 0
        for letter_mark in letter_inf:
            letter, mark = letter_mark.split('_')
            if mark == '0':
                wrong_num += 1
                wrong_letters.append(word[current_position] + '_' + str(current_position))
            else:
                right_num += 1
                correct_letters.append(word[current_position] + '_' + str(current_position))
            current_position += 1
        wrong_ratio = wrong_num / (wrong_num + right_num)
        return wrong_ratio, wrong_num, right_num, wrong_letters, correct_letters

    def construct_information(self, excellent_dataframe, forget_dataframe):
        # 对所有的单词计算，准确度，错误的单词, 音标长度，字母长度,如何综合考虑比只考虑准确度高呢？
        self.extract_feedback()
        new_information = {}

        for task in self.observation:
            wrong_ratio, wrong_num, right_num, wrong_letters, correct_letters = self.extract_wrong_letter(tuple(task))
            self.wrong_ratio_list.append(wrong_num)
        for index in range(len(self.observation)):
            phoneme_length = len(self.observation[index][0].split(' '))
            word_length = len(self.observation[index][1].split(' '))
            new_information[tuple(self.observation[index])] = (self.wrong_ratio_list[index],
                                                               phoneme_length,
                                                               (1 - self.acc_data[tuple(self.observation[index])]),
                                                               word_length)
        return new_information

    def reward_function(self, excellent_dataframe, forget_dataframe):
        """get the word contribution """
        information = self.construct_information(excellent_dataframe, forget_dataframe)
        word_contribution = topsis(information)
        return word_contribution
