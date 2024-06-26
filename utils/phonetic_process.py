"""
This file is to get the CMU/IPA phonemes information which is one of the important vocabulary information.
NLTK and SpaCy library can possibly label the ‘part of speech’ information.

Step 1: get the CMU phonemes first
Step 2: convert the CMU to IPA phonemes based on dictionary below inferring from http://www.speech.cs.cmu.edu/cgi-bin/cmudict
"""
from nltk.corpus import cmudict

# the whole CMU phonemes
CMU_DICT_PHONEMES = [['P'], ['B'], ['T'], ['D'], ['K'], ['G'], ['CH'], ['JH'], ['F'], ['V'], ['TH'], ['DH'], ['S'],
                     ['Z'], ['SH'], ['ZH'], ['HH'], ['M'], ['N'], ['NG'], ['L'], ['R'], ['W'], ['Y'], ['AA'], ['AE'],
                     ['AH'], ['AO'], ['AW'], ['AY'], ['EH'], ['ER'], ['EY'], ['IH'], ['IY'], ['OW'], ['OY'], ['UH'],
                     ['UW']]

# Dictionary: convert CMU to IPA phonemes
CMU_TO_IPA = {'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ', 'B': 'b', 'CH': 'tʃ', 'D': 'd',
              'DH': 'ð', 'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'F': 'f', 'G': 'ɡ', 'HH': 'h', 'IH': 'ɪ', 'IY': 'i',
              'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ', 'OY': 'ɔɪ', 'P': 'p',
              'R': 'r', 'S': 's', 'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v', 'W': 'w', 'Y': 'j',
              'Z': 'z', 'ZH': 'ʒ'}


# ---------------------------------------get the phonetic of word---------------------------------------------
# Load the CMUDict pronunciation dictionary
phonetic_dict = cmudict.dict()


# remove the stress representatives: 0(no stress),1 (primary stress),2 (secondary stress) present the different stress position separately
def remove_stress(pronunciation):
    if pronunciation[-1].isdigit():
        return pronunciation[:-1]
    else:
        return pronunciation


# a function to get the phonetic components of a word, IPA or CMU
def get_phonemes(word):
    cmu_phonemes = []
    ipa_phonemes = []
    if word.lower() in phonetic_dict:  # if word in this dictionary
        phonemes = phonetic_dict[word.lower()][0]  # present all possible pronunciations, we get the first one
        for phoneme in phonemes:
            phoneme_without_stress = remove_stress(phoneme)  # remove stress label next to phoneme
            ipa_phoneme = CMU_TO_IPA[phoneme_without_stress]  # convert CMU phoneme to IPA phoneme
            cmu_phonemes.append(phoneme_without_stress)  # get the cmu phonemes
            ipa_phonemes.append(ipa_phoneme)  # get the ipa phonemes
    return cmu_phonemes, ipa_phonemes

