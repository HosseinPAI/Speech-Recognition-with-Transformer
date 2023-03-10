import unicodedata
import os
import re

dir_path = os.path.dirname(os.path.realpath(__file__))


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
def load_stanford_core_nlp(path):
    from stanfordcorenlp import StanfordCoreNLP
    """
    Load stanford core NLP toolkit object
    args:
        path: String
    output:
        Stanford core NLP objects
    """
    zh_nlp = StanfordCoreNLP(path, lang='zh')
    en_nlp = StanfordCoreNLP(path, lang='en')
    return zh_nlp, en_nlp


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
def is_chinese_char(cc):
    return unicodedata.category(cc) == 'Lo'


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
def is_contain_chinese_word(seq):
    for i in range(len(seq)):
        if is_chinese_char(seq[i]):
            return True
    return False


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
def get_word_segments_per_language(seq):
    cur_lang = -1  # cur_lang = 0 (english), 1 (chinese)
    words = seq.split(" ")
    temp_words = ""
    word_segments = []

    for i in range(len(words)):
        word = words[i]

        if is_contain_chinese_word(word):
            if cur_lang == -1:
                cur_lang = 1
                temp_words = word
            elif cur_lang == 0:  # english
                cur_lang = 1
                word_segments.append(temp_words)
                temp_words = word
            else:
                if temp_words != "":
                    temp_words += " "
                temp_words += word
        else:
            if cur_lang == -1:
                cur_lang = 0
                temp_words = word
            elif cur_lang == 1:  # chinese
                cur_lang = 0
                word_segments.append(temp_words)
                temp_words = word
            else:
                if temp_words != "":
                    temp_words += " "
                temp_words += word

    word_segments.append(temp_words)

    return word_segments


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
def get_word_segments_per_language_with_tokenization(seq, tokenize_lang=-1, zh_nlp=None, en_nlp=None):
    cur_lang = -1
    words = seq.split(" ")
    temp_words = ""
    word_segments = []

    for i in range(len(words)):
        word = words[i]

        if is_contain_chinese_word(word):
            if cur_lang == -1:
                cur_lang = 1
                temp_words = word
            elif cur_lang == 0:  # english
                cur_lang = 1

                if tokenize_lang == 0:
                    word_list = en_nlp.word_tokenize(temp_words)
                    temp_words = ' '.join(word for word in word_list)

                word_segments.append(temp_words)
                temp_words = word
            else:
                if temp_words != "":
                    temp_words += " "
                temp_words += word
        else:
            if cur_lang == -1:
                cur_lang = 0
                temp_words = word
            elif cur_lang == 1:  # chinese
                cur_lang = 0

                if tokenize_lang == 1:
                    word_list = zh_nlp.word_tokenize(temp_words.replace(" ", ""))
                    temp_words = ' '.join(word for word in word_list)

                word_segments.append(temp_words)
                temp_words = word
            else:
                if temp_words != "":
                    temp_words += " "
                temp_words += word

    if tokenize_lang == 0 and cur_lang == 0:
        word_list = en_nlp.word_tokenize(temp_words)
        temp_words = ' '.join(word for word in word_list)
    elif tokenize_lang == 1 and cur_lang == 1:
        word_list = zh_nlp.word_tokenize(temp_words)
        temp_words = ' '.join(word for word in word_list)

    word_segments.append(temp_words)

    return word_segments


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
def remove_emojis(seq):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    seq = emoji_pattern.sub(r'', seq).strip()
    return seq


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
def merge_abbreviation(seq):
    seq = seq.replace("  ", " ")
    words = seq.split(" ")
    final_seq = ""
    temp = ""
    for i in range(len(words)):
        word_length = len(words[i])
        if word_length == 0:  # unknown character case
            continue

        if words[i][word_length - 1] == ".":
            temp += words[i]
        else:
            if temp != "":
                if final_seq != "":
                    final_seq += " "
                final_seq += temp
                temp = ""
            if final_seq != "":
                final_seq += " "
            final_seq += words[i]
    if temp != "":
        if final_seq != "":
            final_seq += " "
        final_seq += temp
    return final_seq


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
def remove_punctuation(seq):
    seq = re.sub("[\s+\\!\/_,$%=^*?:@&^~`(+\"]+|[+???????????????~@#???%??????&*??????:;????????????????????????()????????]+", " ", seq)
    seq = seq.replace(" ' ", " ")
    seq = seq.replace(" ??? ", " ")
    seq = seq.replace(" ??? ", " ")
    seq = seq.replace(" ` ", " ")

    seq = seq.replace(" '", "'")
    seq = seq.replace(" ???", "???")
    seq = seq.replace(" ???", "???")

    seq = seq.replace("' ", " ")
    seq = seq.replace("??? ", " ")
    seq = seq.replace("??? ", " ")
    seq = seq.replace("` ", " ")
    seq = seq.replace(".", "")

    seq = seq.replace("`", "")
    seq = seq.replace("-", " ")
    seq = seq.replace("?", " ")
    seq = seq.replace(":", " ")
    seq = seq.replace(";", " ")
    seq = seq.replace("]", " ")
    seq = seq.replace("[", " ")
    seq = seq.replace("}", " ")
    seq = seq.replace("{", " ")
    seq = seq.replace("|", " ")
    seq = seq.replace("_", " ")
    seq = seq.replace("(", " ")
    seq = seq.replace(")", " ")
    seq = seq.replace("=", " ")

    seq = seq.replace(" dont ", " don't ")
    # seq = seq.replace("welcome?????????", "welcome ?????????")
    seq = seq.replace("doens't", "doesn't")
    seq = seq.replace("o' clock", "o'clock")
    # seq = seq.replace("??????it's", "?????? it's")
    seq = seq.replace("it' s", "it's")
    seq = seq.replace("it ' s", "it's")
    seq = seq.replace("it' s", "it's")
    seq = seq.replace("y'", "y")
    seq = seq.replace("y ' ", "y")
    # seq = seq.replace("???different", "??? different")
    seq = seq.replace("it'self", "itself")
    seq = seq.replace("it'ss", "it's")
    seq = seq.replace("don'r", "don't")
    seq = seq.replace("has't", "hasn't")
    seq = seq.replace("don'know", "don't know")
    seq = seq.replace("i'll", "i will")
    seq = seq.replace("you're", "you are")
    seq = seq.replace("'re ", " are ")
    seq = seq.replace("'ll ", " will ")
    seq = seq.replace("'ve ", " have ")
    seq = seq.replace("'re\n", " are\n")
    seq = seq.replace("'ll\n", " will\n")
    seq = seq.replace("'ve\n", " have\n")
    seq = remove_space_in_between_words(seq)
    seq = seq.lower()
    # print(seq)
    # print('-------------------------------')
    return seq


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
def remove_special_char(seq):
    seq = re.sub("[??????????????????????????????????????????????????????????????????????????????????????????????????]+", " ", seq)
    return seq


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
def remove_space_in_between_words(seq):
    return seq.replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ").strip().lstrip()


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
def remove_return(seq):
    return seq.replace("\n", "").replace("\r", "").replace("\t", "")


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
def preprocess_mixed_language_sentence(seq, tokenize=False, en_nlp=None, zh_nlp=None, tokenize_lang=-1):
    if len(seq) == 0:
        return ""

    seq = seq.lower()
    seq = merge_abbreviation(seq)
    seq = seq.replace("\x7f", "")
    seq = seq.replace("\x80", "")
    seq = seq.replace("\u3000", " ")
    seq = seq.replace("\xa0", "")
    seq = seq.replace("[", " [")
    seq = seq.replace("]", "] ")
    seq = seq.replace("#", "")
    seq = seq.replace(",", "")
    seq = seq.replace("*", "")
    seq = seq.replace("\n", "")
    seq = seq.replace("\r", "")
    seq = seq.replace("\t", "")
    seq = seq.replace("~", "")
    seq = seq.replace("???", "")
    seq = seq.replace("  ", " ").replace("  ", " ")
    seq = re.sub('\<.*?\>', '', seq)  # REMOVE < >
    seq = re.sub('\???.*?\???', '', seq)  # REMOVE ??? ???
    seq = re.sub("[\(\[].*?[\)\]]", "", seq)  # REMOVE ALL WORDS WITH BRACKETS (HESITATION)
    seq = re.sub("[\{\[].*?[\}\]]", "", seq)  # REMOVE ALL WORDS WITH BRACKETS (HESITATION)
    seq = remove_special_char(seq)
    seq = remove_space_in_between_words(seq)
    seq = seq.strip()
    seq = seq.lstrip()

    seq = remove_punctuation(seq)

    temp_words = ""
    if not tokenize:
        segments = get_word_segments_per_language(seq)
    else:
        segments = get_word_segments_per_language_with_tokenization(seq, en_nlp=en_nlp, zh_nlp=zh_nlp,
                                                                    tokenize_lang=tokenize_lang)

    for j in range(len(segments)):
        # if not is_contain_chinese_word(segments[j]):
        #     segments[j] = re.sub(r'[^\x00-\x7f]',r' ',segments[j])

        if temp_words != "":
            temp_words += " "
        temp_words += segments[j].replace("\n", "")
    seq = temp_words

    seq = remove_space_in_between_words(seq)
    seq = seq.strip()
    seq = seq.lstrip()

    # Tokenize chinese characters
    if len(seq) <= 1:
        return ""
    else:
        return seq
