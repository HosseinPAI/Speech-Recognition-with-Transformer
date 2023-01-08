import torch.nn.functional as F
import torch
import numpy as np
import json
import re
import Levenshtein
import tarfile
import os


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
def generate_char_number():
    PAD_CHAR = "¶"
    SOS_CHAR = "§"
    EOS_CHAR = "¤"

    labels_path = './labels.json'
    with open(labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))

    # add PAD_CHAR, SOS_CHAR, EOS_CHAR
    labels = PAD_CHAR + SOS_CHAR + EOS_CHAR + labels
    label2id, id2label = {}, {}
    count = 0
    for i in range(len(labels)):
        if labels[i] not in label2id:
            label2id[labels[i]] = count
            id2label[count] = labels[i]
            count += 1

    return count, label2id, id2label


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
def CTC_Loss(Model_output, Target, input_sizes, target_sizes):
    # Change B*T*C -> T*B*C
    Model_output = Model_output.transpose(0, 1)
    loss = F.ctc_loss(Model_output, Target, input_sizes, target_sizes)

    return loss


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
def learning_rate_decay(step_num, dim_model=256, warmup_steps=4000):
    return (dim_model ** -0.5) * min(step_num ** -0.5, step_num * (warmup_steps ** -1.5))


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------#
def correct_input_sizes(input_sizes):
    input_sizes = (input_sizes - 9) / 2
    # input_sizes[input_sizes>max_size] = max_size
    return input_sizes


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
def indeces_to_sentence(output, target, target_sizes, id2label, batch_size=6):
    pred_best_ids = torch.argmax(output, axis=2)
    tar_sentence = []
    pred_sentence = []

    for i in range(batch_size):
        target_senctence = ''
        for j in range(target_sizes[i]):
            target_senctence += id2label[int(target[i][j])]
        target_senctence = re.sub("\¶|\§|\¤|\_", "", target_senctence)
        target_senctence = target_senctence.strip()
        tar_sentence.append(target_senctence)

        predict_senctence = ''
        for j in range(output.size(1)):
            predict_senctence += id2label[int(pred_best_ids[i][j])]

        predict_senctence = re.sub(r'(.)\1+', r'\1', predict_senctence)
        predict_senctence = re.sub("\¶|\§|\¤|\_", "", predict_senctence)
        predict_senctence = predict_senctence.strip()
        pred_sentence.append(predict_senctence)

    return tar_sentence, pred_sentence


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
def cer_wer_calculator(output, target, target_sizes, id2label, batch_size=6):
    targert_sentence, predict_sentence = indeces_to_sentence(output, target, target_sizes, id2label, batch_size)

    # calculate character error rate
    num_of_chars = 0
    char_distance = 0
    cer = 0.0
    for i in range(batch_size):
        tar_sentence_without_space = re.sub(" ", "", targert_sentence[i])
        pre_sentence_without_space = re.sub(" ", "", predict_sentence[i])
        num_of_chars += len(tar_sentence_without_space)
        char_distance += Levenshtein.distance(tar_sentence_without_space, pre_sentence_without_space)

    cer = float(char_distance / num_of_chars)
    if cer > 1.0:
        cer = 1.0

    # calculate word error rate
    num_of_words = 0
    total_word_error = 0
    wer = 0.0

    for i in range(batch_size):
        tar_sentence = targert_sentence[i].strip()
        pre_sentence = predict_sentence[i].strip()
        tar_list_of_words = tar_sentence.split(' ')
        pre_list_of_words = pre_sentence.split(' ')
        num_of_words += len(tar_list_of_words)
        total_word_error += wer_cal(tar_list_of_words, pre_list_of_words)

    wer = float(total_word_error / num_of_words)
    if wer > 1.0:
        wer = 1.0

    return cer, wer


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
def wer_cal(sentence1, sentence2):
    d = np.zeros((len(sentence1) + 1) * (len(sentence2) + 1), dtype=np.uint8)
    d = d.reshape((len(sentence1) + 1, len(sentence2) + 1))
    for i in range(len(sentence1) + 1):
        for j in range(len(sentence2) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(sentence1) + 1):
        for j in range(1, len(sentence2) + 1):
            if sentence1[i - 1] == sentence2[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(sentence1)][len(sentence2)]


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
def extract(tar_path, extract_path):
    tar = tarfile.open(tar_path, 'r')
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract(item.name, "./" + item.name[:item.name.rfind('/')])


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
def create_directory():
    current_dir = os.getcwd()
    # Main directory
    main_dir = current_dir + '/data'
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)

    # Train directory
    train_dir = main_dir+'/train'
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(train_dir+'/wav'):
        os.makedirs(train_dir+'/wav')

    if not os.path.exists(train_dir+'/txt'):
        os.makedirs(train_dir+'/txt')

    # Test directory
    test_dir = main_dir + '/test'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    if not os.path.exists(test_dir + '/wav'):
        os.makedirs(test_dir + '/wav')

    if not os.path.exists(test_dir + '/txt'):
        os.makedirs(test_dir + '/txt')

    # Valid directory
    valid_dir = main_dir + '/val'
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)

    if not os.path.exists(valid_dir + '/wav'):
        os.makedirs(valid_dir + '/wav')

    if not os.path.exists(valid_dir + '/txt'):
        os.makedirs(valid_dir + '/txt')

    return main_dir
