import torch
import time
import matplotlib.pyplot as plt
import re
import Levenshtein
from functions import CTC_Loss, learning_rate_decay, cer_wer_calculator


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
def train(train_loader, model, device, optimizer, epoch, step_num, id2label, dim_model=256):
    total_loss = 0
    total_batch = 0
    CER = 0
    WER = 0
    start_time = time.time()
    print('Training Epoch: {}'.format(epoch))

    model.train()
    for batch_idx, (inputs, targets, input_percentages, input_sizes, target_sizes) in enumerate(train_loader):
        if inputs.size(3) < 2100:  # for manage Cuda
            if batch_idx % 150 == 0:
                print('.', end='')

            inputs = inputs.to(device)
            targets = targets.to(device)
            input_sizes = input_sizes.to(device)
            target_sizes = target_sizes.to(device)

            optimizer.zero_grad()
            output, mask_input_sizes = model(inputs, input_sizes, device)

            # Calculate Loss
            loss = CTC_Loss(output, targets, mask_input_sizes, target_sizes)
            loss.backward()

            learning_rate = learning_rate_decay(step_num, dim_model=dim_model, warmup_steps=4000)
            for param in optimizer.param_groups:
                param['lr'] = learning_rate
            optimizer.step()

            # CER and WER Calculation
            cer_temp, wer_temp = cer_wer_calculator(output.cpu(), targets.cpu(), target_sizes.cpu(), id2label,
                                                    output.size(0))
            CER += cer_temp
            WER += wer_temp

            total_loss += loss.item()
            step_num += 1
            total_batch += 1

    total_loss /= total_batch
    CER /= total_batch
    WER /= total_batch

    print("\nTrain Loss: {:.4f}\t Train CER: {:.2f}\t Train WER: "
          "{:.2f}\t Training   Time: {:.2f} min".format(total_loss, CER, WER, (time.time() - start_time) / 60.00))

    return total_loss, CER, WER, step_num


##################################################################################################
##################################################################################################
##################################################################################################
def valid(valid_loader, model, device, id2label):
    total_loss = 0
    total_batch = 0
    CER = 0
    WER = 0
    start_time = time.time()

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets, input_percentages, input_sizes, target_sizes) in enumerate(valid_loader):
            if inputs.size(3) < 2100:  # for manage Cuda
                inputs = inputs.to(device)
                targets = targets.to(device)
                input_sizes = input_sizes.to(device)
                target_sizes = target_sizes.to(device)

                output, mask_input_sizes = model(inputs, input_sizes, device)
                # Calculate Loss
                loss = CTC_Loss(output, targets, mask_input_sizes, target_sizes)
                # -----------------------------------------------------------
                # CER and WER Calculation
                cer_temp, wer_temp = cer_wer_calculator(output.cpu(), targets.cpu(), target_sizes.cpu(), id2label,
                                                        output.size(0))
                CER += cer_temp
                WER += wer_temp

                total_loss += loss.item()
                total_batch += 1

        total_loss /= total_batch
        CER /= total_batch
        WER /= total_batch

        print("Valid Loss: {:.4f}\t Valid CER: {:.2f}\t Valid WER: {:.2f}\t Validation Time: "
              "{:.2f} min".format(total_loss, CER, WER, (time.time() - start_time) / 60.00))
        print("====================================================================================================")

        return total_loss, CER, WER


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
def test(test_loader, model, device, id2label, num_of_sample=10):
    sample = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets, input_percentages, input_sizes, target_sizes) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            input_sizes = input_sizes.to(device)
            target_sizes = target_sizes.to(device)

            output, mask_input_sizes = model(inputs, input_sizes, device)
            # -----------------------------------------------------------
            # Show Result base of Best Pass Decoding
            pred_best_ids = torch.argmax(output, axis=2)
            target_senctence = ''
            for j in range(target_sizes[0]):
                target_senctence += id2label[int(targets[0][j])]

            target_senctence = re.sub("\¶|\§|\¤|\_", "", target_senctence)
            target_senctence = target_senctence.strip()

            predict_senctence = ''
            for j in range(mask_input_sizes[0]):
                predict_senctence += id2label[int(pred_best_ids[0][j])]

            predict_senctence = re.sub(r'(.)\1+', r'\1', predict_senctence)
            predict_senctence = re.sub("\¶|\§|\¤|\_", "", predict_senctence)
            predict_senctence = predict_senctence.strip()

            tar_sentence_without_space = re.sub(" ", "", target_senctence)
            pre_sentence_without_space = re.sub(" ", "", predict_senctence)
            num_of_chars = len(tar_sentence_without_space)
            char_distance = Levenshtein.distance(tar_sentence_without_space, pre_sentence_without_space)
            print("Targer Sentence  : ", target_senctence)
            print("Predict Sentence : ", predict_senctence)
            print("======================================================================")
            sample += 1

            if sample == num_of_sample:
                break
    return


# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
def show_plot(train_loss, valid_loss, train_CER, valid_CER, train_WER, valid_WER, save_path):
    plt.figure(figsize=(10, 6), dpi=100)
    plt.title('Train vs Valid Loss', color='darkblue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(train_loss, color='blue', label='Train Loss')
    plt.plot(valid_loss, color='orange', label='Valid Loss')
    plt.legend()
    plt.savefig(save_path+'/Loss.jpg', dpi=200)

    plt.figure(figsize=(10, 6), dpi=100)
    plt.title('Train vs Valid Character Error Rate (CER)', color='darkblue')
    plt.xlabel('Epoch')
    plt.ylabel('Character Error Rate')
    plt.plot(train_CER, color='blue', label='Train CER')
    plt.plot(valid_CER, color='orange', label='Valid CER')
    plt.legend()
    plt.savefig(save_path+'/CER.jpg', dpi=200)

    plt.figure(figsize=(10, 6), dpi=100)
    plt.title('Train vs Valid Word Error Rate (WER)', color='darkblue')
    plt.xlabel('Epoch')
    plt.ylabel('Word Error Rate')
    plt.plot(train_WER, color='blue', label='Train WER')
    plt.plot(valid_WER, color='orange', label='Valid WER')
    plt.legend()
    plt.savefig(save_path+'/WER.jpg', dpi=200)

