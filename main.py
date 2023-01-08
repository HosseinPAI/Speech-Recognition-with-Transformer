import time
import numpy as np
import torch
from model import transformer
from functions import generate_char_number, extract, create_directory
from preparing_data.prepare_data import prepare_valid_data
from train_test import train, valid, test, show_plot
from preparing_data.load_data import load_data
import argparse
import os


if __name__ == '__main__':
    np.seterr(all='ignore')
    # Create Parameters
    parser = argparse.ArgumentParser(description='Speech Recognition with Transformer Model')
    parser.add_argument('--extract_path', type=str, default='./voxforge/enlang/extracted',
                        help='The path of extracted dataset')
    parser.add_argument('--dataset_path', type=str, default='./voxforge/enlang/tgz',
                        help='Dataset path')
    parser.add_argument('--csv_path', type=str, default='',
                        help='Final CSV files path')
    parser.add_argument('--result_path', type=str, default='', help='result path')
    parser.add_argument('--epoch', type=int, default=1, help='Number of epoch')
    parser.add_argument('--batch_size', type=int, default=2, help='The size of each batch')

    # Start Process
    args = parser.parse_args()

    # Create Roots
    current_dir = os.getcwd()
    args.extract_path = current_dir + args.extract_path
    if not os.path.exists(args.extract_path):
        os.makedirs(args.extract_path)

    # Extract All Files in Dataset
    '''
    start_time = time.time()
    print(">>> Starting VoxForge dataset files extraction.")
    files = os.listdir(args.dataset_path)
    for file in files:
        file_path = args.dataset_path + '/' + file
        extract(file_path, args.extract_path)
    print('>>> Extracting Process is Finishes in {0:.1f} min'.format((time.time()-start_time)/60.00))
    '''
    # Create Dataset Diectories to prepare and clean data
    main_dir = create_directory()
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)

    # Prepare and Clean Data
    current_dir = os.getcwd()
    args.csv_path = current_dir + '/csv'
    if not os.path.exists(args.csv_path):
        os.makedirs(args.csv_path)
    #prepare_valid_data(args.csv_path)

    # Create Result Directory
    current_dir = os.getcwd()
    args.result_path = current_dir + '/results'
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    # Initial Variables
    train_loss = np.empty(args.epoch)
    valid_loss = np.empty(args.epoch)
    train_CER = np.empty(args.epoch+1)
    valid_CER = np.empty(args.epoch+1)
    train_WER = np.empty(args.epoch)
    valid_WER = np.empty(args.epoch)

    # If available use GPU memory to load data
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('Selected Device : ', device)

    # Prepare data loaders
    train_manifest_list = [args.csv_path+'/data_train.csv']
    valid_manifest_list = [args.csv_path+'/data_val.csv']
    test_manifest_list = [args.csv_path+'/data_test.csv']
    train_loader, valid_loader_list, test_loader_list = load_data(train_manifest_list,
                                                                  valid_manifest_list,
                                                                  test_manifest_list,
                                                                  batch_size=args.batch_size)

    # Create Model
    print('=========================================================================================')
    print('                               Training Process Started                                  ')
    print('=========================================================================================')
    num_of_char, label2id, id2label = generate_char_number()
    model = transformer.Transformer(dim_input=672, dim_model=256, num_heads=8, dim_key=64,
                                    dim_value=64, dim_inner=1024, output_dim=num_of_char).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    step_num = 1
    train_CER[0] = 1
    valid_CER[0] = 1

    for i in range(args.epoch):
        train_loss[i], train_CER[i+1], train_WER[i], step_num = train(train_loader, model, device, optimizer,
                                                                      i+1, step_num, id2label)
        valid_loss[i], valid_CER[i+1], valid_WER[i] = valid(valid_loader_list[0], model, device, id2label)

    # Save Model
    torch.save(model.state_dict(), args.result_path+'/Transformer_Model.pt')

    # Plot Results
    show_plot(train_loss, valid_loss, train_CER, valid_CER, train_WER, valid_WER, args.result_path)

    # Test 10 Senteces to predict
    test(test_loader_list[0], model, device, id2label, 10)
