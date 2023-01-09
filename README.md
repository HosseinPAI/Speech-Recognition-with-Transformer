## Speech Recognition: Speech to Text with Transformer Model

<p align="justify">
The <strong>Transformer</strong> model is an intense deep model used in various fields, such as machine translation, speech-to-text, and even vision area. In this <strong>PyTorch</strong> project, we use it to convert speech to text. The structure of a <strong>Transformer</strong> model is based on Encoder-Decoder and attention procedure. The whole structure of this model is shown below.
</p>

<br />
<p align="center">
<img src="https://github.com/HosseinPAI/Speech-Recognition-with-Transformer/blob/master/.idea/pics/transformer_arch.png" alt="Pyramid Model" width="450" height='500'/>
</p>

<p align="justify">
It consists of an encoder and a decoder. In this project, we omitted the decoder part due to reducing the model size and implemented the encoder part. A transformer is not like an LSTM model and is not sequential, so it needs to determine the position of each input feature. Therefore, the positional encoding part after the input embedding block is responsible for this task. 

The essential part of a transformer model is the <strong>Multi-Head Attention</strong>, shown in the picture below. There are three inputs, <strong>Query(Q), Key(K),</strong> and <strong>Value(V)</strong>. In each training step, the similarity of Query and Key is calculated, and then it multiplies to the Value to create a vector that shows the attention to each feature.  
</p>

<p align="center">
<img src="https://github.com/HosseinPAI/Speech-Recognition-with-Transformer/blob/master/.idea/pics/SCALDE.png" alt="Pyramid Model" width="400" height='300'/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/HosseinPAI/Speech-Recognition-with-Transformer/blob/master/.idea/pics/multi-head-attention_l1A3G7a.png" alt="Bert Model" width="350" height='300'/>
</p>

<p align="justify">
Some of these blocks make an encoder layer, and we can stack many to create the encoder part.  
</p>

<p align="center">
<img src="https://github.com/HosseinPAI/Speech-Recognition-with-Transformer/blob/master/.idea/pics/encoder_sub.png" alt="Pyramid Model" width="350" height='400'/>
</p>

<p align="justify">
In this project, we use <a href="https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html">CTC Loss</a> (The Connectionist Temporal Classification Loss). It calculates loss between a continuous (unsegmented) time series and a target sequence. CTCLoss sums over the probability of possible alignments of input to the target, producing a loss value that is differentiable with respect to each input node. The alignment of input to target is assumed to be “many-to-one,” which limits the length of the target sequence such that it must be ≤ the input length.
</p>

### How to run this project:
<p align="justify">
You should follow these few steps to show the result, but we should be aware that the dataset size is massive, so training the model takes a lot of time, which depends on the number of epochs.
</p>

1. There is a **.sh** file in the project folder. Running the following command in the terminal downloads the [Voxforge dataset](http://www.voxforge.org/home/Downloads) into the project directory. It may take a couple of hours because it is more than 10 GB.
    ```
    bash download.sh
    ```
2. After downloading the dataset finished, running the below command starts all necessary procedures to train, validate, and test the model. Some directories are created during this process, and you can access everything you need, such as data CSV files, plots, and the saved model.
    ```
    python main.py --batch_size=12 --epoch=20
    ```
  
<p align="justify">
You can change the number of <strong>epochs</strong> and <strong>batch</strong> size based on your GPU power. You must train this model for <strong>at least 20 epochs </strong>to see the acceptable result.
</p>
<br />
<p align="justify">
All necessary libraries are written in the <strong>requirements.txt</strong> file.  
</p>


