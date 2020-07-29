## This is a step-by-step implementation of Chinese Title Entity Recognition via BERT-BiLSTM-CRF Model.

For original codes and tutorials, please visit [here](https://github.com/macanv/BERT-BiLSTM-CRF-NER).<br> 
For Chinese readers, please visit [here](https://blog.csdn.net/macanv/article/details/85684284).<br>

My job lies in the Chinese chat title named entity recognition by fine-tuning the BERT model.<br>
So I changed several lines of codes and extended more Chinese chat title entities to the original dataset.<br>
96.73% accuracy has been achieved via the BERT-BiLSTM-CRF.<br>

The main purpose of the job（examples）:

```
Input One: 小贾你最近忙什么呢？<br>
Input Two: 贾舒越<br>
Output: 小贾 is 贾舒越

Input One: 建勋师兄你何时来实验室？<br>
Input Two: 邸建勋<br>
Output: 建勋师兄 is 王建勋

Input One: 最近王宇航学习怎么样呀<br>
Input Two: 王海生<br>
Output: There is no match for 王海生.

Input One: 贾泽阳现在回家了嘛<br>
Input Two: 吴泽阳<br>
Output: There is no match for 吴泽阳.
```

For Chinese readers, you guys could read the [提取聊天对方的称谓 - 方案与deadline.pdf](https://github.com/SuperBruceJia/Chinese-Chat-Title-NER-BERT-BiLSTM-CRF/blob/master/%E6%8F%90%E5%8F%96%E8%81%8A%E5%A4%A9%E5%AF%B9%E6%96%B9%E7%9A%84%E7%A7%B0%E8%B0%93%20-%20%E6%96%B9%E6%A1%88%E4%B8%8Edeadline.pdf) to get into the details.<br> 

### Step One: configure the tensorflow and bert environment<br>

```
pip install bert-base==0.0.7 -i https://pypi.python.org/simple<br>
tensorflow >= 1.12.0<br> 
tensorflow-gpu >= 1.12.0  # GPU version of TensorFlow.<br> 
GPUtil >= 1.3.0  # no need if you dont have GPU<br> 
pyzmq >= 17.1.0  # python zmq<br> 
```

### Step Two: Download the BERT pre-trained model and training dataset<br>
Download the BERT pre-trained model from [here](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip).<br>
Be sure to place the extracted folder "chinese_L-12_H-768_A-12" on "init_checkpoint" folder.<br>
training dataset from [here](https://drive.google.com/file/d/1bncNHl_E1KBihCNUejYW_PGYxBv0UGs6/view?usp=sharing).<br>
Be sure to place "train.txt" on the "data" folder.<br>

### Step Three: Train the model via command line<br>
Open the CMD terminal or the Anaconda Prompt and be sure to guide it to the working path and tensorflow environment:<br>
e.g. my working path is /Users/shuyuej/Desktop/Python-Files/Chinese-Chat-Title-NER-BERT-BiLSTM-CRF/.

Then input the command: 

```
bert-base-ner-train -data_dir /Users/shuyuej/Desktop/Python-Files/BERT-BiLSTM-CRF-NER/data/ -output_dir /Users/shuyuej/Desktop/Python-Files/BERT-BiLSTM-CRF-NER/final_output/ -init_checkpoint /Users/shuyuej/Desktop/Python-Files/BERT-BiLSTM-CRF-NER/init_checkpoint/chinese_L-12_H-768_A-12\bert_model.ckpt -bert_config_file /Users/shuyuej/Desktop/Python-Files/BERT-BiLSTM-CRF-NER/init_checkpoint/chinese_L-12_H-768_A-12/bert_config.json -vocab_file /Users/shuyuej/Desktop/Python-Files/BERT-BiLSTM-CRF-NER/init_checkpoint/chinese_L-12_H-768_A-12/vocab.txt -batch_size 8
```

FYI, be sure to change my "/Users/shuyuej/Desktop/Python-Files/BERT-BiLSTM-CRF-NER/" to your own BERT-BiLSTM-CRF-NER path.<br>

For Windows OS System: I use the following command line:

```
bert-base-ner-train -data_dir E:\BERT-BiLSTM-CRF-NER\data\ -output_dir E:\BERT-BiLSTM-CRF-NER\final_output\ -init_checkpoint E:\BERT-BiLSTM-CRF-NER\init_checkpoint\chinese_L-12_H-768_A-12\bert_model.ckpt -bert_config_file E:\BERT-BiLSTM-CRF-NER\init_checkpoint\chinese_L-12_H-768_A-12\bert_config.json -vocab_file E:\BERT-BiLSTM-CRF-NER\init_checkpoint\chinese_L-12_H-768_A-12\vocab.txt -batch_size 8
```

The final trained model will be in the "final_output" folder.

### Step Four: Test and enjoy the model<br>
The Test File is "test.py" and could test the "Test-set.xlsx" and get a result.<br>
Before you execute the file, be sure to change the paths of trained BERT model, original pre-trained BERT model, and Test-set.xlsx to your own.<br>
And you could see the results and power of BERT.

Another executive file is "predict-test.py" in which you could input the sentence and name and finally get the match results. Be sure to change the paths same as "test.py" file.<br>
