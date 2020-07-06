# Personik test case

This repo contains script for text classification by 4 classes:
'VACATION-REQUEST', 'SALARY-REQUEST', 'SICK-LEAVE-REPORT', 'OTHER'.

Language: Russian

text_classification folder consists of:
- data folder (training and test datasets)
- experiments.ipynb is jupyter notebook with experiments
- fasttext_mlpclassifier.joblib saved model
- inference.py is main script

## Getting Started

Clone this repo.

```shell script
git clone https://github.com/amamonova/personik_testcase
```

### Prerequisites
For text representation FastText embeddings are used. 
So, you should download the pre-trained model. 

Change directory to personik_testcase/text_classification/data folder.
And download the model:

```shell script
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz
gunzip cc.ru.300.bin.gz
```  

It will take some time. Also, note, that this model needs 7G.

Download all requirements:

```shell script
pip install -r requirements.txt 
```  

## Running the script

To start classifier:

```shell script
cd text_classification
python inference.py
```

To end script run `ctrl+d`. 

