'''This script trains a bpe model on an input file'''

import argparse
import youtokentome as yttm
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path', 
                    help='path to training data file',
                    required=True)
parser.add_argument('--model_path', 
                    help='path to store subwording model', 
                    required=True)
args=parser.parse_args()

# Set the number of merges (iterations) for the BPE algorithm
train_data_path = args.train_data_path
model_path = args.model_path

# Read the input file and tokenize the sentences
with open(train_data_path, 'r', encoding='utf-8') as f:
    sentences = [line.strip() for line in f.readlines()]
    sentences = [s.split() for s in sentences]
    
# Training model
yttm.BPE.train(data=train_data_path, vocab_size=5000, model=model_path)

# Loading model
bpe = yttm.BPE(model=model_path)


print('Training a model for bpe done! Model saved to: ' + model_path)