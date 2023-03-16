'''This script trains a bpe model on an input file'''

import youtokentome as yttm
import sys

# Set the number of merges (iterations) for the BPE algorithm
train_data_path = sys.argv[1]
model_path = sys.argv[2]

# Read the input file and tokenize the sentences
with open(train_data_path, 'r', encoding='utf-8') as f:
    sentences = [line.strip() for line in f.readlines()]
    sentences = [s.split() for s in sentences]
    
# Training model
yttm.BPE.train(data=train_data_path, vocab_size=5000, model=model_path)

# Loading model
bpe = yttm.BPE(model=model_path)


print('Training a model for bpe done! Model saved to: ' + model_path)