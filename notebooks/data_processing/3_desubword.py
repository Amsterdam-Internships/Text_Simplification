'''this script desubwords an input file'''

import argparse
import youtokentome as yttm
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', 
                    help='path to data file that needs to be desubworded',
                    required=True)
parser.add_argument('--model_path', 
                    help='path to subwording model',
                    required=True)
parser.add_argument('--output_path',
                    help='path to store desubworded output file',
                    required=True)
args=parser.parse_args()

model_path = args.model_path
input_file = args.input_path
output_path = args.output_path

#Loading model
bpe = yttm.BPE(model=model_path)


# Read the input file and tokenize the sentences
with open(input_file, 'r', encoding='utf-8') as f:
    subwords = [line.strip() for line in f.readlines()]
    subwords = [s.split() for s in subwords]
    subwords = [[int(x) for x in sublist if x != '<unk>'] for sublist in subwords]

print('='*100)

# Decode the subwords using the BPE model
sentences = []
for s in subwords:
    sentence = bpe.decode(s, ignore_ids=[0])
    sentence = ''.join(sentence)
    sentences.append(sentence)

# Write the desubworded sentences to a new file
with open(output_path, 'w', encoding='utf-8') as f:
    for s in sentences:
        f.write(s + '\n')


print("desubwording done! Output saved to: " + output_path)