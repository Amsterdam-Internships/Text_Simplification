'''This script subwords an input file using bpe'''

import argparse
import youtokentome as yttm
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', 
                    help='path to data file that needs to be subworded',
                    required=True)
parser.add_argument('--model_path', 
                    help='path to subwording model', 
                    required=True)
parser.add_argument('--output_path',
                    help='path to store subworded output file',
                    required=True)
args=parser.parse_args()

model_path = args.model_path
input_path = args.input_path
output_path = args.output_path

# Load the BPE model
bpe = yttm.BPE(model_path)

# Open the input and output files
with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
    # Encode each sentence in the input file and write to the output file
    for line in f_in:
        sentence = line.strip()
        encoded = bpe.encode(sentence)
        f_out.write(' '.join(map(str, encoded)) + '\n')

print('Subwording done! output saved to: ' + output_path)