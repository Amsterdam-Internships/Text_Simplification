'''This script subwords an input file using bpe'''

import youtokentome as yttm
import sys

model_path = sys.argv[1]
input_path = sys.argv[2]
output_path = sys.argv[3]

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