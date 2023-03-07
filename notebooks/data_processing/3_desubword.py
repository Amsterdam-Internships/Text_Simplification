
import youtokentome as yttm
import sys

model_path = sys.argv[1]
input_file = sys.argv[2]
output_path = sys.argv[3]

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