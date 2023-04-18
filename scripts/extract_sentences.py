'''
This script extracts sentences from the opensubtitles corpus.
The sentences are chosen based on their similarity to a number of reference sentences.
'''
import argparse
from tqdm import tqdm

from datasets import load_dataset
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

parser = argparse.ArgumentParser()
parser.add_argument('--reference_file', 
                    help='path to reference file',
                    required=True)
parser.add_argument('--output_path_nl', 
                    help='path to store sampled dutch sentences', 
                    default="NMT-Data/Model_English_S_Dutch_S/opensubtitles_nl_testing")
parser.add_argument('--output_path_en', 
                    help='path to store sampled english sentences', 
                    default="NMT-Data/Model_English_S_Dutch_S/opensubtitles_en_testing")
parser.add_argument('--num_samples',
                    type=int,
                    help="number of sentences to extract",
                    default=1000000)
args=parser.parse_args()

extracted_sentences={}
counter = 0

nl_file_path = args.output_path_nl
en_file_path = args.output_path_en

with open(args.reference_file, "r") as f:
    reference = f.readlines()

vectorizer = TfidfVectorizer()

vectorizer.fit(reference)

reference_vectors = vectorizer.transform(reference)

nn = NearestNeighbors(metric='cosine')

dataset = load_dataset("open_subtitles", split='train', lang1="en", lang2="nl", streaming=True)
def add_to_dict(example):
    global counter
    extracted_sentences["sentence{}".format(counter)] = {}
    extracted_sentences["sentence{}".format(counter)]['sentence nl'] = example['translation']['nl']
    extracted_sentences["sentence{}".format(counter)]['sentence en'] = example['translation']['en']
    counter += 1

print('extracting opensubtitles entries...')
small_dataset = dataset.take(20000000)
for example in tqdm(small_dataset):
    add_to_dict(example)

comparison_sentences = []
print('extracting dutch sentences for comparison...')
for key, value in tqdm(extracted_sentences.items()):
    comparison_sentences.append(value['sentence nl'])

print('fitting extracted sentences to vector space...')
comparison_vectors = vectorizer.transform(comparison_sentences)
nn.fit(comparison_vectors)

n = args.num_samples//len(reference)
distances, indices = nn.kneighbors(reference_vectors, n_neighbors=n)
final_output = {}

print(f'finding {n} most similar sentences to each reference...')
for i, sentence in tqdm(enumerate(reference)):
    final_output[sentence.strip()] = []
    for j in indices[i]:
        final_output[sentence.strip()].append((comparison_sentences[j], extracted_sentences[f'sentence{j}']['sentence en']))

print('writing sentences to files...')
with open(nl_file_path, 'w', encoding='utf-8') as f1, open(en_file_path, 'w', encoding='utf-8') as f2:
    for key, value in tqdm(final_output.items()):
        for pair in value:
            nl_sent, en_sent = pair
            f1.write(nl_sent + '\n')
            f2.write(en_sent + '\n')

print(f'Done! extracted dutch sentences written to {nl_file_path}')
print(f'Done! extracted english sentences written to {en_file_path}')