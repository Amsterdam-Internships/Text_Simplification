'''
This script extracts sentences from the opensubtitles corpus.
The sentences are chosen based on their similarity to a number of reference sentences.
'''
import argparse
import json
import os
from tqdm import tqdm
from datasets import load_dataset
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()
parser.add_argument('--reference_file',
                    help='path to reference file',
                    required=True)
parser.add_argument('--output_path_nl',
                    help='path to store sampled dutch sentences',
                    default="NMT-Data/Model_English_S_Dutch_S/medsubset_bert.nl")
parser.add_argument('--output_path_en',
                    help='path to store sampled english sentences',
                    default="NMT-Data/Model_English_S_Dutch_S/medsubset_bert.en")
parser.add_argument('--num_samples',
                    type=int,
                    help="number of sentences to extract",
                    default=1000000)
parser.add_argument('--encoding_method',
                    choices=['tfidf', 'sentence_transformer'],
                    help="encoding method to use",
                    default='tfidf')
args = parser.parse_args()

extracted_sentences = {}
counter = 0

nl_file_path = args.output_path_nl
en_file_path = args.output_path_en

with open(args.reference_file, "r") as f:
    reference = f.readlines()

if args.encoding_method == 'tfidf':
    vectorizer = TfidfVectorizer()
    vectorizer.fit(reference)
    reference_vectors = vectorizer.transform(reference)
else:
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    reference_vectors = model.encode(reference)

nn = NearestNeighbors(metric='cosine')

dataset = load_dataset("open_subtitles", split='train', lang1="en", lang2="nl", streaming=True)


def add_to_dict(sent):
    '''
    Extract english and dutch sentence pairs and store them in our dictionary
    Args:
        sent {iterable dataset item}
    '''
    global counter
    extracted_sentences["sentence{}".format(counter)] = {}
    extracted_sentences["sentence{}".format(counter)]['sentence nl'] = sent['translation']['nl']
    extracted_sentences["sentence{}".format(counter)]['sentence en'] = sent['translation']['en']
    counter += 1


base_folder = os.path.dirname(en_file_path)

num_samples_general = 2000000
samples_file_general = f"{base_folder}/opensubtitles_samples_{num_samples_general}"
try:
    extracted_sentences = json.load(open(samples_file_general, 'r'))
except:
    print('extracting opensubtitles entries...')
    small_dataset = dataset.take(num_samples_general)
    for example in tqdm(small_dataset):
        add_to_dict(example)
    with open(samples_file_general, 'w') as f:
        json.dump(extracted_sentences, f)

comparison_sentences = []
print('extracting dutch sentences for comparison...')
for key, value in tqdm(extracted_sentences.items()):
    comparison_sentences.append(value['sentence nl'])

if args.encoding_method == 'tfidf':
    comparison_vectors = vectorizer.transform(comparison_sentences)
else:
    comparison_vectors = model.encode(comparison_sentences)

print('fitting extracted sentences to vector space')
nn.fit(comparison_vectors)

n = args.num_samples // len(reference)
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

print(f'Done! Extracted Dutch sentences written to {nl_file_path}')
print(f'Done! Extracted English sentences written to {en_file_path}')