'''
This script takes extracts sentences from the opensubtitles corpus.
The sentences are chosen based on their similarity to a number of reference sentences.
'''

from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
from tqdm import tqdm

extracted_sentences={}
counter = 0

nl_file_path = "NMT-Data/Model_English_S_Dutch_S/opensubtitles_nl_testing"
en_file_path = "NMT-Data/Model_English_S_Dutch_S/opensubtitles_en_testing"

with open("NMT-Data/eval_Medical_Dutch_C_Dutch_S/NL_test_org", "r") as f:
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

n = 1000000//len(reference)
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