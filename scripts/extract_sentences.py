#This python script is used to retrieve a subset of sentences from the opensubtitles corpus.
#The subset is retrieved on the basis of cosine similarity to a reference corpus (in our case the evaluation test set provided by marloes)
from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer, util
import time

start = time.time()

nl_file_path = "NMT-Data/Model_English_S_Dutch_S/opensubtitles_nl"
en_file_path = "NMT-Data/Model_English_S_Dutch_S/opensubtitles_en"

max_sentences = 500000 #number of aligned sentences in final output
extracted_sentences={}
counter = 0
lowest_cosine_similarity = float(2)  # initialize with a very large value


#function to save the n most similar sentences to a dictionary
def compute_cosine_sim(sentence):
    global counter
    global lowest_cosine_similarity

    # encode sentence
    embedding = model.encode(sentence['translation']['nl'], convert_to_tensor=True)
    
    # Compute the cosine similarity between the Dutch sentence embedding and a list of reference embeddings
    cosine_scores = util.cos_sim(embedding, ref_embeddings)    
    
    #if dict not yet max length
    if counter <= max_sentences:

        # add sentences to dict
        extracted_sentences["sentence{}".format(counter)] = {}
        extracted_sentences["sentence{}".format(counter)]['sentence nl'] = sentence['translation']['nl']
        extracted_sentences["sentence{}".format(counter)]['sentence en'] = sentence['translation']['en']
        extracted_sentences["sentence{}".format(counter)]['cosine similarity'] = cosine_scores.max()

        #print('Added Sentence')
        counter +=1

    #if dict max length
    else:

        lowest_cosine_similarity_id = min(extracted_sentences, key=lambda id: extracted_sentences[id]['cosine similarity'].item())
        lowest_cosine_similarity=min(extracted_sentences.values(), key = lambda x: x['cosine similarity'].item())
        lowest_cosine_similarity_score = lowest_cosine_similarity['cosine similarity']
        print(lowest_cosine_similarity_score)

        if cosine_scores.max() > lowest_cosine_similarity_score:

            
            #update dict items
            extracted_sentences[lowest_cosine_similarity_id]['sentence nl'] = sentence['translation']['nl']
            extracted_sentences[lowest_cosine_similarity_id]['sentence en'] = sentence['translation']['en']
            extracted_sentences[lowest_cosine_similarity_id]['cosine similarity'] = cosine_scores.max()


            #print('Replaced Sentence')
            


        else:
            return
            #print("Discarded Sentence")

#load embedding model
model = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v1')

#Load the reference dataset.
with open("/Users/danielvlantis/Text_Simplification/eval_data/NL_test_org", "r") as f:
    reference = f.readlines()

ref_embeddings = model.encode(reference, convert_to_tensor=True)

#load opensubtitles dataset
dataset = load_dataset("open_subtitles", split='train', lang1="en", lang2="nl", streaming=True)
small_dataset = dataset.take(20000000)

print('Working...')
for example in small_dataset:
    compute_cosine_sim(example)

end = time.time()
print("total time taken in seconds: {}".format(end - start))

#create nl sentences file and en sentences file
with open(nl_file_path, 'w', encoding='utf-8') as f1, open(en_file_path, 'w', encoding='utf-8') as f2:
    for key in extracted_sentences:
        f1.write(extracted_sentences[key]['sentence nl'] + '\n')
        f2.write(extracted_sentences[key]['sentence en'] + '\n')