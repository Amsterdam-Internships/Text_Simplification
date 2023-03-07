#This python script is used to retrieve a subset of sentences from the opensubtitles corpus.
#The subset is retrieved on the basis of cosine similarity to a reference corpus (in our case the evaluation test set provided by marloes)

from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer, util
import sys

#nl_file_path = sys.argv(1)
#en_file_path = sys.argv(2)



max_sentences = 10
extracted_sentences={}
counter = 0
lowest_cosine_similarity = float('inf')  # initialize with a very large value




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


        print(extracted_sentences)
        counter +=1

    #if dict max length
    else:

        lowest_cosine_similarity_id = min(extracted_sentences, key=lambda id: extracted_sentences[id]['cosine similarity'].item())
        print('min Lowest cosine similarity id' + lowest_cosine_similarity_id)

        if cosine_scores.max() > lowest_cosine_similarity:

            
            #update dict items
            print(sentence)
            print(extracted_sentences[dict_key])
            extracted_sentences[dict_key]['sentence nl'] = sentence['translation']['nl']
            extracted_sentences[dict_key]['sentence en'] = sentence['translation']['en']
            extracted_sentences[dict_key]['cosine similarity'] = cosine_scores.max()


            print('replaced')
            print(extracted_sentences)
            


        else:
            print('nay')



#load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

#Load the reference dataset.
with open("eval_data/NL_test_org", "r") as f:
    reference = f.readlines()

ref_embeddings = model.encode(reference, convert_to_tensor=True)

#load opensubtitles dataset
dataset = load_dataset("open_subtitles", split='train', lang1="en", lang2="nl", streaming=True)

for example in dataset:
    compute_cosine_sim(example)







'''
#create nl sentences file and en sentences file
with open(nl_file_path, 'w', encoding='utf-8') as f1, open(en_file_path, 'w', encoding='utf-8') as f2:
    for item in extracted_sentences:
        f1.write(item['sentence nl'] + '\n')
        f2.write(item['sentence_en'] + '\n')
'''