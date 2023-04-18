'''
This script is used to calculate automatic evaluation metrics for translated sentences.
parameters: --source_path --reference_path --target_path
'''

#Compute Bleu Sari and Meteor Scores for Validation set
import argparse
import sys
import evaluate
import os

parser = argparse.ArgumentParser()
parser.add_argument('--source_path', 
                    help='path to source sentences file',
                    required=True)
parser.add_argument('--target_path', 
                    help='path to predicted sentences file', 
                    required=True)
parser.add_argument('--reference_path',
                    help='path to reference sentences file',
                    required=True)
args=parser.parse_args()

source_path = args.source_path
reference_path = args.reference_path
predictions_path = args.target_path

#load source sentences
sources = []
#path to source sentences file
with open(source_path,'r') as f:
    for line in f:
        sources.append(line.strip())

#load reference sentences
references = []
#path to reference sentences file
with open(reference_path,'r') as f:
    for line in f:
        references.append([line.strip()])

#load predicted sentences
predictions = []
#path to predicted sentences file
with open(predictions_path,'r') as f:
    for line in f:
        predictions.append(line.strip())


#load metrics
sari = evaluate.load("sari")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

sari_score = sari.compute(sources=sources,predictions=predictions,references=references)
bleu_score = bleu.compute(predictions=predictions,references=references)
meteor_score = meteor.compute(predictions=predictions, references=references)

#print scores
term_size = os.get_terminal_size()
print('=' * term_size.columns)
print(sari_score)
print(bleu_score)
print(meteor_score)
print('=' * term_size.columns)