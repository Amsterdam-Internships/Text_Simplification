# This script is used to calculate automatic evaluation metrics for translated sentences.
# Run the script and provide the following arguments: evaluate_script.py source_path reference_path predictions_path


#Compute Bleu Sari and Meteor Scores for Validation set
import sys
import evaluate
import os


source_path = sys.argv[1]
reference_path = sys.argv[2]
predictions_path = sys.argv[3]

print(source_path, reference_path, predictions_path)

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