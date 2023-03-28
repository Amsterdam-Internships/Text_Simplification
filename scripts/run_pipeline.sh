'''
This file executes the full pipeline for pivot based neural machine translation.
This includes downloading and preprocessing data, subwording data, training 3 translation models, and evaluating results on a test set.
'''

pip install openNMT-py

#Download Data
wget https://opus.nlpl.eu/download.php?f=EMEA/v3/moses/en-nl.txt.zip -O NMT-Data/Model_Dutch_C_English_C/emea_en-nl.txt.zip #EMEA download
unzip NMT-Data/Model_Dutch_C_English_C/emea_en-nl.txt.zip -d NMT-Data/Model_Dutch_C_English_C #unzip EMEA

wget -c https://cs.pomona.edu/~dkauchak/simplification/data.v1/data.v1.tar.gz -O NMT-Data/Model_English_C_English_S/WikiSimple.tar.gz #WikiSimple Download
tar -xvf NMT-Data/Model_English_C_English_S/WikiSimple.tar.gz -C NMT-Data/Model_English_C_English_S #WikiSimple unzip

python3 scripts/extract_sentences.py  #OpenSubtiltes Download (change params in file to adjust number of aligned sentences and sample size)

#Preprocess data
python3 notebooks/data_processing/filter.py NMT-Data/Model_Dutch_C_English_C/EMEA.en-nl.nl NMT-Data/Model_Dutch_C_English_C/EMEA.en-nl.en nl en #filter EMEA
python3 notebooks/data_processing/filter.py NMT-Data/Model_English_C_English_S/data.v1/wiki.unsimplified NMT-Data/Model_English_C_English_S/data.v1/wiki.simple en_c en_s #filter WikiSimple
python3 notebooks/data_processing/filter.py NMT-Data/Model_English_S_Dutch_S/opensubtitles_en NMT-Data/Model_English_S_Dutch_S/opensubtitles_nl en nl #filter opensubtitles


#build subword models
python3 notebooks/data_processing/1_train_bpe.py NMT-Data/Model_Dutch_C_English_C/EMEA.en-nl.nl-filtered.nl NMT-Data/Model_Dutch_C_English_C/subword_model/yttm_source.model #train bpe for EMEA source
python3 notebooks/data_processing/1_train_bpe.py NMT-Data/Model_Dutch_C_English_C/EMEA.en-nl.en-filtered.en NMT-Data/Model_Dutch_C_English_C/subword_model/yttm_target.model #train bpe for EMEA target

python3 notebooks/data_processing/1_train_bpe.py NMT-Data/Model_English_C_English_S/data.v1/wiki.unsimplified-filtered.en_c NMT-Data/Model_English_C_English_S/subword_model/yttm_source.model #train bpe for WikiSimple source
python3 notebooks/data_processing/1_train_bpe.py NMT-Data/Model_English_C_English_S/data.v1/wiki.simple-filtered.en_s NMT-Data/Model_English_C_English_S/subword_model/yttm_target.model #train bpe for WikiSimple target

python3 notebooks/data_processing/1_train_bpe.py NMT-Data/Model_English_S_Dutch_S/opensubtitles_en-filtered.en NMT-Data/Model_English_S_Dutch_S/subword_model/yttm_source.model #train bpe for OpenSubtitles source
python3 notebooks/data_processing/1_train_bpe.py NMT-Data/Model_English_S_Dutch_S/opensubtitles_nl-filtered.nl NMT-Data/Model_English_S_Dutch_S/subword_model/yttm_target.model #train bpe for OpenSubtitles target

#subword data
python3 notebooks/data_processing/2_subword.py NMT-Data/Model_Dutch_C_English_C/subword_model/yttm_source.model NMT-Data/Model_Dutch_C_English_C/EMEA.en-nl.nl-filtered.nl NMT-Data/Model_Dutch_C_English_C/EMEA.en-nl.nl-filtered.nl-subword.nl #subword EMEA source
python3 notebooks/data_processing/2_subword.py NMT-Data/Model_Dutch_C_English_C/subword_model/yttm_target.model NMT-Data/Model_Dutch_C_English_C/EMEA.en-nl.en-filtered.en NMT-Data/Model_Dutch_C_English_C/EMEA.en-nl.en-filtered.en-subword.en #subword EMEA target

python3 notebooks/data_processing/2_subword.py NMT-Data/Model_English_C_English_S/subword_model/yttm_source.model NMT-Data/Model_English_C_English_S/data.v1/wiki.unsimplified-filtered.en_c NMT-Data/Model_English_C_English_S/data.v1/wiki.unsimplified-filtered.en_c-subword.en_c #subword WikiSimple
python3 notebooks/data_processing/2_subword.py NMT-Data/Model_English_C_English_S/subword_model/yttm_target.model NMT-Data/Model_English_C_English_S/data.v1/wiki.simple-filtered.en_s NMT-Data/Model_English_C_English_S/data.v1/wiki.simple-filtered.en_s-subword.en_s #subword WikiSimple

python3 notebooks/data_processing/2_subword.py NMT-Data/Model_English_S_Dutch_S/subword_model/yttm_source.model NMT-Data/Model_English_S_Dutch_S/opensubtitles_en-filtered.en NMT-Data/Model_English_S_Dutch_S/opensubtitles_en-filtered.en-subword.en
python3 notebooks/data_processing/2_subword.py NMT-Data/Model_English_S_Dutch_S/subword_model/yttm_target.model NMT-Data/Model_English_S_Dutch_S/opensubtitles_nl-filtered.nl NMT-Data/Model_English_S_Dutch_S/opensubtitles_nl-filtered.nl-subword.nl

#Split data into train/test/dev
python3 notebooks/data_processing/train_dev_test_split.py 2000 2000 NMT-Data/Model_Dutch_C_English_C/EMEA.en-nl.nl-filtered.nl-subword.nl NMT-Data/Model_Dutch_C_English_C/EMEA.en-nl.en-filtered.en-subword.en #split EMEA data

python3 notebooks/data_processing/train_dev_test_split.py 2000 2000 NMT-Data/Model_English_C_English_S/data.v1/wiki.unsimplified-filtered.en_c-subword.en_c NMT-Data/Model_English_C_English_S/data.v1/wiki.simple-filtered.en_s-subword.en_s #split WikiSimple data

python3 notebooks/data_processing/train_dev_test_split.py 2000 2000 NMT-Data/Model_English_S_Dutch_S/opensubtitles_en-filtered.en-subword.en NMT-Data/Model_English_S_Dutch_S/opensubtitles_nl-filtered.nl-subword.nl #split OpenSubtitles data

#Build Vocab
#EMEA
touch NMT-Data/Model_Dutch_C_English_C/run/vocab.src; #create empty files where vocab will be stored
touch NMT-Data/Model_Dutch_C_English_C/run/vocab.tgt;
onmt_build_vocab -config model_1.yaml -overwrite True #Build vocab


#WikiSimple
touch NMT-Data/Model_English_C_English_S/run/vocab.src; #create empty files where vocab will be stored
touch NMT-Data/Model_English_C_English_S/run/vocab.tgt;
onmt_build_vocab -config model_2.yaml -overwrite True #Build vocab

#OpenSubtitles
touch NMT-Data/Model_English_S_Dutch_S/run/vocab.src; #create empty files where vocab will be stored
touch NMT-Data/Model_English_S_Dutch_S/run/vocab.tgt;
onmt_build_vocab -config model_3.yaml -overwrite True #build vocab

'''
#Train Models
onmt_train -config model_1.yaml -src_vocab NMT-Data/Model_Dutch_C_English_C/run/.vocab.src -tgt_vocab NMT-Data/Model_Dutch_C_English_C/run/.vocab.tgt #Train Emea model
#onmt_train -config model_2.yaml -src_vocab NMT-Data/Model_English_C_English_S/run/.vocab.src -tgt_vocab NMT-Data/Model_English_C_English_S/run/.vocab.tgt #Train WikiSimple model
#omnt_train -config model_3.yaml -src_vocab NMT-Data/Model_English_S_Dutch_S/run/.vocab.src -tgt_vocab NMT-Data/Model_English_S_Dutch_S/run/.vocab.tgt #train OpenSubtitles Model

#Run input data through pipeline
#model 1
python3 notebooks/data_processing/2_subword.py NMT-Data/Model_Dutch_C_English_C/subword_model/yttm_source.model eval_data/NL_test_org eval_data/NL_test_org_subword_1
onmt_translate -ban_unk_token -model NMT-Data/Model_Dutch_C_English_C/model/yttm_model_1000.pt -src eval_data/NL_test_org_subword_1 -output NMT-Data/Model_Dutch_C_English_C/model_output/Model_1_pred.txt -verbose 
python3 notebooks/data_processing/3_desubword.py NMT-Data/Model_Dutch_C_English_C/subword_model/yttm_target.model NMT-Data/Model_Dutch_C_English_C/model_output/Model_1_pred.txt NMT-Data/Model_Dutch_C_English_C/model_output/Model_1_pred.txt.desubword

#model 2
python3 notebooks/data_processing/2_subword.py NMT-Data/Model_English_C_English_S/subword_model/yttm_source.model NMT-Data/Model_Dutch_C_English_C/model_output/Model_1_pred.txt.desubword eval_data/NL_test_org_subword_2
onmt_translate -ban_unk_token -model NMT-Data/Model_English_C_English_S/model/yttm_model_1000.pt -src eval_data/NL_test_org_subword_2 -output NMT-Data/Model_English_C_English_S/model_output/Model_2_pred.txt -verbose
python3 notebooks/data_processing/3_desubword.py NMT-Data/Model_English_C_English_S/subword_model/yttm_target.model NMT-Data/Model_English_C_English_S/model_output/Model_2_pred.txt NMT-Data/Model_English_C_English_S/model_output/Model_2_pred.txt.desubword

#model 3
python3 notebooks/data_processing/2_subword.py NMT-Data/Model_English_S_Dutch_S/subword_model/yttm_source.model NMT-Data/Model_English_C_English_S/model_output/Model_2_pred.txt.desubword eval_data/NL_test_org_subword_3
onmt_translate -ban_unk_token -model NMT-Data/Model_English_S_Dutch_S/model/mybasemodel_step_9000.pt -src eval_data/NL_test_org_subword_3 -output NMT-Data/Model_English_S_Dutch_S/model_output/Model_3_pred.txt -verbose
python3 notebooks/data_processing/3_desubword.py NMT-Data/Model_English_S_Dutch_S/subword_model/yttm_target.model NMT-Data/Model_English_S_Dutch_S/model_output/Model_3_pred.txt NMT-Data/Model_English_S_Dutch_S/model_output/Model_3_pred.txt.desubword

#Evaluate Results
python3 notebooks/evaluate_script.py eval_data/NL_test_org eval_data/NL_test_simp NMT-Data/Model_English_S_Dutch_S/model_output/Model_3_pred.txt.desubword
'''