'''
This file executes the full pipeline for pivot based neural machine translation.
This includes downloading and preprocessing data, subwording data, training 3 translation models, and evaluating results on a test set.
'''

pip install openNMT-py

data_folder=NMT-Data
emea_folder=$data_folder/Model_Dutch_C_English_C
wikisimple_folder = $data_folder/Model_English_C_English_S
opensubtitles_folder = $data_folder/Model_English_S_Dutch_S

#Download Data
wget https://opus.nlpl.eu/download.php?f=EMEA/v3/moses/en-nl.txt.zip -O $emea_folder/emea_en-nl.txt.zip #EMEA download
unzip $emea_folder/emea_en-nl.txt.zip -d $emea_folder #unzip EMEA

sudo wget -c https://cs.pomona.edu/~dkauchak/simplification/data.v1/data.v1.tar.gz -O $Wikisimple_folder/WikiSimple.tar.gz #WikiSimple Download
tar -xvf NMT-Data/Model_English_C_English_S/WikiSimple.tar.gz -C NMT-Data/Model_English_C_English_S #WikiSimple unzip

# if NMT-Data/Model_English_S_Dutch_S/opensubtitles_en_testing doesnt exist:
if [[ ! -f NMT-Data/Model_English_S_Dutch_S/opensubtitles_en_testing ]] #OpenSubtiltes Download params: (--reference file, --output_path_nl, --output_path_en, --num_samples --)
then
    python scripts/extract_sentences.py --reference_file NMT-Data/Eval_Medical_Dutch_C_Dutch_S/NL_test_org
else 
    echo "OpenSubtitles Medical Subset already exists"
fi

#Preprocess data
python3 notebooks/data_processing/filter.py $emea_folder/EMEA.en-nl.nl $emea_folder/EMEA.en-nl.en nl en #filter EMEA
python3 notebooks/data_processing/filter.py NMT-Data/Model_English_C_English_S/data.v1/wiki.unsimplified NMT-Data/Model_English_C_English_S/data.v1/wiki.simple en_c en_s #filter WikiSimple
python3 notebooks/data_processing/filter.py NMT-Data/Model_English_S_Dutch_S/opensubtitles_en_testing NMT-Data/Model_English_S_Dutch_S/opensubtitles_nl_testing en nl #filter opensubtitles

#build subword models
python3 notebooks/data_processing/1_train_bpe.py $emea_folder/EMEA.en-nl.nl-filtered.nl $emea_folder/subword_model/yttm_source.model #train bpe for EMEA source
python3 notebooks/data_processing/1_train_bpe.py $emea_folder/EMEA.en-nl.en-filtered.en $emea_folder/subword_model/yttm_target.model #train bpe for EMEA target

python3 notebooks/data_processing/1_train_bpe.py NMT-Data/Model_English_C_English_S/data.v1/wiki.unsimplified-filtered.en_c NMT-Data/Model_English_C_English_S/subword_model/yttm_source.model #train bpe for WikiSimple source
python3 notebooks/data_processing/1_train_bpe.py NMT-Data/Model_English_C_English_S/data.v1/wiki.simple-filtered.en_s NMT-Data/Model_English_C_English_S/subword_model/yttm_target.model #train bpe for WikiSimple target

python3 notebooks/data_processing/1_train_bpe.py NMT-Data/Model_English_S_Dutch_S/opensubtitles_en_testing-filtered.en NMT-Data/Model_English_S_Dutch_S/subword_model/yttm_source.model #train bpe for OpenSubtitles source
python3 notebooks/data_processing/1_train_bpe.py NMT-Data/Model_English_S_Dutch_S/opensubtitles_nl_testing-filtered.nl NMT-Data/Model_English_S_Dutch_S/subword_model/yttm_target.model #train bpe for OpenSubtitles target


#subword data
python3 notebooks/data_processing/2_subword.py $emea_folder/subword_model/yttm_source.model $emea_folder/EMEA.en-nl.nl-filtered.nl $emea_folder/EMEA.en-nl.nl-filtered.nl-subword.nl #subword EMEA source
python3 notebooks/data_processing/2_subword.py $emea_folder/subword_model/yttm_target.model $emea_folder/EMEA.en-nl.en-filtered.en $emea_folder/EMEA.en-nl.en-filtered.en-subword.en #subword EMEA target

python3 notebooks/data_processing/2_subword.py NMT-Data/Model_English_S_Dutch_S/subword_model/yttm_source.model NMT-Data/Model_English_C_English_S/data.v1/wiki.unsimplified-filtered.en_c NMT-Data/Model_English_C_English_S/data.v1/wiki.unsimplified-filtered.en_c-subword.en_c #subword WikiSimple
python3 notebooks/data_processing/2_subword.py NMT-Data/Model_English_S_Dutch_S/subword_model/yttm_target.model NMT-Data/Model_English_C_English_S/data.v1/wiki.simple-filtered.en_s NMT-Data/Model_English_C_English_S/data.v1/wiki.simple-filtered.en_s-subword.en_s #subword WikiSimple

python3 notebooks/data_processing/2_subword.py NMT-Data/Model_English_S_Dutch_S/subword_model/yttm_source.model NMT-Data/Model_English_S_Dutch_S/opensubtitles_en_testing-filtered.en NMT-Data/Model_English_S_Dutch_S/opensubtitles_en-filtered.en-subword.en #subword OpenSubtitles source
python3 notebooks/data_processing/2_subword.py NMT-Data/Model_English_S_Dutch_S/subword_model/yttm_target.model NMT-Data/Model_English_S_Dutch_S/opensubtitles_nl_testing-filtered.nl NMT-Data/Model_English_S_Dutch_S/opensubtitles_nl-filtered.nl-subword.nl #subword Opensubtitles target

#Split data into train/test/dev
python3 notebooks/data_processing/train_dev_test_split.py 2000 2000 $emea_folder/EMEA.en-nl.nl-filtered.nl-subword.nl $emea_folder/EMEA.en-nl.en-filtered.en-subword.en #split EMEA data

python3 notebooks/data_processing/train_dev_test_split.py 2000 2000 NMT-Data/Model_English_C_English_S/data.v1/wiki.unsimplified-filtered.en_c-subword.en_c NMT-Data/Model_English_C_English_S/data.v1/wiki.simple-filtered.en_s-subword.en_s #split WikiSimple data

python3 notebooks/data_processing/train_dev_test_split.py 2000 2000 NMT-Data/Model_English_S_Dutch_S/opensubtitles_en-filtered.en-subword.en NMT-Data/Model_English_S_Dutch_S/opensubtitles_nl-filtered.nl-subword.nl #split OpenSubtitles data

#Build Vocab
#EMEA
touch $emea_folder/run/vocab.src; #create empty files where vocab will be stored
touch $emea_folder/run/vocab.tgt;
onmt_build_vocab -config config/model_1.yaml -overwrite True #Build vocab


#WikiSimple
touch NMT-Data/Model_English_C_English_S/run/vocab.src; #create empty files where vocab will be stored
touch NMT-Data/Model_English_C_English_S/run/vocab.tgt;
onmt_build_vocab -config config/model_2.yaml -overwrite True #Build vocab

#OpenSubtitles
touch NMT-Data/Model_English_S_Dutch_S/run/vocab.src; #create empty files where vocab will be stored
touch NMT-Data/Model_English_S_Dutch_S/run/vocab.tgt;
onmt_build_vocab -config config/model_3.yaml -overwrite True #build vocab

'''
#Train Models
onmt_train -config config/model_1.yaml -src_vocab $emea_folder/run/.vocab.src -tgt_vocab $emea_folder/run/.vocab.tgt #Train Emea model
onmt_train -config config/model_2.yaml -src_vocab NMT-Data/Model_English_C_English_S/run/.vocab.src -tgt_vocab NMT-Data/Model_English_C_English_S/run/.vocab.tgt #Train WikiSimple model
omnt_train -config config/model_3.yaml -src_vocab NMT-Data/Model_English_S_Dutch_S/run/.vocab.src -tgt_vocab NMT-Data/Model_English_S_Dutch_S/run/.vocab.tgt #train OpenSubtitles Model

#Run input data through pipeline
#model 1
python3 notebooks/data_processing/2_subword.py $emea_folder/subword_model/yttm_source.model eval_data/NL_test_org eval_data/NL_test_org_subword_1
onmt_translate -ban_unk_token -model $emea_folder/model/yttm_model_1000.pt -src eval_data/NL_test_org_subword_1 -output $emea_folder/model_output/Model_1_pred.txt -verbose 
python3 notebooks/data_processing/3_desubword.py $emea_folder/subword_model/yttm_target.model $emea_folder/model_output/Model_1_pred.txt $emea_folder/model_output/Model_1_pred.txt.desubword

#model 2
python3 notebooks/data_processing/2_subword.py NMT-Data/Model_English_C_English_S/subword_model/yttm_source.model $emea_folder/model_output/Model_1_pred.txt.desubword eval_data/NL_test_org_subword_2
onmt_translate -ban_unk_token -model NMT-Data/Model_English_C_English_S/model/yttm_model_1000.pt -src eval_data/NL_test_org_subword_2 -output NMT-Data/Model_English_C_English_S/model_output/Model_2_pred.txt -verbose
python3 notebooks/data_processing/3_desubword.py NMT-Data/Model_English_C_English_S/subword_model/yttm_target.model NMT-Data/Model_English_C_English_S/model_output/Model_2_pred.txt NMT-Data/Model_English_C_English_S/model_output/Model_2_pred.txt.desubword

#model 3
python3 notebooks/data_processing/2_subword.py NMT-Data/Model_English_S_Dutch_S/subword_model/yttm_source.model NMT-Data/Model_English_C_English_S/model_output/Model_2_pred.txt.desubword eval_data/NL_test_org_subword_3
onmt_translate -ban_unk_token -model NMT-Data/Model_English_S_Dutch_S/model/mybasemodel_step_9000.pt -src eval_data/NL_test_org_subword_3 -output NMT-Data/Model_English_S_Dutch_S/model_output/Model_3_pred.txt -verbose
python3 notebooks/data_processing/3_desubword.py NMT-Data/Model_English_S_Dutch_S/subword_model/yttm_target.model NMT-Data/Model_English_S_Dutch_S/model_output/Model_3_pred.txt NMT-Data/Model_English_S_Dutch_S/model_output/Model_3_pred.txt.desubword

#Evaluate Results
python3 notebooks/evaluate_script.py NMT-Data/Eval_Medical_Dutch_C_Dutch_S/NL_test_org NMT-Data/Eval_Medical_Dutch_C_Dutch_S/NL_test_simp NMT-Data/Model_English_S_Dutch_S/model_output/Model_3_pred.txt.desubword
'''