#pip install --upgrade OpenNMT-py==2.0.0rc1
pip install OpenNMT-py
pip install sacremoses

data_folder=NMT-Data
emea_folder=$data_folder/Model_Dutch_C_English_C
wikisimple_folder=$data_folder/Model_English_C_English_S
opensubtitles_folder=$data_folder/Model_English_S_Dutch_S
europarl_folder=$data_folder/europarl
experiment_prefix=mun_model
steps=`cat ${experiment_prefix}_3.yaml | grep train_steps | cut -f2 -d ' '`

#Download Data
wget https://opus.nlpl.eu/download.php?f=Europarl/v8/moses/en-nl.txt.zip -O $europarl_folder/en-nl.txt.zip #EMEA download
unzip $europarl_folder/en-nl.txt.zip -d $europarl_folder #unzip EMEA

sudo wget -c https://cs.pomona.edu/~dkauchak/simplification/data.v1/data.v1.tar.gz -O $wikisimple_folder/WikiSimple.tar.gz #WikiSimple Download
tar -xvf $wikisimple_folder/WikiSimple.tar.gz -C $wikisimple_folder #WikiSimple unzip
#sudo unzip $wikisimple_folder/WikiSimple.tar.gz -d $wikisimple_folder


# if NMT-Data/Model_English_S_Dutch_S/opensubtitles_en_testing doesnt exist:
if [[ ! -f NMT-Data/Model_English_S_Dutch_S/munsubset_bert.en ]] #OpenSubtiltes Download params: (--reference file, --output_path_nl, --output_path_en, --num_samples --)
then
    python src/extract_sentences.py --reference_file NMT-Data/reference_mun.txt --output_path_en $opensubtitles_folder/munsubset_bert.en --output_path_nl $opensubtitles_folder/munsubset_bert.nl --encoding_method sentence_transformer
else 
    echo "OpenSubtitles Municipal Subset (bert) already exists"
fi

#Preprocess data
python3 src/data_processing/filter.py $europarl_folder/Europarl.en-nl.nl $europarl_folder/Europarl.en-nl.en nl en #filter EMEA
python3 src/data_processing/filter.py $wikisimple_folder/data.v1/wiki.unsimplified $wikisimple_folder/data.v1/wiki.simple en_c en_s #filter WikiSimple
python3 src/data_processing/filter.py $opensubtitles_folder/munsubset_bert.en $opensubtitles_folder/munsubset_bert.nl en nl #filter opensubtitles

#build subword models
python3 src/data_processing/1_train_bpe.py --train_data_path $europarl_folder/Europarl.en-nl.nl-filtered.nl --model_path $europarl_folder/subword_model/yttm_source.model #train bpe for EMEA source
python3 src/data_processing/1_train_bpe.py --train_data_path $europarl_folder/Europarl.en-nl.en-filtered.en --model_path $europarl_folder/subword_model/yttm_target.model #train bpe for EMEA target

python3 src/data_processing/1_train_bpe.py --train_data_path $wikisimple_folder/data.v1/wiki.unsimplified-filtered.en_c --model_path NMT-Data/Model_English_C_English_S/subword_model/yttm_source.model #train bpe for WikiSimple source
python3 src/data_processing/1_train_bpe.py --train_data_path $wikisimple_folder/data.v1/wiki.simple-filtered.en_s --model_path NMT-Data/Model_English_C_English_S/subword_model/yttm_target.model #train bpe for WikiSimple target

python3 src/data_processing/1_train_bpe.py --train_data_path $opensubtitles_folder/munsubset_bert.en-filtered.en --model_path NMT-Data/Model_English_S_Dutch_S/subword_model/mun_bert_source.model #train bpe for OpenSubtitles source
python3 src/data_processing/1_train_bpe.py --train_data_path $opensubtitles_folder/munsubset_bert.nl-filtered.nl --model_path NMT-Data/Model_English_S_Dutch_S/subword_model/mun_bert_target.model #train bpe for OpenSubtitles target


#subword data
python3 src/data_processing/2_subword.py --model_path $europarl_folder/subword_model/yttm_source.model --input_path $europarl_folder/Europarl.en-nl.nl-filtered.nl --output_path $europarl_folder/Europarl.en-nl.nl-filtered.nl-subword.nl #subword EMEA source
python3 src/data_processing/2_subword.py --model_path $europarl_folder/subword_model/yttm_target.model --input_path $europarl_folder/Europarl.en-nl.en-filtered.en --output_path $europarl_folder/Europarl.en-nl.en-filtered.en-subword.en #subword EMEA target

python3 src/data_processing/2_subword.py --model_path $wikisimple_folder/subword_model/yttm_source.model --input_path NMT-Data/Model_English_C_English_S/data.v1/wiki.unsimplified-filtered.en_c --output_path NMT-Data/Model_English_C_English_S/data.v1/wiki.unsimplified-filtered.en_c-subword.en_c #subword WikiSimple
python3 src/data_processing/2_subword.py --model_path $wikisimple_folder/subword_model/yttm_target.model --input_path NMT-Data/Model_English_C_English_S/data.v1/wiki.simple-filtered.en_s --output_path NMT-Data/Model_English_C_English_S/data.v1/wiki.simple-filtered.en_s-subword.en_s #subword WikiSimple

python3 src/data_processing/2_subword.py --model_path $opensubtitles_folder/subword_model/mun_bert_source.model --input_path NMT-Data/Model_English_S_Dutch_S/munsubset_bert.en-filtered.en --output_path NMT-Data/Model_English_S_Dutch_S/munsubset_bert.en-filtered.en-subword.en #subword OpenSubtitles source
python3 src/data_processing/2_subword.py --model_path $opensubtitles_folder/subword_model/mun_bert_target.model --input_path NMT-Data/Model_English_S_Dutch_S/munsubset_bert.nl-filtered.nl --output_path NMT-Data/Model_English_S_Dutch_S/munsubset_bert.nl-filtered.nl-subword.nl #subword Opensubtitles target

#Split data into train/test/dev
python3 src/data_processing/train_dev_test_split.py 2000 2000 $europarl_folder/Europarl.en-nl.nl-filtered.nl-subword.nl $europarl_folder/Europarl.en-nl.en-filtered.en-subword.en #split EMEA data

python3 src/data_processing/train_dev_test_split.py 2000 2000 $wikisimple_folder/data.v1/wiki.unsimplified-filtered.en_c-subword.en_c NMT-Data/Model_English_C_English_S/data.v1/wiki.simple-filtered.en_s-subword.en_s #split WikiSimple data

python3 src/data_processing/train_dev_test_split.py 2000 2000 NMT-Data/Model_English_S_Dutch_S/munsubset_bert.nl-filtered.nl-subword.nl NMT-Data/Model_English_S_Dutch_S/munsubset_bert.en-filtered.en-subword.en #split OpenSubtitles data

#Build Vocab
#EMEA
touch $europarl_folder/run/mun_vocab.src; #create empty files where vocab will be stored
touch $europarl_folder/run/mun_vocab.tgt;
onmt_build_vocab -config config/model_nl-en_europarl.yaml -overwrite True #Build vocab

#WikiSimple
touch $wikisimple_folder/run/vocab.src; #create empty files where vocab will be stored
touch $wikisimple_folder/run/vocab.tgt;
onmt_build_vocab -config config/model_en_c-s.yaml -overwrite True #Build vocab

#OpenSubtitles
touch $opensubtitles_folder/run/mun_bert_vocab.src; #create empty files where vocab will be stored
touch $opensubtitles_folder/run/mun_bert_vocab.tgt;
onmt_build_vocab -config config/model_en-nl_munsubset_bert.yaml -overwrite True #build vocab

#Train Models
#onmt_train -config config/model_nl-en_europarl.yaml #-src_vocab $emea_folder/run/.vocab.src -tgt_vocab $emea_folder/run/.vocab.tgt #Train Emea model
#onmt_train -config config/model_en_c-s.yaml #-src_vocab $wikisimple_folder/run/.vocab.src -tgt_vocab $wikisimple_folder/run/.vocab.tgt #Train WikiSimple model
onmt_train -config config/model_en-nl_munsubset_bert.yaml #-src_vocab NMT-Data/Model_English_S_Dutch_S/run/.vocab.src -tgt_vocab NMT-Data/Model_English_S_Dutch_S/run/.vocab.tgt #train OpenSubtitles Model

#Run input data through pipeline
#model 1
python3 src/data_processing/2_subword.py --model_path $europarl_folder/subword_model/yttm_source.model --input_path NMT-Data/Eval_Municipal/complex --output_path NMT-Data/Eval_Municipal/mun_complex_subword_1
onmt_translate -ban_unk_token -model $europarl_folder/model/mymunbasemodel_step_10000.pt -src NMT-Data/Eval_Municipal/mun_complex_subword_1 -output $europarl_folder/model_output/mun_Model_1_pred.txt -verbose 
python3 src/data_processing/3_desubword.py --model_path $europarl_folder/subword_model/yttm_target.model --input_path $europarl_folder/model_output/mun_Model_1_pred.txt --output_path $europarl_folder/model_output/mun_Model_1_pred.txt.desubword

#model 2
python3 src/data_processing/2_subword.py --model_path $wikisimple_folder/subword_model/yttm_source.model --input_path $europarl_folder/model_output/mun_Model_1_pred.txt.desubword --output_path NMT-Data/Eval_Medical_Dutch_C_Dutch_S/NL_test_org_mun_subword_2
onmt_translate -ban_unk_token -model $wikisimple_folder/model/mybasemodel_step_10000.pt -src NMT-Data/Eval_Medical_Dutch_C_Dutch_S/NL_test_org_mun_subword_2 -output $wikisimple_folder/model_output/mun_Model_2_pred.txt -verbose
python3 src/data_processing/3_desubword.py --model_path $wikisimple_folder/subword_model/yttm_target.model --input_path $wikisimple_folder/model_output/mun_Model_2_pred.txt --output_path $wikisimple_folder/model_output/mun_Model_2_pred.txt.desubword

#model 3
python3 src/data_processing/2_subword.py --model_path $opensubtitles_folder/subword_model/mun_bert_source.model --input_path $wikisimple_folder/model_output/mun_Model_2_pred.txt.desubword --output_path NMT-Data/Eval_Municipal/mun_NL_test_org_subword_3
onmt_translate -ban_unk_token -model $opensubtitles_folder/model/mymunbasemodel_bert_step_10000.pt -src NMT-Data/Eval_Municipal/mun_NL_test_org_subword_3 -output $opensubtitles_folder/model_output/mun_Model_3_pred.txt -verbose
python3 src/data_processing/3_desubword.py --model_path $opensubtitles_folder/subword_model/mun_bert_target.model --input_path $opensubtitles_folder/model_output/mun_Model_3_pred.txt --output_path $opensubtitles_folder/model_output/mun_Model_3_pred.txt.desubword

#Evaluate Results
python3 src/evaluate_script.py --source_path NMT-Data/Eval_Municipal/complex --reference_path NMT-Data/Eval_Municipal/simple --target_path $opensubtitles_folder/model_output/mun_Model_3_pred.txt.desubword --chart_title "Municipal Pipeline Evaluation with Opensubtitles Subset Bert"