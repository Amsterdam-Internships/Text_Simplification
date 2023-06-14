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
cat $europarl_folder/Europarl.en-nl.nl $opensubtitles_folder/munsubset.nl >> $europarl_folder/extended_europarl.nl
cat $europarl_folder/Europarl.en-nl.en $opensubtitles_folder/munsubset.en >> $europarl_folder/extended_europarl.en

sudo wget -c https://cs.pomona.edu/~dkauchak/simplification/data.v1/data.v1.tar.gz -O $wikisimple_folder/WikiSimple.tar.gz #WikiSimple Download
tar -xvf $wikisimple_folder/WikiSimple.tar.gz -C $wikisimple_folder #WikiSimple unzip
#sudo unzip $wikisimple_folder/WikiSimple.tar.gz -d $wikisimple_folder


# if NMT-Data/Model_English_S_Dutch_S/opensubtitles_en_testing doesnt exist:
if [[ ! -f NMT-Data/Model_English_S_Dutch_S/munsubset.en ]] #OpenSubtiltes Download params: (--reference file, --output_path_nl, --output_path_en, --num_samples --)
then
    python scripts/extract_sentences.py --reference_file NMT-Data/reference_mun.txt --output_path_en $opensubtitles_folder/munsubset.en --output_path_nl $opensubtitles_folder/munsubset.nl
else 
    echo "OpenSubtitles Municipal Subset already exists"
fi

#Preprocess data
#python3 notebooks/data_processing/filter.py $europarl_folder/extended_europarl.nl $europarl_folder/extended_europarl.en nl en #filter EMEA
#python3 notebooks/data_processing/filter.py $wikisimple_folder/data.v1/wiki.unsimplified $wikisimple_folder/data.v1/wiki.simple en_c en_s #filter WikiSimple
#python3 notebooks/data_processing/filter.py $opensubtitles_folder/munsubset.en $opensubtitles_folder/munsubset.nl en nl #filter opensubtitles

#build subword models
#python3 notebooks/data_processing/1_train_bpe.py --train_data_path $europarl_folder/extended_europarl.nl-filtered.nl --model_path $europarl_folder/subword_model/extended_source.model #train bpe for EMEA source
#python3 notebooks/data_processing/1_train_bpe.py --train_data_path $europarl_folder/extended_europarl.en-filtered.en --model_path $europarl_folder/subword_model/extended_target.model #train bpe for EMEA target

#python3 notebooks/data_processing/1_train_bpe.py --train_data_path $wikisimple_folder/data.v1/wiki.unsimplified-filtered.en_c --model_path NMT-Data/Model_English_C_English_S/subword_model/yttm_source.model #train bpe for WikiSimple source
#python3 notebooks/data_processing/1_train_bpe.py --train_data_path $wikisimple_folder/data.v1/wiki.simple-filtered.en_s --model_path NMT-Data/Model_English_C_English_S/subword_model/yttm_target.model #train bpe for WikiSimple target

#python3 notebooks/data_processing/1_train_bpe.py --train_data_path $opensubtitles_folder/munsubset.en-filtered.en --model_path NMT-Data/Model_English_S_Dutch_S/subword_model/mun_yttm_source.model #train bpe for OpenSubtitles source
#python3 notebooks/data_processing/1_train_bpe.py --train_data_path $opensubtitles_folder/munsubset.nl-filtered.nl --model_path NMT-Data/Model_English_S_Dutch_S/subword_model/mun_yttm_target.model #train bpe for OpenSubtitles target


#subword data
#python3 notebooks/data_processing/2_subword.py --model_path $europarl_folder/subword_model/extended_source.model --input_path $europarl_folder/extended_europarl.nl-filtered.nl --output_path $europarl_folder/extended_europarl.nl-filtered.nl-subword.nl #subword EMEA source
#python3 notebooks/data_processing/2_subword.py --model_path $europarl_folder/subword_model/extended_target.model --input_path $europarl_folder/extended_europarl.en-filtered.en --output_path $europarl_folder/extended_europarl.en-filtered.en-subword.en #subword EMEA target

#python3 notebooks/data_processing/2_subword.py --model_path $wikisimple_folder/subword_model/yttm_source.model --input_path NMT-Data/Model_English_C_English_S/data.v1/wiki.unsimplified-filtered.en_c --output_path NMT-Data/Model_English_C_English_S/data.v1/wiki.unsimplified-filtered.en_c-subword.en_c #subword WikiSimple
#python3 notebooks/data_processing/2_subword.py --model_path $wikisimple_folder/subword_model/yttm_target.model --input_path NMT-Data/Model_English_C_English_S/data.v1/wiki.simple-filtered.en_s --output_path NMT-Data/Model_English_C_English_S/data.v1/wiki.simple-filtered.en_s-subword.en_s #subword WikiSimple

#python3 notebooks/data_processing/2_subword.py --model_path $opensubtitles_folder/subword_model/mun_yttm_source.model --input_path NMT-Data/Model_English_S_Dutch_S/munsubset.en-filtered.en --output_path NMT-Data/Model_English_S_Dutch_S/munsubset.en-filtered.en-subword.en #subword OpenSubtitles source
#python3 notebooks/data_processing/2_subword.py --model_path $opensubtitles_folder/subword_model/mun_yttm_target.model --input_path NMT-Data/Model_English_S_Dutch_S/munsubset.nl-filtered.nl --output_path NMT-Data/Model_English_S_Dutch_S/munsubset.nl-filtered.nl-subword.nl #subword Opensubtitles target

#Split data into train/test/dev
#python3 notebooks/data_processing/train_dev_test_split.py 2000 2000 $europarl_folder/extended_europarl.nl-filtered.nl-subword.nl $europarl_folder/extended_europarl.en-filtered.en-subword.en #split EMEA data

#python3 notebooks/data_processing/train_dev_test_split.py 2000 2000 $wikisimple_folder/data.v1/wiki.unsimplified-filtered.en_c-subword.en_c NMT-Data/Model_English_C_English_S/data.v1/wiki.simple-filtered.en_s-subword.en_s #split WikiSimple data

#python3 notebooks/data_processing/train_dev_test_split.py 2000 2000 NMT-Data/Model_English_S_Dutch_S/munsubset.nl-filtered.nl-subword.nl NMT-Data/Model_English_S_Dutch_S/munsubset.en-filtered.en-subword.en #split OpenSubtitles data

#Build Vocab
#EMEA
#touch $europarl_folder/run/extended_mun_vocab.src; #create empty files where vocab will be stored
#touch $europarl_folder/run/extended_mun_vocab.tgt;
#onmt_build_vocab -config config/ext_mun_model_1.yaml -overwrite True #Build vocab

#WikiSimple
#touch $wikisimple_folder/run/vocab.src; #create empty files where vocab will be stored
#touch $wikisimple_folder/run/vocab.tgt;
#onmt_build_vocab -config config/model_2.yaml -overwrite True #Build vocab

#OpenSubtitles
#touch $opensubtitles_folder/run/ext_mun_vocab.src; #create empty files where vocab will be stored
#touch $opensubtitles_folder/run/ext_mun_vocab.tgt;
#onmt_build_vocab -config config/ext_mun_model_3.yaml -overwrite True #build vocab

#Train Models
#onmt_train -config config/ext_mun_model_1.yaml -src_vocab NMT-Data/europarl/run/ext_mun_vocab.src -tgt_vocab NMT-Data/europarl/run/ext_mun_vocab.tgt #Train Emea model
#onmt_train -config config/model_2.yaml #-src_vocab $wikisimple_folder/run/.vocab.src -tgt_vocab $wikisimple_folder/run/.vocab.tgt #Train WikiSimple model
#onmt_train -config config/ext_mun_model_3.yaml #-src_vocab NMT-Data/Model_English_S_Dutch_S/run/.vocab.src -tgt_vocab NMT-Data/Model_English_S_Dutch_S/run/.vocab.tgt #train OpenSubtitles Model

#Run input data through pipeline
#model 1
python3 notebooks/data_processing/2_subword.py --model_path $europarl_folder/subword_model/extended_source.model --input_path NMT-Data/Eval_Municipal/complex --output_path NMT-Data/Eval_Municipal/mun_complex_subword_1
onmt_translate -ban_unk_token -model $europarl_folder/model/extended_model_step_10000.pt -src NMT-Data/Eval_Municipal/mun_complex_subword_1 -output $europarl_folder/model_output/mun_Model_1_pred.txt -verbose 
python3 notebooks/data_processing/3_desubword.py --model_path $europarl_folder/subword_model/extended_target.model --input_path $europarl_folder/model_output/mun_Model_1_pred.txt --output_path $europarl_folder/model_output/mun_Model_1_pred.txt.desubword

#model 2
python3 notebooks/data_processing/2_subword.py --model_path $wikisimple_folder/subword_model/yttm_source.model --input_path $europarl_folder/model_output/mun_Model_1_pred.txt.desubword --output_path NMT-Data/Eval_Medical_Dutch_C_Dutch_S/NL_test_org_mun_subword_2
onmt_translate -ban_unk_token -model $wikisimple_folder/model/mybasemodel_step_10000.pt -src NMT-Data/Eval_Medical_Dutch_C_Dutch_S/NL_test_org_mun_subword_2 -output $wikisimple_folder/model_output/mun_Model_2_pred.txt -verbose
python3 notebooks/data_processing/3_desubword.py --model_path $wikisimple_folder/subword_model/yttm_target.model --input_path $wikisimple_folder/model_output/mun_Model_2_pred.txt --output_path $wikisimple_folder/model_output/mun_Model_2_pred.txt.desubword

#model 3
python3 notebooks/data_processing/2_subword.py --model_path $europarl_folder/subword_model/extended_target.model --input_path $wikisimple_folder/model_output/mun_Model_2_pred.txt.desubword --output_path NMT-Data/Eval_Municipal/mun_NL_test_org_subword_3
onmt_translate -ban_unk_token -model NMT-Data/Model_English_S_Dutch_S/model/extended_model_step_10000.pt -src NMT-Data/Eval_Municipal/mun_NL_test_org_subword_3 -output $opensubtitles_folder/model_output/mun_Model_3_pred.txt -verbose
python3 notebooks/data_processing/3_desubword.py --model_path $europarl_folder/subword_model/extended_source.model --input_path $opensubtitles_folder/model_output/mun_Model_3_pred.txt --output_path $opensubtitles_folder/model_output/mun_Model_3_pred.txt.desubword

#Evaluate Results
python3 notebooks/evaluate_script.py --source_path NMT-Data/Eval_Municipal/complex --reference_path NMT-Data/Eval_Municipal/simple --target_path $opensubtitles_folder/model_output/mun_Model_3_pred.txt.desubword --chart_title "Extended Europarl Pipe error calc $1"