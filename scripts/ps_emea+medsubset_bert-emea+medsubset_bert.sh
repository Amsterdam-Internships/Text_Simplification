#This file executes the full pipeline for pivot based neural machine translation.
#This includes downloading and preprocessing data, subwording data, training 3 translation models, and evaluating results on a test set.


pip install openNMT-py
pip install sacremoses


data_folder=NMT-Data
emea_folder=$data_folder/Model_Dutch_C_English_C
wikisimple_folder=$data_folder/Model_English_C_English_S
opensubtitles_folder=$data_folder/Model_English_S_Dutch_S
experiment_prefix=model
steps=`cat ${experiment_prefix}_3.yaml | grep train_steps | cut -f2 -d ' '`

#Download Data
wget https://opus.nlpl.eu/download.php?f=EMEA/v3/moses/en-nl.txt.zip -O $emea_folder/emea_en-nl.txt.zip #EMEA download
unzip $emea_folder/emea_en-nl.txt.zip -d $emea_folder -n #unzip EMEA
cat $emea_folder/EMEA.en-nl.nl $opensubtitles_folder/opensubtitles_nl_testing >> $emea_folder/extended_emea.nl
cat $emea_folder/EMEA.en-nl.en $opensubtitles_folder/opensubtitles_en_testing >> $emea_folder/extended_emea.en


sudo wget -c https://cs.pomona.edu/~dkauchak/simplification/data.v1/data.v1.tar.gz -O $wikisimple_folder/WikiSimple.tar.gz #WikiSimple Download
tar -xvf $wikisimple_folder/WikiSimple.tar.gz -C $wikisimple_folder #WikiSimple unzip
#sudo unzip $wikisimple_folder/WikiSimple.tar.gz -d $wikisimple_folder

# if NMT-Data/Model_English_S_Dutch_S/opensubtitles_en_testing doesnt exist:
if [[ ! -f NMT-Data/Model_English_S_Dutch_S/medsubset_bert.en ]] #OpenSubtiltes Download params: (--reference file, --output_path_nl, --output_path_en, --num_samples --)
then
    python scripts/extract_sentences.py --reference_file NMT-Data/CleanSimpleMedicalCorpora.txt --output_path_en $opensubtitles_folder/medsubset_bert.en --output_path_nl $opensubtitles_folder/medsubset_bert.nl --encoding_method sentence_transformer
else 
    echo "OpenSubtitles Medical Subset already exists"
fi

#Preprocess data
python3 notebooks/data_processing/filter.py $emea_folder/extended_emea.nl $emea_folder/extended_emea.en nl en #filter EMEA
python3 notebooks/data_processing/filter.py $wikisimple_folder/data.v1/wiki.unsimplified $wikisimple_folder/data.v1/wiki.simple en_c en_s #filter WikiSimple

#build subword models
python3 notebooks/data_processing/1_train_bpe.py --train_data_path $emea_folder/extended_emea.nl-filtered.nl --model_path $emea_folder/subword_model/extended_source.model #train bpe for EMEA source
python3 notebooks/data_processing/1_train_bpe.py --train_data_path $emea_folder/extended_emea.en-filtered.en --model_path $emea_folder/subword_model/extended_target.model #train bpe for EMEA target

python3 notebooks/data_processing/1_train_bpe.py --train_data_path $wikisimple_folder/data.v1/wiki.unsimplified-filtered.en_c --model_path NMT-Data/Model_English_C_English_S/subword_model/yttm_source.model #train bpe for WikiSimple source
python3 notebooks/data_processing/1_train_bpe.py --train_data_path $wikisimple_folder/data.v1/wiki.simple-filtered.en_s --model_path NMT-Data/Model_English_C_English_S/subword_model/yttm_target.model #train bpe for WikiSimple target

#subword data
python3 notebooks/data_processing/2_subword.py --model_path $emea_folder/subword_model/extended_source.model --input_path $emea_folder/extended_emea.nl-filtered.nl --output_path $emea_folder/extended_emea.nl-filtered.nl-subword.nl #subword EMEA source
python3 notebooks/data_processing/2_subword.py --model_path $emea_folder/subword_model/extended_target.model --input_path $emea_folder/extended_emea.en-filtered.en --output_path $emea_folder/extended_emea.en-filtered.en-subword.en #subword EMEA target

python3 notebooks/data_processing/2_subword.py --model_path $wikisimple_folder/subword_model/yttm_source.model --input_path NMT-Data/Model_English_C_English_S/data.v1/wiki.unsimplified-filtered.en_c --output_path NMT-Data/Model_English_C_English_S/data.v1/wiki.unsimplified-filtered.en_c-subword.en_c #subword WikiSimple
python3 notebooks/data_processing/2_subword.py --model_path $wikisimple_folder/subword_model/yttm_target.model --input_path NMT-Data/Model_English_C_English_S/data.v1/wiki.simple-filtered.en_s --output_path NMT-Data/Model_English_C_English_S/data.v1/wiki.simple-filtered.en_s-subword.en_s #subword WikiSimple

#Split data into train/test/dev
python3 notebooks/data_processing/train_dev_test_split.py 2000 2000 $emea_folder/extended_emea.nl-filtered.nl-subword.nl $emea_folder/extended_emea.en-filtered.en-subword.en #split EMEA data

python3 notebooks/data_processing/train_dev_test_split.py 2000 2000 $wikisimple_folder/data.v1/wiki.unsimplified-filtered.en_c-subword.en_c NMT-Data/Model_English_C_English_S/data.v1/wiki.simple-filtered.en_s-subword.en_s #split WikiSimple data

#Build Vocab
#EMEA
touch $emea_folder/run/extended_vocab.src; #create empty files where vocab will be stored
touch $emea_folder/run/extended_vocab.tgt;
onmt_build_vocab -config config/model_nl-en_emea_medsubset_bert.yaml -overwrite True #Build vocab

#WikiSimple
touch $wikisimple_folder/run/vocab.src; #create empty files where vocab will be stored
touch $wikisimple_folder/run/vocab.tgt;
onmt_build_vocab -config config/model_en_c-s.yaml -overwrite True #Build vocab

#OpenSubtitles
touch $opensubtitles_folder/run/extended_emea_vocab.src; #create empty files where vocab will be stored
touch $opensubtitles_folder/run/extended_emea_vocab.tgt;
onmt_build_vocab -config config/model_en-nl_emea+medsubset_bert.yaml -overwrite True #build vocab

#Train Models
#onmt_train -config config/model_nl-en_emea_medsubset_bert.yaml #-src_vocab $emea_folder/run/.vocab.src -tgt_vocab $emea_folder/run/.vocab.tgt #Train Emea model
#onmt_train -config config/model_en_c-s.yaml #-src_vocab $wikisimple_folder/run/.vocab.src -tgt_vocab $wikisimple_folder/run/.vocab.tgt #Train WikiSimple model
onmt_train -config config/model_en-nl_emea+medsubset_bert.yaml #-src_vocab NMT-Data/Model_English_S_Dutch_S/run/.vocab.src -tgt_vocab NMT-Data/Model_English_S_Dutch_S/run/.vocab.tgt #train OpenSubtitles Model

#Run input data through pipeline
#model 1
python3 notebooks/data_processing/2_subword.py --model_path $emea_folder/subword_model/extended_source.model --input_path NMT-Data/Eval_Medical_Dutch_C_Dutch_S/NL_test_org --output_path NMT-Data/Eval_Medical_Dutch_C_Dutch_S/NL_test_org_subword_1
onmt_translate -ban_unk_token -model $emea_folder/model/extended_model_step_10000.pt -src NMT-Data/Eval_Medical_Dutch_C_Dutch_S/NL_test_org_subword_1 -output $emea_folder/model_output/Model_1_pred.txt -verbose 
python3 notebooks/data_processing/3_desubword.py --model_path $emea_folder/subword_model/extended_target.model --input_path $emea_folder/model_output/Model_1_pred.txt --output_path $emea_folder/model_output/Model_1_pred.txt.desubword

#model 2
python3 notebooks/data_processing/2_subword.py --model_path $wikisimple_folder/subword_model/yttm_source.model --input_path $emea_folder/model_output/Model_1_pred.txt.desubword --output_path NMT-Data/Eval_Medical_Dutch_C_Dutch_S/NL_test_org_subword_2
onmt_translate -ban_unk_token -model $wikisimple_folder/model/mybasemodel_step_10000.pt -src NMT-Data/Eval_Medical_Dutch_C_Dutch_S/NL_test_org_subword_2 -output $wikisimple_folder/model_output/Model_2_pred.txt -verbose
python3 notebooks/data_processing/3_desubword.py --model_path $wikisimple_folder/subword_model/yttm_target.model --input_path $wikisimple_folder/model_output/Model_2_pred.txt --output_path $wikisimple_folder/model_output/Model_2_pred.txt.desubword

#model 3
python3 notebooks/data_processing/2_subword.py --model_path $emea_folder/subword_model/extended_target.model --input_path $wikisimple_folder/model_output/Model_2_pred.txt.desubword --output_path NMT-Data/Eval_Medical_Dutch_C_Dutch_S/NL_test_org_subword_3
onmt_translate -ban_unk_token -model $opensubtitles_folder/model/extended_emea_model_step_10000.pt -src NMT-Data/Eval_Medical_Dutch_C_Dutch_S/NL_test_org_subword_3 -output $opensubtitles_folder/model_output/Model_3_pred.txt -verbose
python3 notebooks/data_processing/3_desubword.py --model_path $emea_folder/subword_model/extended_source.model --input_path $opensubtitles_folder/model_output/Model_3_pred.txt --output_path $opensubtitles_folder/model_output/Model_3_pred.txt.desubword

#Evaluate Results
python3 notebooks/evaluate_script.py --source_path NMT-Data/Eval_Medical_Dutch_C_Dutch_S/NL_test_org --reference_path NMT-Data/Eval_Medical_Dutch_C_Dutch_S/NL_test_simp --target_path $opensubtitles_folder/model_output/Model_3_pred.txt.desubword --chart_title "Extended Emea Medical Pipeline"