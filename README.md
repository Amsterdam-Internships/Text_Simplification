# Automatic Text Simplification for Low Resource Languages using a Pivot Approach

This repository contains the code to run a pivot-based text simplification for the Dutch medical domain and municipal domains.
The full pipeline consists of the 3 models:
* 1st model (M<sup>NL&rarr;EN</sup>): Translates complex dutch sentences to complex english sentences
* 2nd Model (M<sup>C&rarr;S</sup>): Simplifies complex english sentences to simple english sentences
* 3rd Model (M<sup>EN&rarr;NL</sup>): Translates simple english sentences to simple dutch sentences

On top of training the models, the repo contains code for evaluating the pipeline's quality using a number of automatic evaluation metrics (BLEU,SARI,METEOR).

[//]: # (![]&#40;./media/pivot_pipeline_TS.png&#41;)
<div align="center">
   <img src="./media/pivot_pipeline_TS.png" width="600"/>
   <br>
   <em>Figure 1. Pivot pipeline for text simplification</em>
</div>

## Project Folder Structure

Explain briefly what's where so people can find their way around. For example:

There are the following folders in the structure:


1) [`scripts`](./scripts): Folder with the scripts used to perform all experiments,
including individual bash scripts for each one of the pivot-based models pipelines and
a python script for the [gpt-based experiment](./scripts/chatgpt.py).
1) [`src`](./src): Folder containing all supporting code, such as
preprocessing and filtering scripts, tokenization,
extraction of domain-specific subsets of the translation corpora, etc.
1) [`config`](./config): Folder containing configuration files for the training of the models
1) [`NMT-Data`](./NMT-Data): Folder where all data will be downloaded and models will be saved
1) [`media`](./media): Folder containing media files for demo purposes

[//]: # (1&#41; [`notebooks`]&#40;./notebooks&#41;: folder containing notebooks for running the pipeline as well as data-processing scripts for filtering, subwording, desubwording and splitting data)

## Installation
You can install this repo by following these steps:

1) Clone this repository:
    ```bash
    git clone https://github.com/Amsterdam-Internships/Text_Simplification
    ```

1) Install all dependencies:
    ```bash
    pip install -r requirements.txt
    ```
---

## Usage
To Run the pipeline the script expects evaluation data to be uploaded: <br>
Original sentences: NMT-Data/Eval_Medical_Dutch_C_Dutch_S/NL_test_org <br>
Simplified sentences: NMT-Data/Eval_Medical_Dutch_C_Dutch_S/NL_test_simp

In many of our experiments we use in-domain data, extracted from the Opensubtitles corpus on the basis of similarity to a reference corpus. To generate these in-domain data use the following script.

    python scripts/extract_sentences.py

If you wish to create your own in-domain subset you can substitute the reference_file, as well as tweak other arguments such as encoding_method and num_samples.

By default, the extract_sentences.py script will generate an in-domain medical translation corpora. Which is used in many of our pipelines. The default pipeline script downloads data, processes it, trains the relevant models, performs the simplification and evaluates the simplification. It can be executed using the following script:

```
$ /scripts/run_pipeline.sh
```

Different pipeline setups are available in the scripts folder.



---
## Acknowledgements
Our code uses preproccesing scripts from [MT-Preparation](https://github.com/ymoslem/MT-Preparation)