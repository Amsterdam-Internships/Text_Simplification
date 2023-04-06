# Automatic Text Simplification for Low Resource Languages using a Pivot Approach
#TODO! 

This repository contains the code to run a pivot-based text simplification for the dutch medical domain. It trains 3 models:
-1st model: Translates complex dutch sentences to complex english sentences 
-2nd Model: Simplifies complex english sentences to simple english sentences
-3rd Model: Translates simple english sentences to simple dutch sentences

On top of training the models the repo performs the simplification and evaluates its quality using a number of automatic evaluation metrics (BLEU,SARI,METEOR)

![](media/Pipeline_Text_Simplification_Pivot.pdf)


## Project Folder Structure

Explain briefly what's where so people can find their way around. For example:

There are the following folders in the structure:


1) [`src`](./src): Folder for all source files specific to this project
1) [`scripts`](./scripts): Folder with example scripts for performing different tasks (could serve as usage documentation)
1) ['notebooks'] (./notebooks): folder containing notebooks for running the pipeline as well as data-processing scripts for filtering, subwording, desubwording and splitting data
1) [`media`](./media): Folder containing media files (icons, video)
1) ['NMT-Data'](./NMT-Data): Folder where all data and models will be saved
1) [`config`](./config): Folder containing configuration files for the training of the models

## Installation

Explain how to set up everything. 
Let people know if there are weird dependencies - if so feel free to add links to guides and tutorials.

A person should be able to clone this repo, follow your instructions blindly, and still end up with something *fully working*!

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
To Run the pipeline the script expects evaluation data to be uploaded:
original sentences: NMT-Data/Eval_Medical_Dutch_C_Dutch_S/NL_test_org
simplified sentences: NMT-Data/Eval_Medical_Dutch_C_Dutch_S/NL_test_simp

The pipeline script also expects some reference sentences in order to extract useful sentences from the opensubtitles domain, by default these sentences are taken from the original sentences in the evaluation data. If you wish to change this you can do so in the run_pipeline.sh file by changing reference_file argument in the following line:

    python scripts/extract_sentences.py --reference_file NMT-Data/Eval_Medical_Dutch_C_Dutch_S/NL_test_org

Once this is done you can run scripts/run_pipeline.sh (bear in mind that training the models requires a machine with a gpu)

The pipeline script downloads data, processes it, trains the relevant models, performs the simplification and evaluates the simplification.

```
$ /scripts/run_pipeline.sh
```
---

## How it works

#TODO!
---
## Acknowledgements
Our code uses preproccesing scripts from [MT-Preparation](https://github.com/ymoslem/MT-Preparation)