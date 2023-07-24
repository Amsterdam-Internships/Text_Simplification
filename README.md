# Automatic Text Simplification for Low Resource Languages using a Pivot Approach

This repository contains the code to run a pivot-based text simplification for the Dutch medical domain. It trains 3 models:
-1st model: Translates complex dutch sentences to complex english sentences
-2nd Model: Simplifies complex english sentences to simple english sentences
-3rd Model: Translates simple english sentences to simple dutch sentences

On top of training the models the repo performs the simplification and evaluates its quality using a number of automatic evaluation metrics (BLEU,SARI,METEOR)

![](media/Pipeline_Text_Simplification_Pivot.pdf)


## Project Folder Structure

Explain briefly what's where so people can find their way around. For example:

There are the following folders in the structure:


1) [`scripts`](./scripts): Folder with example scripts for performing different tasks (could serve as usage documentation)
1) [`notebooks`] (./notebooks): folder containing notebooks for running the pipeline as well as data-processing scripts for filtering, subwording, desubwording and splitting data
1) [`media`](./media): Folder containing media files (icons, video)
1) ['NMT-Data'](./NMT-Data): Folder where all data and models will be saved
1) [`config`](./config): Folder containing configuration files for the training of the models

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