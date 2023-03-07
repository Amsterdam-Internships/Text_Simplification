# Automatic Text Simplification for Low Resource Languages using a Pivot Approach


#TODO! 

Explain in short what this repository is. Mind the target audience.
No need to go into too much technical details if you expect some people would just use it as end-users 
and don't care about the internals (so focus on what the code really *does*), not how.
The *_How it works_* section below would contain more technical details for curious people.

If applicable, you can also show an example of the final output.

![](media/Pipeline_Text_Simplification_Pivot.pdf)


## Project Folder Structure


Explain briefly what's where so people can find their way around. For example:

There are the following folders in the structure:


1) [`src`](./src): Folder for all source files specific to this project
1) [`scripts`](./scripts): Folder with example scripts for performing different tasks (could serve as usage documentation)
1) ['notebooks'] (./notebooks): folder containing notebooks for running the pipeline
1) [`media`](./media): Folder containing media files (icons, video)

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


#TODO!

"download data from ... place it in ... etc"

Explain example usage, possible arguments, etc. E.g.:

To train... 


```
$ python train.py --some-importang-argument
```

If there are too many command line arguments, you can add a nice table with explanation (thanks, [Diana Epureano](https://www.linkedin.com/in/diana-epureanu-235104153/)!)

|Argument | Type or Action | Description | Default |
|---|:---:|:---:|:---:|
|`--batch_size`| int| `Batch size.`|  32|
|`--device`| str| `Training device, cpu or cuda:0.`| `cpu`|
|`--early-stopping`|  `store_true`| `Early stopping for training of sparse transformer.`| True|
|`--epochs`| int| `Number of epochs.`| 21|
|`--input_size`|  int| `Input size for model, i.e. the concatenation length of te, se and target.`| 99|
|`--loss`|  str|  `Type of loss to be used during training. Options: RMSE, MAE.`|`RMSE`|
|`--lr`|  float| `Learning rate.`| 1e-3|
|`--train_ratio`|  float| `Percentage of the training set.`| 0.7|
|...|...|...|...|


Alternatively, as a way of documenting the intended usage, you could add a `scripts` folder with a number of scripts for setting up the environment, performing training in different modes or different tasks, evaluation, etc (thanks, [Tom Lotze](https://www.linkedin.com/in/tom-lotze/)!)

---

## How it works

#TODO!

---
## Acknowledgements
Our code uses preproccesing scripts from [MT-Preparation](https://github.com/ymoslem/MT-Preparation)
