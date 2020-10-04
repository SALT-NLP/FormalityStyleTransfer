# Code for Semi-supervised Formality Style Transfer 

*Kunal Chawla, Diyi Yang*: Semi-supervised Formality Style Transfer using Language Model Discriminator and Mutual Information Maximization. In Findings of the 25th Annual Meeting of the Empirical Methods in Natural Language Processing (EMNLP'2020 Findings)

If you would like to refer to it, please cite the paper mentioned above. 

## Getting Started
Following are the instructions to get started with the code.

### Requirements
* Python 3.6 or higher
* Pytorch >= 1.2.0 (preferably with CUDA support)
* nltk


### Code Structure
```
|__ fairseq/
        |__ models				
            |__ BART
            	|__ model.py 						Model file
        |__ criterions
            |__ classification.py 					Pre-training Discrminator
            |__ label_smoothed_cross_entropy.py 	Training main model
        |__ criterions
            |__ language_pair_dataset.py 			Dataset Processing
        |__ trainer.py 								Training helper code
        |__ options.py 								Options and default values

|__fairseq_cli/
        |__ train.py 								Main training code
        |__ generate.py 							Generation code
|__ preprocess.ph 									Preprocess data
|__ pipeline.sh 									Training scipt
```

### Build the code

The code is based on Fairseq (https://github.com/pytorch/fairseq). To build it, run

    pip install --editable setup.py
Further instructions can be found on Fairseq official page.

## Dataset and Pre-Processing

The Grammarly Yahoo Corpus Dataset (GYAFC) is available on request from [here](https://github.com/raosudha89/GYAFC-corpus). Please download it and place it in the root directory. 

To preprocess the dataset, run

    bash preprocess.sh [options]
We followed the same instructions as [Fairseq's BART model](https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md). Follow the instructions for further converting the data to binary format that can be used for training.

## Training

Download the pre-trained "bart.large" model. To start training, run 

    bash pipeline.sh [options]

The details of all parameters are given in [fairseq/options.py](fairseq/options.py). For details on parameters and values, refer to the paper and appendix.

## Evaluation and Outputs

For generation, run 

    python evaluation/gen.py

Some folder paths may need to be changed depending on configuration. For evaluation and BLEU scores, run

    python evaluation/calc_score.py path_to_output_file

The outputs for our and various other models are given in [evaluation/outputs](evaluation/outputs). As mentioned in Table 4 of the paper, we provide outputs for Hybrid Annotations, Pretrained w/ rules, Ours and Target. "\_family" refers to F&R Domain and "\_music" refers to E&M Domain.
