# Learning Protein Secondary Structure from Sequences Using Transformer as Language Model.

This project uses BERT models from [ProtTrans](https://www.biorxiv.org/content/10.1101/2020.07.12.199554v2) to train proteins for secondary structure prediction (Q8 - the 8-class version). System is designed to add one more label (`X` - unknown) to the set to capture unknown or invalid amino acids. Two versions of the model are used: one trained on [UniRef100](https://www.nature.com/articles/s41592-019-0598-1) and the other trained on BFD dataset.

## Installation

`conda install -n <env_name> requirements.txt`

If for some reason, installation of some packages fail, it may be because it's not on conda. In that case, use `pip`.

## Run

Run `python main.py --help` to see all available options. All parameters have default value so **learn about the parameters** before deciding to run. The default is to train. To test, pass in `-t`.

