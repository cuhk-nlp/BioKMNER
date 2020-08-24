# BioKMNER

We will keep updating this repository recently.

## To-do list

* Release the code to pre-process the data;
* Improve the code to train and test the models;
* Implement the code to predict the NE labels with a given sentence;
* Release our pre-trained models.

## Environment

The code works with the following environment:

* `python=3.6`
* `pytorch=1.1`


## Data

Following [BioBERT](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506), the data used in our paper can be found at [here](https://github.com/dmis-lab/biobert#datasets) (or [here](https://drive.google.com/open?id=1OletxmPYNkz2ltOr9pyT0b0iBtUWxslh)). You can see our sample data at [./data/sample_data](./data/sample_data) for reference.

## Run on Sample Data

To run our code, you first need to set the environment up and download [biobert](https://github.com/naver/biobert-pretrained) and put it into biobert_pyt directory (please use our config.json file).

If the model is tf version, you need to [convert](https://github.com/huggingface/transformers) it to pytorch version.

Also, you need to replace original the config.json in your model directory with the config.json in the bert model directory provided by us.

You can run run.sh directly to train and evaluate our model on the sample data.

