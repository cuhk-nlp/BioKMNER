# BioKMNER

This is the implementation of [Improving Biomedical Named Entity Recognition with Syntactic Information](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03834-6) at BMC Bioinformatics.

Please contact us at `yhtian@uw.edu` or `cuhksz.nlp@gmail.com` if you have any questions.


## Citation

If you use or extend our work, please cite our paper at ACL2020.

```
@article{tian2020improving,
  title={Improving Biomedical Named Entity Recognition with Syntactic Information},
  author={Tian, Yuanhe and Shen, Wang and Song, Yan and Xia, Fei and He, Min and Li, Kenli},
  year={2020}
  jurnal={BMC Bioinformatics}
  volume={21}
  page={539}
}
```

## Environment

The code works with the following environment:

* `python=3.6`
* `pytorch=1.1`


## Data

Following [BioBERT](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506), the data used in our paper can be found at [here](https://github.com/dmis-lab/biobert#datasets) (or [here](https://drive.google.com/open?id=1OletxmPYNkz2ltOr9pyT0b0iBtUWxslh)). You can see our [sample data](./data/sample_data) for reference.

To obtain the syntactic information, please follow the following steps:
1. Download [Stanford CoreNLP Toolkits (v3.9.2)](https://stanfordnlp.github.io/CoreNLP/history.html) and put the folder ``stanford-corenlp-full-2018-10-05`` under the current directory.
2. Run `python data_helper.py --dataset=/path/to/the/dataset/` to preprocess the data.

## Run on Sample Data

To run our code, you first need to set the environment up and download [biobert](https://github.com/naver/biobert-pretrained) and put it into biobert_pyt directory (please use our config.json file).

If the model is tf version, you need to [convert](https://github.com/huggingface/transformers) it to pytorch version.

Also, you need to replace original the config.json in your model directory with the config.json in the bert model directory provided by us.

You can run run.sh directly to train and evaluate our model on the sample data.

## To-do list

* Release our pre-trained models.
* Regular maintenance.

We will keep updating this repository recently.

