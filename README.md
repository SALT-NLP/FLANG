# When FLUE Meets FLANG: Benchmarks and Large Pretrained Language Model for Financial Domain

## Requirements

The code requires Python 3.6+ with Pytorch 1.4+ and Huggingface library. Simple transformers library is needed for Electra models.

## Data

### Raw data

The merged financial dictionary that we use for preferential finance word and phrase masking is in vocabulary/

The raw data for the benchmarks is available at [Agam will add].

A detailed description of datasets is available in [this word doc](https://docs.google.com/document/d/1oMvJgLtz3f0dHPcDvvx3q63WadgRSVdIy5qV-806oxM/edit?usp=sharing)


### Processed data

Data can be sampled from various datasets in different proportions. Two such data samples are in the above link: "data/newsonly" contains only news data (Reuters + Bloomberg), while "data/all" contains data sampled from all datasets.

## Usage

### Training language model

To train BERT/RoBERTa/DistilBERT/DeBERTA and other language models, run

    ./scripts/train_bert.sh

To train Electra based models, use:

    python scripts/train_electra.py
    
Please modify the parameters in the script according to specifications.

### Fine-tune on downstream task

For Financial PhraseBank sentiment classification task, use:

    python scripts/fine_tune_bert.py
    python scripts/fine_tune_electra.py

For fine-tuing on more datasets, refer to code from [Dheeraj will add].

### Code changes from basic libraries

The code change from Huggingface/ SimpleTransformers library is in "src/"

## FLANG: Financial LANGuage model

### FLANG-BERT

FLANG-BERT was trained based on BERT model.

### FLANG-ELECTRA
FLANG-BERT was trained based on ELECTRA model.


## FLUE: Financial Language Understanding Evaluation
| Name       | Task                             | Source                  | Dataset Size |
|------------|----------------------------------|-------------------------|--------------|
| FPB        | Financial Sentiment Analysis     | Malo et al. 2014b       | 4,845        |
| FiQA SA    | Financial Sentiment Analysis     | FiQA 2018               | 1,173        |
| Headline   | News Headline Classification     | Sinha and Khandait 2020 | 11,412       |
| NER        | Named Entity Recognition         | Alvarado et al. 2015    | 1,466        |
| FinSBD3    | Structure Boundary Detection     | FinSBD3 (FinWeb-2021)   | 756          |
| FiQA QA    | Question Answering               | FiQA 2018               | 6,640        |
| Volatility | Volatility and Return Prediction | Ours                    | 6,500        |

### Financial Sentiment Analysis
1. Financial Phrase Bank (Classification)
    * Code: Kunal will upload 
    * Data: Kunal will provide reference
2. FiQA 2018 Task-1 (Regression)
    * Code: [FiQA Sentiment Analysis](https://github.com/GT-SALT/FLANG/blob/master/flue_benchmarks/fiqa_sentiment_analysis.py)
    * Data and Ref: [FiQA 2018](https://sites.google.com/view/fiqa/home)

### News Headline Classification
 * Code: [News Classification](https://github.com/GT-SALT/FLANG/blob/master/flue_benchmarks/news_headline_classification.py)
 * Data: [Gold Commodity News and Dimensions](https://www.kaggle.com/daittan/gold-commodity-news-and-dimensions/version/1)
 * Cite: ```Sinha, A., & Khandait, T. (2021, April). Impact of News on the Commodity Market: Dataset and Results. In Future of Information and Communication Conference (pp. 589-601). Springer, Cham.```

### Named Entity Recognition
 * Code: [NER](https://github.com/GT-SALT/FLANG/blob/master/flue_benchmarks/ner.py)
 * Data: [NER Data on Loan Agreement](https://people.eng.unimelb.edu.au/tbaldwin/resources/finance-sec/)
 * Cite: ```Alvarado, J. C. S., Verspoor, K., & Baldwin, T. (2015, December). Domain adaption of named entity recognition to support credit risk assessment. In Proceedings of the Australasian Language Technology Association Workshop 2015 (pp. 84-90).```

### Structure Boundary Detection
 * Code: [SBD](https://github.com/GT-SALT/FLANG/blob/master/flue_benchmarks/sbd.py)
 * Data: [FinSBD Data](https://drive.google.com/file/d/11GPrAD6plmTNf2652-B4SWVzfpmpL85P/view?usp=sharing)
 * Ref: [FinSBD-3](https://sites.google.com/nlg.csie.ntu.edu.tw/finweb2021/shared-task-finsbd-3)
 * License: [FinSBD-3 Licence](https://drive.google.com/file/d/1c_SLV4ek0UoGHEis-s78ObAsc3NYcqdX/view?usp=sharing)

### Question Answering
 * Code: Dheeraj will add
 * Data: Dheeraj will add

### Volatility and Market Reaction Prediction
 * Code: Dheeraj will add
 * Data: Dheeraj will add
