# When FLUE Meets FLANG: Benchmarks and Large Pretrained Language Model for Financial Domain

## FLANG: Financial LANGuage model

### FLANG-BERT

### FLANG-ELECTRA

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
