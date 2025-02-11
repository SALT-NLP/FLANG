# When FLUE Meets FLANG: Benchmarks and Large Pretrained Language Model for Financial Domain
## Abstract
<p align="justify">
Pre-trained language models have shown impressive performance on a variety of tasks and domains. Previous research on financial language models usually employs a generic training scheme to train standard model architectures, without completely leveraging the richness of the financial data. We propose a novel domain specific Financial LANGuage model (FLANG) which uses financial keywords and phrases for better masking, together with span boundary objective and in-filing objective. Additionally, the evaluation benchmarks in the field have been limited. To this end, we contribute the Financial Language Understanding Evaluation (FLUE), an open-source comprehensive suite of benchmarks for the financial domain. These include new benchmarks across 5 NLP tasks in financial domain as well as common benchmarks used in the previous research. Experiments on these benchmarks suggest that our model outperforms those in prior literature on a variety of NLP tasks. 
</p>

## FLANG
FLANG is a set of large language models for Financial LANGuage tasks. These models use domain specific pre-training with preferential masking to build more robust representations for the domain. The models in the set are:\
[FLANG-BERT](https://huggingface.co/SALT-NLP/FLANG-BERT)\
[FLANG-SpanBERT](https://huggingface.co/SALT-NLP/FLANG-SpanBERT)\
[FLANG-DistilBERT](https://huggingface.co/SALT-NLP/FLANG-DistilBERT)\
[FLANG-Roberta](https://huggingface.co/SALT-NLP/FLANG-Roberta)\
[FLANG-ELECTRA](https://huggingface.co/SALT-NLP/FLANG-ELECTRA)


## FLANG-ELECTRA Architecture
![Architecture of our model. We use finance specific datasets and general English datasets (Wikpedia and BooksCorpus) for training the model. We follow the training strategy of ELECTRA with span boundary task which first predicts masked tokens using language model and then uses a discriminator to assess if a token is original or replaced. The generator and discriminator are trained end-to-end, and both words and phrases from financial vocabulary are used for masking. The final discriminator is then fine-tuned on individual tasks on our contributed benchmark suite, Financial Language Understanding Evaluation (FLUE). Note that our method is not specific to ELECTRA and can be generalized to other models.](/images/flang.jpg)
<sub>Architecture of our model. We use finance specific datasets and general English datasets (Wikpedia and BooksCorpus) for training the model. We follow the training strategy of ELECTRA with span boundary task which first predicts masked tokens using language model and then uses a discriminator to assess if a token is original or replaced. The generator and discriminator are trained end-to-end, and both words and phrases from financial vocabulary are used for masking. The final discriminator is then fine-tuned on individual tasks on our contributed benchmark suite, Financial Language Understanding Evaluation (FLUE). Note that our method is not specific to ELECTRA and can be generalized to other models.</sub>

## FLUE: Financial Language Understanding Evaluation
FLUE (Financial Language Understanding Evaluation) is a comprehensive and heterogeneous benchmark that has been built from 5 diverse financial domain specific datasets.

| Name       | Task                             | Source                  | Dataset Size |
|------------|----------------------------------|-------------------------|--------------|
| FPB        | Financial Sentiment Analysis     | Malo et al. 2014b       | 4,845        |
| FiQA SA    | Financial Sentiment Analysis     | FiQA 2018               | 1,173        |
| Headline   | News Headline Classification     | Sinha and Khandait 2020 | 11,412       |
| NER        | Named Entity Recognition         | Alvarado et al. 2015    | 1,466        |
| FinSBD3    | Structure Boundary Detection     | FinSBD3 (FinWeb-2021)   | 756          |
| FiQA QA    | Question Answering               | FiQA 2018               | 6,640        |

### Performance of FLANG Architecture of FLUE datasets

| Model/Metric        | FPB      | FiQA SA | Headline | NER  | FinSBD3 | FiQA QA |
| ------------------- | -------- | ------- | -------- | ---- | ------- | ------- |
|                     | Accuracy | MSE     | Mean F-1 | F-1  | F-1     | nDCG    |
| BERT-base           | 0.856    | 0.073   | 0.967    | 0.79 | 0.95    | 0.46    |
| FinBERT             | 0.872    | 0.07    | 0.968    | 0.8  | 0.89    | 0.42    |
| FLANG-BERT(ours)    | 0.912    | 0.054   | 0.972    | 0.83 | 0.96    | 0.51    |
| ELECTRA             | 0.881    | 0.066   | 0.966    | 0.78 | 0.94    | 0.52    |
| FLANG-ELECTRA(ours) | 0.919    | 0.034   | 0.98     | 0.82 | 0.97    | 0.55    |

### Financial Sentiment Analysis
1. Financial PhraseBank (Classification)
    * Data: [Financial PhraseBank](https://huggingface.co/datasets/financial_phrasebank)
    * Cite: ```Malo, Pekka, et al. "Good debt or bad debt: Detecting semantic orientations in economic texts." Journal of the Association for Information Science and Technology 65.4 (2014): 782-796.```
2. FiQA 2018 Task-1 (Regression)
    * Data and Ref: [FiQA 2018](https://huggingface.co/datasets/SALT-NLP/FLUE-FiQA)
    * Cite: ```Maia, Macedo & Handschuh, Siegfried & Freitas, Andre & Davis, Brian & McDermott, Ross & Zarrouk, Manel & Balahur, Alexandra. (2018). WWW'18 Open Challenge: Financial Opinion Mining and Question Answering. WWW '18: Companion Proceedings of the The Web Conference 2018. 1941-1942. 10.1145/3184558.3192301.``` 

### News Headline Classification
 * Data: [Gold Commodity News and Dimensions](https://www.kaggle.com/datasets/daittan/gold-commodity-news-and-dimensions)
 * Cite: ```Sinha, A., & Khandait, T. (2021, April). Impact of News on the Commodity Market: Dataset and Results. In Future of Information and Communication Conference (pp. 589-601). Springer, Cham.```

### Named Entity Recognition
 * Data: [NER Data on Loan Agreement](https://paperswithcode.com/dataset/fin)
 * Cite: ```Alvarado, J. C. S., Verspoor, K., & Baldwin, T. (2015, December). Domain adaption of named entity recognition to support credit risk assessment. In Proceedings of the Australasian Language Technology Association Workshop 2015 (pp. 84-90).```

### Structure Boundary Detection
 * Data: [FinSBD3](https://sites.google.com/nlg.csie.ntu.edu.tw/finweb2021/shared-task-finsbd-3)
 * Cite: ```Willy Au, Abderrahim Ait-Azzi, and Juyeon Kang. 2021. FinSBD-2021: The 3rd Shared Task on Structure Boundary Detection in Unstructured Text in the Financial Domain. In Companion Proceedings of the Web Conference 2021 (WWW '21). Association for Computing Machinery, New York, NY, USA, 276–279. https://doi.org/10.1145/3442442.3451378```

### Question Answering
 * Data and Ref: [FiQA 2018](https://huggingface.co/datasets/SALT-NLP/FLUE-FiQA)
 * Cite: ```Maia, Macedo & Handschuh, Siegfried & Freitas, Andre & Davis, Brian & McDermott, Ross & Zarrouk, Manel & Balahur, Alexandra. (2018). WWW'18 Open Challenge: Financial Opinion Mining and Question Answering. WWW '18: Companion Proceedings of the The Web Conference 2018. 1941-1942. 10.1145/3184558.3192301.```

## Leaderboard
Coming soon!

## Citation
Please cite the model with the following citation:
```bibtex
@inproceedings{shah-etal-2022-flue,
    title = "When {FLUE} Meets {FLANG}: Benchmarks and Large Pretrained Language Model for Financial Domain",
    author = "Shah, Raj  and
      Chawla, Kunal  and
      Eidnani, Dheeraj  and
      Shah, Agam  and
      Du, Wendi  and
      Chava, Sudheer  and
      Raman, Natraj  and
      Smiley, Charese  and
      Chen, Jiaao  and
      Yang, Diyi",
    editor = "Goldberg, Yoav  and
      Kozareva, Zornitsa  and
      Zhang, Yue",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.148",
    doi = "10.18653/v1/2022.emnlp-main.148",
    pages = "2322--2335",
    abstract = "Pre-trained language models have shown impressive performance on a variety of tasks and domains. Previous research on financial language models usually employs a generic training scheme to train standard model architectures, without completely leveraging the richness of the financial data. We propose a novel domain specific Financial LANGuage model (FLANG) which uses financial keywords and phrases for better masking, together with span boundary objective and in-filing objective. Additionally, the evaluation benchmarks in the field have been limited. To this end, we contribute the Financial Language Understanding Evaluation (FLUE), an open-source comprehensive suite of benchmarks for the financial domain. These include new benchmarks across 5 NLP tasks in financial domain as well as common benchmarks used in the previous research. Experiments on these benchmarks suggest that our model outperforms those in prior literature on a variety of NLP tasks. Our models, code and benchmark data will be made publicly available on Github and Huggingface.",
}
```

## Contact information
Please contact Raj Sanjay Shah (rajsanjayshah[at]gatech[dot]edu) or Sudheer Chava (schava6[at]gatech[dot]edu) or Diyi Yang (diyiy[at]stanford[dot]edu) about any issues and questions.
