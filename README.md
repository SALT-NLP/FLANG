# When FLUE Meets FLANG: Benchmarks and Large Pretrained Language Model for Financial Domain

## FLANG: Financial LANGuage model

### FLANG-BERT

### FLANG-ELECTRA

## FLUE: Financial Language Understanding Evaluation
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-1wig{font-weight:bold;text-align:left;vertical-align:top}
.tg .tg-0lax{text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-1wig">Name</th>
    <th class="tg-1wig">Task</th>
    <th class="tg-1wig">Source</th>
    <th class="tg-1wig">Dataset Size</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">FPB</td>
    <td class="tg-0lax">Financial Sentiment Analysis</td>
    <td class="tg-0lax">Malo et al. 2014b</td>
    <td class="tg-0lax">4,845</td>
  </tr>
  <tr>
    <td class="tg-0lax">FiQA SA</td>
    <td class="tg-0lax">Financial Sentiment Analysis</td>
    <td class="tg-0lax">FiQA 2018</td>
    <td class="tg-0lax">1,173</td>
  </tr>
  <tr>
    <td class="tg-0lax">Headline</td>
    <td class="tg-0lax">News Headline Classification</td>
    <td class="tg-0lax">Sinha and Khandait 2020</td>
    <td class="tg-0lax">11,412</td>
  </tr>
  <tr>
    <td class="tg-0lax">NER</td>
    <td class="tg-0lax">Named Entity Recognition</td>
    <td class="tg-0lax">Alvarado et al. 2015</td>
    <td class="tg-0lax">1,466</td>
  </tr>
  <tr>
    <td class="tg-0lax">FinSBD3</td>
    <td class="tg-0lax">Structure Boundary Detection</td>
    <td class="tg-0lax">FinSBD3 (FinWeb-2021)</td>
    <td class="tg-0lax">756</td>
  </tr>
  <tr>
    <td class="tg-0lax">FiQA QA</td>
    <td class="tg-0lax">Question Answering</td>
    <td class="tg-0lax">FiQA 2018</td>
    <td class="tg-0lax">6,640</td>
  </tr>
  <tr>
    <td class="tg-0lax">Volatility</td>
    <td class="tg-0lax">Volatility and Return Prediction</td>
    <td class="tg-0lax">Ours</td>
    <td class="tg-0lax">6,500</td>
  </tr>
</tbody>
</table>

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
