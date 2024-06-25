# CDL: Inflation Narrative

### 1 Scope 
The scope of this project is to extract information from media text (Dow Jones Newswires, Handelsblatt etc.),
perhaps social media text and call transcripts. Information to extract concerns the following.
1. Does the given article mention the cause of inflation?
2. if yes, what are the causes?

Before extracting information from the data, we need to filter out non-relevant articles by keywords and tags (if they exist).

### 2 Filtering
1. keywords
   1. en: {"inflation", "prices"}
   2. de: {}
2. tags
   1. DJN: {"N/DJIB", "N/DJG", "N/GPRW", "N/DJAN", "AWSJ", "WSJE", "N/PREL", "N/NRG", "N/DJBN", "N/AWP", "N/BRNS", "N/JNL", "N/WAL", "N/WLS", "N/WSJ"}
   2. FAZ: {}
   3. SZ: {}
   4. Handelsblatt: {}


### 3 Tasks
#### Task 1. Does the given article mention the cause of inflation?
- [ ] Binary Classification
  - [ ] Zero-shot prompting
  - [ ] Few-shot prompting
  - [ ] tba

Resulting articles mentioning the cause of inflation are considered to be a filtered set of articles that will go to the second task. 

#### Task 2. what are the causes?
- [ ] Multiclass Classification following schema defined by Andre et al. 
  - [ ] Zero-shot prompting
  - [ ] Few-shot prompting
  - [ ] tba

#### Sentiment, Tones, and other narrative aspects extraction


### TODOs
1. Dataset creation
   1. Target data
      1. Methodology validation: Dow Jones Newswires (en) <-> Andre et al. 
      2. Generalization: FAZ, SZ, Handelsblatt (de) <-> Ontology from Uli and Max survey
   2. Filtering
      1. Regex (inflation, price**s**)
      2. Article tags (quite aggressive, but do not know the inner mechanism. DJN has it, not sure about FAZ, SZ and Handelsblatt)
   3. Annotation
      1. 500 news articles for each dataset
      2. For DJN, we follow Andre et al., ontology to build DAG 
      3. For German dataset, we need to wait unutil the survey is done. 
      4. Additionally, we want the direction of inflation and factors to be annotated
2. Tasks
   1. Has causes?
   2. What causes?
   3. Sentiment analysis (document level) -> should be generalizable from public corpora 
