1. Building vocab based on your corpus  
bert-vocab -c data/corpus.small -o data/vocab.small  

**pay special attention to the corpus.small file**  
**especially the '\t' and '\n' characters**  
**read issue: https://github.com/codertimo/BERT-pytorch/issues/52**  

2. Train your own BERT model  
bert -c data/corpus.small -v data/vocab.small -o output/bert.model  
