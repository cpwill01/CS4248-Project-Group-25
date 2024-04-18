# Classification Models for News Reliability Prediction

Code for CS4248 Group Project (AY23/24 Sem 2) - Group 25

By A0223917A, A0275756M, A0275611H, A0216840J, A0275647N

## Abstract

Fake news has become a pressing issue due to the dangers it poses in areas such as politics and public health. Thus, fake news detection has become increasingly important. In language-based fake news detection, the attention mechanism has been used in several models achieving state-of-the-art performance. Hence, this project explores 3 attention-based models for fake news detection, specifically as a 4-way classification problem between Satire, Hoax, Propaganda and Real (Trusted) news. We applied these models on the Labeled Unreliable News (LUN) dataset and compared their performances.  We found that the attention mechanism is effective for fake news detection and that hierarchical structures can further improve them. However, underperformance in Health news articles suggests that additional training data or a more specialised model may be suitable for detection in this domain.
## Folder Structure

```
|-EDA+FE+Baselines/
|-HAN_and_FAN/
|-DistilBERT/
```

- `EDA+FE+Baselines/` contains 
  - `eda.ipynb` for our exploratory data analysis 
  - `features_base.ipynb` for our feature engineering and baseline model
- `HAN_and_FAN/` contains 
  - `training_and_eval.ipynb` for training & evaluation of the HAN and FAN models 
  - `analysis.ipynb` for analysis of subcategory performance and attention weights corresponding to our report's discussion section
- `DistilBERT/` contains
  - `` training & evaluation of our DistilBERT classifier
  - `` for labelling the LUN dataset with news categories

> For clarifications on the models, please refer to our report at [CS4248_Project_Group_25_Final_Report.pdf](https://github.com/cpwill01/CS4248-Project-Group-25/tree/main/CS4248_Project_Group_25_Final_Report.pdf)
## Dataset & Additional files

Due to GitHub file size limits, we are unable to upload all files used in our project. Please downloaded the necessary files from the below links.

The original dataset by Rashkin et al. (2017) can be downloaded from [this link](https://hrashkin.github.io/factcheck.html). Only the  **Labeled Unreliable News (LUN) dataset** is used in our project.

The `.txt` file for GloVe embeddings used in HAN and FAN can be downloaded from [here](https://nlp.stanford.edu/data/glove.6B.zip)

All additional files can be downloaded from [this google drive link](https://drive.google.com/drive/folders/1ctc_15-p7vZtnwIbbD39z_UWF2a5M-4T?usp=sharing). The following type of files are included:
- pre-trained models for HAN, FAN and DistilBERT that produced the results shown in our report
- the LUN dataset labelled with news categories using [this pre-trained DistilBART classifier](https://huggingface.co/IT-community/distilBART_cnn_news_text_classification)
- preprocessed inputs for HAN and FAN respectively
- preprocessed glove embeddings `glove_embds.npy`

## Running Instructions

Before running the notebooks, please ensure the following:

1. Download the necessary files from the previous section and place them in a local directory. The original dataset must be downloaded for **all notebooks**. For specific notebooks,
   - HAN and FAN:
        - To run all steps from scratch, download the `.txt` file for GloVe embeddngs.
        - To run training on preprocessed inputs, download preprocessed inputs for the model **and** `glove_embs.py`
        - To run evaluation using pre-trained model, download the preprocessed inputs **and** the pretrained model (filename starting with `bestHan`/`bestFAN`)
        - To run analysis using pre-trained model, download the augmented dataset, the `.txt` file for GloVe embeddings, **and** the pretrained model
   - DistilBERT:
        - Download the augmented dataset and both files from the `DistilBERT_pretrained` folder
2. Edit the file paths in the notebook accordingly.

Also note that the following python packages are required and may need to be installed separately:
```
scikit-learn
pytorch
nltk
numpy
pandas
matplotlib
seaborn
spacy
transformers
```

If you encounter any difficulties, please raise an [Issue](https://github.com/cpwill01/CS4248-Project-Group-25/issues). 
