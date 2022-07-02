The related preprocessed datasets and trained models can be found at: 

https://drive.google.com/drive/folders/15aQejnRjGDa5OEmzegyd6iG0Nv9Nur3z?usp=sharing

Find the link below the purpose of this project:

https://www.kaggle.com/competitions/feedback-prize-2021

Below are explanations of each script:

### utils.py
Helper classes and methods for data preprocessing & modelling.

### data_preprocessing.py
Download the original dataset "feedback-prize-2021" (which contains the raw data) from
https://www.kaggle.com/competitions/feedback-prize-2021
and save it to the project folder. Then, after running this script, a pickle file of about 1.55 GB, which 
contains the input of the Longformer model will be generated in the "model" folder. The script converts 
the raw data to a list of dictionaries, and each dictionary has three attributes: "input_ids", "attention_mask",
and "labels". You can also set the tokenizer in the Config class to be the roberta tokenizer, to get input of 
the roberta model (which will be much smaller, about 200 MB). 

### Training_With_RoBERTa.ipynb
Records how we used the RoBERTa model for training. 
The pickle file produced by data_preprocessing.py was further split into train, dev, and test set, and is saved in the google drive folder above.
Data used in this process: from the link above, you can find three .pickle files, "roberta_train_set.pickle", 
"roberta_dev.pickle", "roberta_test.pickle". They are pre-processed train, dev, and test sets used in this notebook.
The three files above are generated using the pickle file created by "data_preprocessing.py".
The model is evaluated on token-level F1 (not mention-level), and the test set F1 is 0.777. The final model is 
also in the link above (named "writing_model_roberta.pt").

### Training_With_Longformer.ipynb
Records how we used the Longformer model for training. Since Longformer is our final solution, this notebook also writes an inference method,
and records both token-level F1 and mention-level F1.
Data used in this process: from the google drive link above, you can find three .pickle files, "longformer_train_set.pickle", 
"longformer_dev.pickle", "longformer_test.pickle". They are pre-processed train, dev, and test sets used in this notebook.
The three files above are also generated using the pickle file created by "data_preprocessing.py".
The model is evaluated on both token-level F1 and mention-level F1. The test set token-level F1 is 0.795. For calculating
the mention(sentence)-level F1 easier, we used a different version of the test set (also attached in the link above, 
named "longformer_original_test_set.pkl"). The test set mention-level F1 is 0.783. The final model is also in the 
link above (named "writing_model_longformer.pt").
