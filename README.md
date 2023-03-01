# 497_HW1

Link to saved FFNN model [here](https://drive.google.com/file/d/1JhyJOzdX5Iw_x4NpYDUdtJYaJtVd-Wsx/view?usp=sharing).


Link to saved LSTM model [here](https://drive.google.com/file/d/1W_LZTNmZ5NIhET-sHn-NfqXbMZDkz8Rb/view?usp=sharing).

Place these files in the source directory of the project code to run.


# Tasks

## 1. FFNN

[Please describe how you organized and used the datasets provided, the operation of your FFNN and hyper-parameter settings. Present the learning curves for training and test perplexity, and report Fakes Detection results in a confusion matrix with labeled axes. Please discuss any limitations of your approach and possible solutions. Please do not exceed one page for this.]

### Setup

We organized the datasets into subsets of training sets and validation / testing. We split each corpus using the "<start_bio>...<end_bio>" tags, considering everything in between as a standalone biography. We then extracted the [Real] or [Fake] label and attributed that label to the biography. We stored this information in a pandas dataframe containing 3 columns: *Text*, *Class*, and *Train (type)*. The first contains the biography text, the second contains the label, and the third contains a boolean value indicating whether the biography is part of the training set or not. We then used the *Text* and *Class* columns to train the FFNN and the *Train (type)* column to split the data into training and validation / testing sets. We used the *Text* column to train the FFNN.

### Model Parameters

Namespace(batch_size=20, clip=2.0, d_hidden=100, d_model=100, dropout=0.35, epochs=20, lr=0.001, model='FFNN_CLASSIFY', n_layers=2, printevery=5000, seq_len=30, window=3)


### Results

![Confusion Matrix](https://github.com/joshualevitas/497_HW1/blob/main/figures/Figure_1.png?raw=true)
![Perplexity](https://github.com/joshualevitas/497_HW1/blob/main/figures/Perplexity_FFNN.JPG?raw=true)


## 2. LSTM

[Please describe how you organized and used the datasets provided, the operation of your LSTM LM and hyper-parameter settings. Present the learning curves for training and test perplexity, and report Fakes Detection results in a confusion matrix with labeled axes. Please discuss any limitations of your approach and possible solutions. Please do not exceed one page for this.]


## 3. Results

[Results (10 pts): Use your best model to perform Fakes Detection on the Blind dataset, which consists of 500 unlabeled observations. You may also construct you own non-neural model. A simple rules- based approach may be effective -- working with incongruent dates is one approach. Please provide a.csvfilecontainingyour[REAL]or[FAKE]labelsforeachobservation. Iplantosharesummary results for each group with the class. Discuss how you selected your model, any limitations and possible solutions. Please do not exceed 1⁄2 of a page for this.]