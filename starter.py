import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import math
import time
import numpy as np
import sys
import argparse
import os
import pandas as pd
import re
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
from gensim import corpora
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json

torch.manual_seed(497)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def create_dp(random=True):
    ret_df = pd.DataFrame(columns=['Text', 'Class', 'Train'])
    directory = 'data'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        file_arr = filename.split('.')
        if os.path.isfile(f) and len(file_arr) > 1 and file_arr[0] != "blind":
            train = 1 if file_arr[1] == "train" else 0
            file = open(directory+"/"+filename,mode='r',encoding="utf8")
            content = file.read()
            text = re.findall(r'<start_bio>((?s)((?:[^\n][\n]?)+))<end_bio>', content)
            text = [line[0].replace('\n','') for line in text]
            if file_arr[0] == "mix":
                types = re.findall(r'<end_bio>([\s\S]*?)\]',content)
                types = [i.replace('\n','') for i in types]
                types = [i.replace(' ','') for i in types]
                types = [i+"]" for i in types]
                classes = [1 if i == "[REAL]" else 0 for i in types]
            else:
                classes = [1]*len(text) if file_arr[0] == "real" else [0]*len(text)
            train = [train]*len(text)
            tmp_df = pd.DataFrame({'Text':text,'Class':classes, 'Train':train})
            ret_df = pd.concat([ret_df, tmp_df], ignore_index = True)
            ret_df.reset_index()
            file.close()
        elif os.path.isfile(f) and len(file_arr) > 1 and file_arr[0] == "blind":
            file = open(directory+"/"+filename,mode='r',encoding="utf8")
            content = file.read()
            text = re.findall(r'<start_bio>((?s)((?:[^\n][\n]?)+))<end_bio>', content)
            text = [line[0].replace('\n','') for line in text]
            classes = [-1]*len(text)
            train = [-1]*len(text)
            tmp_df = pd.DataFrame({'Text':text,'Class':classes, 'Train':train})
            ret_df = pd.concat([ret_df, tmp_df], ignore_index = True)
            ret_df.reset_index()
            file.close()
    tokens = [simple_preprocess(line, deacc=True) for line in ret_df['Text']]
    porter_stemmer = PorterStemmer()
    ret_df['processed_tokens'] = [[porter_stemmer.stem(word) for word in token_arr] for token_arr in tokens]
    if random:
        ret_df = ret_df.sample(frac=1,random_state=497).reset_index(drop=True)
    return ret_df

class FFNN(nn.Module):
    def __init__(self, in_dim,hid_dim,out_dim, number_of_hidden_layers):
        super().__init__() 
        dims_in = [in_dim] + [hid_dim] * number_of_hidden_layers
        dims_out = [hid_dim] * number_of_hidden_layers + [out_dim]
        layers = []
        for i in range(number_of_hidden_layers + 1):
            layers.append(torch.nn.Linear(dims_in[i], dims_out[i]))
            if i < number_of_hidden_layers:
                layers.append(nn.Tanh())
        self.net = torch.nn.Sequential(*layers) 

    def forward(self, x):
        return F.softmax(self.net(x), dim=1)


class LSTM(nn.Module):
    def __init__(self,dimension):
        super().__init__()
        self.embedding = nn.Embedding(len(text_field.vocab), 300)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300, hidden_size=dimension, num_layers=3, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2*dimension, 1)
        
    def forward(self,text,text_len):
        text_emb = self.embedding(text)
        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)
        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)
        return text_out
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_model', type=int, default=100)
    parser.add_argument('-d_hidden', type=int, default=100)
    parser.add_argument('-n_layers', type=int, default=2)
    parser.add_argument('-batch_size', type=int, default=20)
    parser.add_argument('-seq_len', type=int, default=30)
    parser.add_argument('-printevery', type=int, default=5000)
    parser.add_argument('-window', type=int, default=3)
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-dropout', type=int, default=0.35)
    parser.add_argument('-clip', type=int, default=2.0)
    parser.add_argument('-model', type=str,default='FFNN_CLASSIFY')
    
    params = parser.parse_args()    
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if params.model == 'FFNN':
        dataset_df = create_dp()
        review_dict = corpora.Dictionary(dataset_df['processed_tokens'])
        model = FFNN(len(review_dict), 500, 2, 4)
        model.to(device)
        bce_loss = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        train_df = dataset_df[dataset_df["Train"] == 1]
        losses = []
        perps = []
        for epoch in range(40):
            tmp_loss = 0
            for _, X in train_df.iterrows():
                optimizer.zero_grad()
                input_vector = torch.zeros(len(review_dict),device=device)
                for token in X['processed_tokens']:
                    input_vector[review_dict.token2id[token]] += 1
                input_vector = input_vector.view(1, -1).float()
                probs = model(input_vector)
                target = torch.tensor([X['Class']],device=device)
                loss = bce_loss(probs, target)
                tmp_loss += loss.item()
                loss.backward()
                optimizer.step()
            print("Epoch completed: " + str(epoch+1))
            print("Loss: " + str(tmp_loss/len(train_df)))
            losses.append(tmp_loss / len(train_df))
            tmp_loss = 0
            # torch.save(model.state_dict(), "ffnn.t1")
        torch.save(model.state_dict(), "ffnn.t1")

    if params.model == 'LSTM':
        return
#          {add code to instantiate the model, train for K epochs and save model to disk}

    if params.model == 'FFNN_CLASSIFY':
        dataset_df = create_dp()
        review_dict = corpora.Dictionary(dataset_df['processed_tokens'])
        valid_df = dataset_df[dataset_df["Train"] == 0]
        model = FFNN(len(review_dict), 500, 2, 4)
        model.load_state_dict(torch.load("ffnn.t1"))
        model.to(device)
        preds = []
        y_real = []
        with torch.no_grad():
            for _, X in valid_df.iterrows():
                input_vector = torch.zeros(len(review_dict),device=device)
                for token in X['processed_tokens']:
                    input_vector[review_dict.token2id[token]] += 1
                input_vector = input_vector.view(1, -1).float()
                probs = model(input_vector)
                preds.append(torch.argmax(probs, dim=1).cpu().numpy()[0])
                y_real.append(torch.tensor([X['Class']]).cpu().numpy()[0])
        print(classification_report(y_real,preds))
        print(accuracy_score(y_real, preds))
        print(confusion_matrix(y_real,preds))

    if params.model == 'LSTM_CLASSIFY':
        return
#          {add code to instantiate the model, recall model parameters and perform/learn classification}
        
    print(params)
    

if __name__ == "__main__":
    main()