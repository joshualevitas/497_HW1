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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import json
import matplotlib.pyplot as plt
import csv

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
    def __init__(self,in_dim,hid_dim,number_of_hidden_layers, batch_size):
        super().__init__()
        self.embedding = nn.Embedding(in_dim, 300)
        self.hid_dim = hid_dim
        self.lstm = nn.LSTM(input_size=300, hidden_size=self.hid_dim, num_layers=number_of_hidden_layers, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p=0.35)
        self.fc = nn.Linear(2*self.hid_dim, batch_size)

    def forward(self,x):
        x_len = torch.tensor([len(x)],dtype=torch.int64)
        x = pack_padded_sequence(self.embedding(x), x_len, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = torch.cat((x[range(len(x)),x_len - 1,:self.hid_dim], x[:,0,self.hid_dim:]), 1)
        return torch.sigmoid(torch.squeeze(self.fc(self.drop(x)), 1))
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str,default='OUTPUT')
    
    params = parser.parse_args()    
    torch.manual_seed(0)
    device = torch.device("cuda")
    
    if params.model == 'FFNN':
        dataset_df = create_dp()
        review_dict = corpora.Dictionary(dataset_df['processed_tokens'])
        model = FFNN(len(review_dict), 128, 2, 4).to(device)
        bce_loss = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        train_df = dataset_df[dataset_df["Train"] == 1]
        valid_df = dataset_df[dataset_df["Train"] == 0]
        train_perps = []
        valid_perps = []
        batch_size = 20
        for epoch in range(40):
            tmp_train_perplexity = 0
            tmp_valid_perplexity = 0
            tmp_train_loss = 0
            tmp_valid_loss = 0
            model.train()
            for i in range(0,len(train_df),batch_size):
                entries = train_df[i:i+batch_size]
                optimizer.zero_grad()
                input_vector = torch.zeros(len(review_dict)*batch_size,device=device)
                idx = 0
                targets = []
                for _,X in entries.iterrows():
                    for token in X['processed_tokens']:
                        input_vector[idx*len(review_dict)+review_dict.token2id[token]] += 1
                    targets.append(X["Class"])
                    idx += 1
                if len(targets) < batch_size:
                    input_vector = input_vector[:len(targets)*len(review_dict)]
                    input_vector = input_vector.view(len(targets), len(review_dict)).float()
                else:
                    input_vector = input_vector.view(batch_size, len(review_dict)).float()
                probs = model(input_vector)
                target = torch.tensor(targets,device=device)
                loss = bce_loss(probs, target)
                tmp_train_loss += loss.item()
                tmp_train_perplexity += torch.exp(loss).item()
                loss.backward()
                optimizer.step()

            model.eval()
            preds = []
            y_real = []
            with torch.no_grad():
                for i in range(0,len(valid_df),batch_size):
                    entries = valid_df[i:i+batch_size]
                    input_vector = torch.zeros(len(review_dict)*batch_size,device=device)
                    idx = 0
                    targets = []
                    for _,X in entries.iterrows():
                        for token in X['processed_tokens']:
                            input_vector[idx*len(review_dict)+review_dict.token2id[token]] += 1
                        targets.append(X["Class"])
                        idx+=1
                    if len(targets) < batch_size:
                        input_vector = input_vector[:len(targets)*len(review_dict)]
                        input_vector = input_vector.view(len(targets), len(review_dict)).float()
                    else:
                        input_vector = input_vector.view(batch_size, len(review_dict)).float()
                    probs = model(input_vector)
                    target = torch.tensor(targets,device=device)
                    valid_loss = bce_loss(probs, target)
                    tmp_valid_loss += valid_loss.item()
                    tmp_valid_perplexity += torch.exp(valid_loss).item()
                    preds += [torch.argmax(prob).cpu().numpy().item() for prob in probs]
                    y_real += [torch.tensor(target).cpu().numpy().item() for target in targets]

            print("Epoch completed: " + str(epoch+1))
            print("Training Perplexity: "+ str(tmp_train_perplexity*batch_size/len(train_df)))
            print("Validation Perplexity: "+ str(tmp_valid_perplexity*batch_size/len(valid_df)))
            print("Training Loss: "+ str(tmp_train_loss*batch_size/len(train_df)))
            print("Validation Loss: "+ str(tmp_valid_loss*batch_size/len(valid_df)))
            print("----------------------------------")
            train_perps.append(tmp_train_perplexity * batch_size/ len(train_df))
            valid_perps.append(tmp_valid_perplexity * batch_size/ len(valid_df))
            torch.save(model.state_dict(), "ffnn.t1")
        torch.save(model.state_dict(), "ffnn.t1")
        print(classification_report(y_real,preds))
        print(accuracy_score(y_real, preds))
        print(confusion_matrix(y_real,preds))
        plt.figure()
        plt.plot(range(len(train_perps)),train_perps,'r-',label='Train')
        plt.plot(range(len(valid_perps)),valid_perps,'b-',label='Valid')
        plt.xlabel("Epoch")
        plt.ylabel("Perplexity")
        plt.legend(loc="upper left")
        plt.title("Learning Curve for FFNN")
        plt.savefig("ffnn_plot.png")

    if params.model == 'LSTM':
        dataset_df = create_dp()
        review_dict = corpora.Dictionary(dataset_df['processed_tokens'])
        vocab_size = len(review_dict)
        batch_size = 50
        model = LSTM(vocab_size, 128, 2, batch_size).to(device)
        bce_loss = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        train_df = dataset_df[dataset_df["Train"] == 1]
        valid_df = dataset_df[dataset_df["Train"] == 0]
        train_perps = []
        valid_perps = []
        for epoch in range(20):
            tmp_train_perplexity = 0
            tmp_valid_perplexity = 0
            tmp_train_loss = 0
            tmp_valid_loss = 0
            model.train()
            for i in range(0,len(train_df),batch_size):
                entries = train_df[i:i+batch_size]
                max_len = 0
                for _,X in entries.iterrows():
                    max_len = max(len(X["processed_tokens"]),max_len)
                optimizer.zero_grad()
                arr = []
                targets = []
                for _,X in entries.iterrows():
                    tmp_arr = []
                    if X["processed_tokens"] == []:
                        tmp_arr = [review_dict.token2id[token] for token in ["of"]]
                    else:
                        tmp_arr = [review_dict.token2id[token] for token in X["processed_tokens"]]
                    while len(tmp_arr) < max_len:
                        tmp_arr.append(0)
                    arr.append(tmp_arr)
                    targets.append(X['Class'])
                if len(targets) < batch_size:
                    break
                inp_text = torch.tensor(arr,dtype=torch.int64).to(device).view(batch_size, max_len)
                probs = model(inp_text)
                target = torch.tensor(targets,device=device)
                loss = bce_loss(probs[0].float(), target.float())
                tmp_train_loss += loss.item()
                tmp_train_perplexity += torch.exp(loss).item()
                loss.backward()
                optimizer.step()

            model.eval()
            preds = []
            y_real = []
            with torch.no_grad():
                for i in range(0,len(valid_df),batch_size):
                    entries = valid_df[i:i+batch_size]
                    max_len = 0
                    for _,X in entries.iterrows():
                        max_len = max(len(X["processed_tokens"]),max_len)
                    arr = []
                    targets = []
                    for _,X in entries.iterrows():
                        tmp_arr = []
                        if X["processed_tokens"] == []:
                            tmp_arr = [review_dict.token2id[token] for token in ["of"]]
                        else:
                            tmp_arr = [review_dict.token2id[token] for token in X["processed_tokens"]]
                        while len(tmp_arr) < max_len:
                            tmp_arr.append(0)
                        arr.append(tmp_arr)
                        targets.append(X['Class'])
                    if len(targets) < batch_size:
                        break
                    y_real += targets
                    inp_text = torch.tensor(arr,dtype=torch.int64).to(device).view(batch_size, max_len)
                    probs = model(inp_text)
                    target = torch.tensor(targets,device=device)
                    valid_loss = bce_loss(probs[0].float(), target.float())
                    tmp_valid_loss += valid_loss.item()
                    tmp_valid_perplexity += torch.exp(valid_loss).item()
                    preds += [torch.argmax(prob).cpu().numpy().item() for prob in probs[0]]

            print("Epoch completed: " + str(epoch+1))
            print("Training Perplexity: "+ str(tmp_train_perplexity*batch_size/len(train_df)))
            print("Validation Perplexity: "+ str(tmp_valid_perplexity*batch_size/len(valid_df)))
            print("Training Loss: "+ str(tmp_train_loss*batch_size/len(train_df)))
            print("Validation Loss: "+ str(tmp_valid_loss*batch_size/len(valid_df)))
            print("----------------------------------")
            train_perps.append(tmp_train_perplexity * batch_size/ len(train_df))
            valid_perps.append(tmp_valid_perplexity * batch_size/ len(valid_df))
            torch.save(model.state_dict(), "lstm.t1")
        torch.save(model.state_dict(), "lstm.t1")
        print(classification_report(y_real,preds))
        print(accuracy_score(y_real, preds))
        print(confusion_matrix(y_real,preds))
        plt.figure()
        plt.plot(range(len(train_perps)),train_perps,'r-',label='Train')
        plt.plot(range(len(valid_perps)),valid_perps,'b-',label='Valid')
        plt.xlabel("Epoch")
        plt.ylabel("Perplexity")
        plt.legend(loc="upper left")
        plt.title("Learning Curve for LSTM")
        plt.savefig("lstm_plot.png")

    def ffnn_classify():
        dataset_df = create_dp()
        review_dict = corpora.Dictionary(dataset_df['processed_tokens'])
        valid_df = dataset_df[dataset_df["Train"] == 0]
        model = FFNN(len(review_dict), 128, 2, 4)
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

    def lstm_classify():
        dataset_df = create_dp()
        review_dict = corpora.Dictionary(dataset_df['processed_tokens'])
        vocab_size = len(review_dict)
        valid_df = dataset_df[dataset_df["Train"] == 0]
        model = LSTM(vocab_size,128,3,1)
        model.load_state_dict(torch.load("lstm.t1"))
        model.to(device)
        preds = []
        y_real = []
        with torch.no_grad():
            for _, X in valid_df.iterrows():
                if X["processed_tokens"] == []:
                    arr = [review_dict.token2id[token] for token in ["of"]]
                else:
                    arr = [review_dict.token2id[token] for token in X["processed_tokens"]]
                inp_text = torch.tensor([arr],dtype=torch.int64).to(device)
                pred = (model(inp_text) > 0.5).int()
                preds.append(torch.tensor([pred]).cpu().numpy()[0])
                y_real.append(torch.tensor([X['Class']]).cpu().numpy()[0])
        print(classification_report(y_real,preds))
        print(accuracy_score(y_real, preds))
        print(confusion_matrix(y_real,preds))

    def best_classify():
        dataset_df = create_dp(random=False)
        review_dict = corpora.Dictionary(dataset_df['processed_tokens'])
        test_df = dataset_df[dataset_df["Train"] == -1]
        model = FFNN(len(review_dict), 128, 2, 4)
        model.load_state_dict(torch.load("ffnn.t1"))
        model.to(device)
        preds = []
        with torch.no_grad():
            for _, X in test_df.iterrows():
                input_vector = torch.zeros(len(review_dict),device=device)
                for token in X['processed_tokens']:
                    input_vector[review_dict.token2id[token]] += 1
                input_vector = input_vector.view(1, -1).float()
                probs = model(input_vector)
                preds.append(torch.argmax(probs, dim=1).cpu().numpy()[0])
            preds = ["[REAL]" if i == 1 else "[FAKE]" for i in preds]
        preds = np.array(preds)
        with open('predictions.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for row in range(0,preds.shape[0]):
                myList = []
                myList.append(preds[row])
                writer.writerow(myList)

    print("RESULTS FOR FFNN:")
    ffnn_classify()
    print("-------------------------")
    print("RESULTS FOR LSTM:")
    lstm_classify()
    print("-------------------------")
    best_classify()
    print("RESULTS FOR OUR BEST PERFORMING MODEL HAS BEEN SAVED AT PREDICTIONS.CSV")
    print("-------------------------")

if __name__ == "__main__":
    main()