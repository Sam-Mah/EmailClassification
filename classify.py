# NOTE: This is only a starter template. Wherever additional changes are required, please feel free modify/update.

import pickle

import pandas as pd
from nltk import PorterStemmer
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
import re
import matplotlib.pyplot as plt
import string
import json
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torchtext as tt
from sklearn.metrics import accuracy_score, f1_score, precision_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.optim as optim


# TODO: Feel free to improve the model
class EmailClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_class, num_layers, word_embeddings, bidirectional):
        # Constructor
        super(EmailClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding.weight = nn.Parameter(word_embeddings, requires_grad=False)
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                           # dropout=dropout,
                           batch_first=True
                            )
        self.fc = nn.Linear(hidden_size*2, num_class)
        # self.act = nn.Sigmoid()
        self.act = nn.Softmax(dim=1)

    def forward(self, text, text_lengths):
        embeddings = self.embedding(text)
        # packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embeddings, text_lengths, batch_first=True)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # Concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # hidden = [batch size, hid dim * num directions]
        dense_outputs = self.fc(hidden)

        # Final activation function
        out = self.act(dense_outputs)

        # Classic LSTM
        # embeddings = self.embedding(text)
        # hidden_out = self.lstm(embeddings)
        # dense_output = self.fc(hidden_out[0])
        # output = self.act(dense_output)
        # # output = output[:, -1]
        # out = torch.mean(output, 1)
        return out



# Step #0: Load data
def load_data(path: str) -> list:
    """Load Pickle files"""

    with open(path, 'rb') as f:
        data_list = pickle.load(f)
    return data_list

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Step #1: Analyse data
def analyse_data(data: list) -> None:
    """Analyse data files"""

    # Show the number of emails in each topic
    s = np.array([email['Label'] for email in data])
    unique, counts = np.unique(s, return_counts=True)
    class_dist = dict(np.column_stack((unique, counts)))

    # Visualize the class distribution using a pie chart
    fig = plt.figure(figsize=(5, 5))
    # labels for the four classes
    labels = 'sports', 'world', 'scitech', 'business'
    # Sizes for each slide
    sizes = [class_dist['sports'], class_dist['world'], class_dist['scitech'], class_dist['business']]
    # Declare pie chart, where the slices will be ordered and plotted counter-clockwise:
    plt.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.axis('equal')
    # Display the chart
    plt.show()
    return None

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Step #2: Define data fields
def data_fields() -> dict:
    # Type of fields on Data
    SUBJECT = tt.legacy.data.Field(sequential=True,
                                        batch_first=True,
                                     init_token='<sos>',
                                     eos_token='<eos>',
                                     lower=True,
                                     stop_words=stop_words,  # Remove English stop words
                                     tokenize=tt.legacy.data.utils.get_tokenizer("basic_english"))
    BODY = tt.legacy.data.Field(sequential=True,
                                     init_token='<sos>',
                                     eos_token='<eos>',
                                     lower=True,
                                     stop_words=stop_words,  # Remove English stop words
                                     tokenize=tt.legacy.data.utils.get_tokenizer("basic_english"))
    LABEL = tt.legacy.data.Field(sequential=False,
                                      use_vocab=False,
                                      unk_token=None,
                                      is_target=True)

    # fields = {'Subject': ('subject', SUBJECT), 'Body': ('body', BODY), 'Label': ('label', LABEL)}
    fields = [('subject', SUBJECT), ('body', BODY), ('label', LABEL)]

    return fields, SUBJECT, BODY, LABEL

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Step #2: Clean data
def data_clean(data: list, fields: dict) -> list:
    """A data cleaning routine."""

    clean_data = []
    for curr_data in data:
        # Remove hyperlinks
        curr_data["Subject"] = re.sub(r'https?://[^\s\n\r]+', '', curr_data["Subject"])
        curr_data["Body"] = re.sub(r'https?://[^\s\n\r]+', '', curr_data["Body"])

        # Remove punctuations
        curr_data["Subject"] = curr_data["Subject"].translate(
            str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        curr_data["Body"] = curr_data["Body"].translate(
            str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

        # Define a single training or test tokenized example
        # tokenized_data = tt.legacy.data.Example.fromJSON(json.dumps(curr_data), fields)
        tokenized_data = tt.legacy.data.Example.fromlist(list(curr_data.values()), fields)

        # Apply stemming on the emails' subjects and bodies
        stemmer = PorterStemmer()
        for i in range(len(tokenized_data.body)):
            tokenized_data.body[i] = stemmer.stem(tokenized_data.body[i])
        for i in range(len(tokenized_data.subject)):
            tokenized_data.subject[i] = stemmer.stem(tokenized_data.subject[i])

        # Remove empty data points
        if (len(tokenized_data.subject) != 0 or len(tokenized_data.body) != 0):
            clean_data.append(tokenized_data)
    return clean_data
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Step #2: Prepare data
def data_prepare(data: list, fields: dict, val_percent: int) -> list:
    """A data preparation routine."""

    clean_train, clean_val = tt.legacy.data.Dataset(data, fields).split(split_ratio=val_percent)

    return clean_train, clean_val

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Step #3: Extract features
def extract_features(X_train, X_valid, SUBJECT: tt.legacy.data.Field, BODY: tt.legacy.data.Field, LABEL: tt.legacy.data.Field,
                     batch_s) :
    train_iter, val_iter = [], []
    if X_train:
        #Initilize with glove embeddings
        SUBJECT.build_vocab(X_train, vectors="glove.6B.100d")
        BODY.build_vocab(X_train, vectors="glove.6B.100d")
        LABEL.build_vocab(X_train)
        train_iter = tt.legacy.data.BucketIterator(X_train, batch_size=batch_s, sort_key=lambda x: len(x.subject),
                                                   device=device, sort=True, sort_within_batch=True)

    if X_valid:
        val_iter = tt.legacy.data.BucketIterator(X_valid, batch_size=batch_s, sort_key=lambda x: len(x.subject),
                                                 device=device, sort=True, sort_within_batch=True)

    print(list(SUBJECT.vocab.stoi.items()))
    # No. of unique tokens in text
    print("Size of SUBJECT vocabulary:", len(SUBJECT.vocab))

    print("Size of BODY vocabulary:", len(BODY.vocab))

    # No. of unique tokens in label
    print("Size of LABEL vocabulary:", len(LABEL.vocab))

    # Commonly used words
    print("Commonly used words:", SUBJECT.vocab.freqs.most_common(10))

    # Word dictionary
    print(LABEL.vocab.stoi)

    return train_iter, val_iter, SUBJECT, BODY , LABEL
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# define accuracy metric
def binary_accuracy(preds, y):
    # round predictions to the closest integer
    if (torch.argmax(preds)==y):
        correct=1
    else:
        correct=0
    return correct
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def one_hot_vector(label, num_class):
        # Get the actual labels and return one-hot vectors
        st = np.zeros((num_class))
        st[label] = 1
        return st
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Step #4: Train model
def train_model(classification_model: EmailClassifier, BODY: tt.legacy.data.Field, LABEL: tt.legacy.data.Field, train_iter: tt.legacy.data.BucketIterator, optimizer,
                loss_func, num_class) :
    """Create a training loop"""
    # Initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # Set the model in training phase
    classification_model.train()
    train_iter.create_batches()
    for batch in train_iter.batches:
        # Resets the gradients after every batch
        optimizer.zero_grad()
        batch_loss = 0
        batch_acc = 0
        for data_point in batch:
            if (len(data_point.body) == 0):
                x = data_point.subject
            else:
                x = data_point.body
            # Convert to integer sequence
            indexed = [BODY.vocab.stoi[t] for t in x]
            # Compute no. of words
            length = [len(indexed)]
            # Convert to tensor
            tensor = torch.LongTensor(indexed).to(device)
            tensor = tensor.unsqueeze(1).T
            length_tensor = torch.LongTensor(length)
            y = LABEL.vocab.stoi[data_point.label]
            # Convert to 1d tensor
            predictions = classification_model(tensor, length_tensor).squeeze()
            y = torch.LongTensor([y])
            predictions = torch.reshape(predictions, (1, num_class))
            loss = loss_func(predictions, y)
            acc = binary_accuracy(predictions, y)
            # Backpropage the loss and compute the gradients
            loss.backward()
            # Update the weights
            optimizer.step()
            # Keep track of loss and accuracy of each batch
            batch_loss += loss.item()
            batch_acc += acc
        # keep track of loss and accuracy of each epoch
        epoch_loss += (batch_loss/len(batch))
        epoch_acc += (batch_acc/len(batch))

    return classification_model, epoch_loss / len(train_iter), epoch_acc / len(train_iter)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def evaluate_model(classification_model: EmailClassifier, BODY: tt.legacy.data.Field, LABEL: tt.legacy.data.Field, val_iter: tt.legacy.data.BucketIterator,
                   loss_func, num_class) :
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    classification_model.eval()
    val_iter.create_batches()
    # Deactivates autograd
    with torch.no_grad():
        for batch in val_iter.batches:
            batch_loss = 0
            batch_acc = 0
            for data_point in batch:
                if (len(data_point.body)==0):
                    x = data_point.subject
                else:
                    x = data_point.body
                # Convert to integer sequence
                indexed = [BODY.vocab.stoi[t] for t in x]
                # Compute no. of words
                length = [len(indexed)]
                # Convert to tensor
                tensor = torch.LongTensor(indexed).to(device)
                tensor = tensor.unsqueeze(1).T
                # Convert to tensor
                length_tensor = torch.LongTensor(length)
                y = LABEL.vocab.stoi[data_point.label]
                # Convert to 1d tensor
                predictions = classification_model(tensor, length_tensor).squeeze()
                y = torch.LongTensor([y])
                predictions = torch.reshape(predictions, (1, num_class))
                loss = loss_func(predictions, y)
                acc = binary_accuracy(predictions, y)
                # keep track of loss and accuracy of each batch
                batch_loss += loss.item()
                batch_acc += acc
            # keep track of loss and accuracy of each epoch
            epoch_loss += (batch_loss / len(batch))
            epoch_acc += (batch_acc / len(batch))

    return classification_model, epoch_loss / len(val_iter), epoch_acc / len(val_iter)

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Step #5: Stand-alone Test data & Compute metrics
def compute_metrics(classification_model: EmailClassifier, test_data: list, BODY: tt.legacy.data.Field, LABEL: tt.legacy.data.Field, num_class) -> None:
    test_iter = tt.legacy.data.BucketIterator(test_data, batch_size=len(test_data), sort_key=lambda x: len(x.subject),
                                               device=device, sort=True, sort_within_batch=True)

    classification_model.eval()
    test_iter.create_batches()
    predictions = []
    true_labels = []
    # For the whole test samples
    for sample in test_iter.batches:
        for data_point in sample:
            if (len(data_point.body) == 0):
                x = data_point.subject
            else:
                x = data_point.body
            # Convert to integer sequence
            indexed = [BODY.vocab.stoi[t] for t in x]
            # Compute no. of words
            length = [len(indexed)]
            # convert to tensor
            tensor = torch.LongTensor(indexed).to(device)
            tensor = tensor.unsqueeze(1).T
            # Convert to tensor
            length_tensor = torch.LongTensor(length)
            y = LABEL.vocab.stoi[data_point.label]
            y = torch.FloatTensor(one_hot_vector(y, num_class))
            true_labels.append(y)
            # Convert to 1d tensor
            prediction = classification_model(tensor, length_tensor).squeeze()
            predictions.append(prediction)

    # Compute Performance Metrics
    lbls = [torch.argmax(t) for t in true_labels]
    preds = [torch.argmax(t) for t in predictions]
    ACC = accuracy_score(lbls, preds)
    PR = precision_score(lbls, preds, average='weighted',  labels=np.unique(preds))
    F1 = f1_score(lbls, preds, average='weighted', labels=np.unique(preds))

    # Save metrics into a CSV file
    data_pd = [['Accuracy', ACC], ['Precision', PR], ['F1_Score', F1]]
    df = pd.DataFrame(data_pd, columns=['Measure', 'Percentage'])
    np.savetxt('./Metric_Values_Test.csv', df, delimiter=',', fmt='%s')

    return None

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def main(train_path: str, test_path: str) -> None:
    # Define hyperparameters
    embedding_dim = 100
    num_hidden_nodes = 25
    num_classes = 4
    n_layers = 1
    bidirection = True
    # dropout = 0.2
    N_EPOCHS = 10
    # LOSS_THRESH = 0.001
    batch_size = 25
    # l_rate = [0.001, 0.005, 0.01, 0.1]
    l_rate = 0.001

    ### Perform the following steps and complete the code

    ### Step #0: Load data
    train_data = load_data(train_path)

    ### Step #1: Analyse data
    analyse_data(train_data)

    ### Step #2: Clean and prepare data
    fields, SUBJECT, BODY, LABEL = data_fields()
    train_data = data_clean(train_data, fields)

    train_ds, val_ds = data_prepare(train_data, fields, val_percent=0.5)

    ### Step #3: Extract features
    train_iter, val_iter, SUBJECT, BODY, LABEL = extract_features(train_ds, val_ds, SUBJECT, BODY, LABEL, batch_size)
    word_embeds = BODY.vocab.vectors
    vocab_size = len(BODY.vocab.stoi)
    ### Step #4: Train model

    # Initilize the model
    classification_model = EmailClassifier(vocab_size=vocab_size, embed_size=embedding_dim,
                                           hidden_size=num_hidden_nodes, num_class=num_classes,
                                           num_layers=n_layers,word_embeddings=word_embeds, bidirectional=bidirection)

    # Define optimizer and loss function
    optimizer = optim.Adam(classification_model.parameters(), lr=l_rate)
    # loss_func = nn.BCELoss()
    loss_func = nn.CrossEntropyLoss()

    best_valid_loss = float('inf')

    # epoch = 0
    # old_loss = -1
    # new_loss = 0
    # Train until the maximum epochs is reached or validation loss of two consecutive epochs is less than the loss threshold
    # while (epoch < N_EPOCHS and abs(new_loss-old_loss) >= LOSS_THRESH) :
    for epoch in range(N_EPOCHS):
        # train the model
        classification_model, train_loss, train_acc = train_model(classification_model, BODY, LABEL, train_iter, optimizer,
                                                                  loss_func, num_classes)
        # Evaluate the model
        classification_model, valid_loss, valid_acc = evaluate_model(classification_model,  BODY, LABEL, val_iter, loss_func, num_classes)

        # Save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(classification_model.state_dict(), 'saved_weights.pt')

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

        # old_loss = new_loss
        # new_loss = valid_loss
        # epoch+=1

    ### Step #5: Stand-alone Test data & Compute metrics
    test_data = load_data(test_path)
    analyse_data(test_data)
    test_data = data_clean(test_data, fields)
    compute_metrics(classification_model, test_data, BODY, LABEL, num_classes)

    return 0
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
if __name__ == "__main__":
    train_path = "./agnews_combined_train.pkl"
    test_path = "./agnews_combined_train.pkl"
    main(train_path, test_path)
