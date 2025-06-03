#needed imports for the Symbolic music generation
#from workbook3
import glob
import random
from typing import List
from collections import defaultdict

import numpy as np
from numpy.random import choice

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from symusic import Score
from miditok import REMI, TokenizerConfig

#added imports maybe nessary
from tensorflow import keras
from collections import defaultdict
from numpy.random import choice
from midiutil import MIDIFile

#random seed can be changed to get different results, default its 42
#from CSE_153R Homework 3(Spring 2025)
random.seed(42)

#Loading the music data
#first is popular pop songs I enjoy
midi_files = glob('data/*.mid')
#same midi file data We have trained on for Assigment 1 Task 1
#midi_files = glob('Assignment1(Task1_midis)/*.midi')
len(midi_files)

class MusicRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(MusicRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch_size, seq_length)
        x = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        out, hidden = self.rnn(x, hidden)  # out: (batch_size, seq_length, hidden_dim)
        out = self.fc(out)  # (batch_size, seq_length, vocab_size)
        return out, hidden
    
def train(model, train_loader, val_loader, vocab_size, num_epochs=20, lr=0.001, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        # --------- Training ---------
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            batch = batch['input_ids'].to(device)  # (batch_size, seq_length)

            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            outputs = outputs.reshape(-1, vocab_size)
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --------- Validation ---------
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch['input_ids'].to(device)

                inputs = batch[:, :-1]
                targets = batch[:, 1:]

                outputs, _ = model(inputs)
                outputs = outputs.reshape(-1, vocab_size)
                targets = targets.reshape(-1)

                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")


# Example usage
if __name__ == "__main__":
    vocab_size = tokenizer.vocab_size
    embedding_dim = 256
    hidden_dim = 512
    num_layers = 2

    model = MusicRNN(vocab_size, embedding_dim, hidden_dim, num_layers)
    train(model, train_loader, test_loader, vocab_size)


