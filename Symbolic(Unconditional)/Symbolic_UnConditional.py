#needed imports for the Symbolic music generation
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import random
from glob import glob
from collections import defaultdict
from numpy.random import choice
from symusic import Score
from miditok import REMI, TokenizerConfig
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



