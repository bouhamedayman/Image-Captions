import torch 
import os
import pandas as pd 
import spacy
import torch.nn as nn 
from torch.utils.data import dataloader , dataset
from torch.nn.utils.rnn import  pad_sequence
from PIL import Image
spacy_eng=spacy.load("en")
from torchvision import transforms
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm

