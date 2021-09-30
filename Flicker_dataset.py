from Vocabulary import Vocabulary
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pandas as pd
import torch
import pickle

""" in Flicker_dataset we are going to make the data callable so we 
    can get the image tensor transformed and get the numericalzed 
    version of the  text """
class Flicker_dataset(Dataset):
    def __init__(self,root_dir,captions_file,transform=None,freq_threshold=5,build_vocab=False):
        self.root_dir=root_dir  #the directory of the images 
        self.freq_threshold=freq_threshold
        self.df=pd.read_csv(captions_file)
        self.transform=transform #image transformers
        self.captions=self.df.caption
        self.image_id=self.df.image
        if build_vocab: #if we build the vocab for the first time 
            self.vocabulary=Vocabulary(self.freq_threshold)
            self.vocabulary.build_vocabulary(self.captions.to_list())
        else: #if we load the vocab file 
            with open('vocab.pkl','rb') as vocab:
                self.vocabulary=pickle.load(vocab)
    def __len__(self):#length of the data 
        return(self.df.shape[0])
    def __getitem__(self, index) :#get the transformed image and the numericalized text

        caption=self.captions[index]
        image_id=self.image_id[index]
        image=Image.open(os.path.join(self.root_dir,str(image_id))).convert("RGB")
        if self.transform is not None:
            image=self.transform(image)
        numericalized_caption=[self.vocabulary.string_to_index["<SOS>"]]
        numericalized_caption+=self.vocabulary.numericalize(caption)
        numericalized_caption+=[self.vocabulary.string_to_index["<EOS>"]]
        return  (image,torch.tensor(numericalized_caption))
