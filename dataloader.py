import torch
from torch.utils.data import dataloader
from torchvision import transforms
from Flicker_dataset import Flicker_dataset
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import  pad_sequence
from torchvision.transforms import Normalize,RandomCrop,ToTensor,Resize
import os

transform=transforms.Compose([Resize((365,365)), 
RandomCrop((299,299)),
ToTensor(),Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
#transformation to apply on images


class Collate:#operations we are going to apply on the batch 
    def __init__(self,pad_idx):
        self.pad_idx=pad_idx
    def __call__(self,batch):
        
        images=[item[0].unsqueeze(0) for item in batch] 
        images=torch.cat(images,dim=0) #concat the images to get a single matrix as a batch 
                                       #shape=(B,3,299,299)
        targets=[item[1] for item in batch]
        targets=pad_sequence(targets,batch_first=False,padding_value=self.pad_idx)
        #shape=(len caption,B)
        return images,targets

        

def get_loader(root_dir,captions_file,transform,batch_size=4, num_workers=1,shuffle=True,pin_memory=True):
    """"returns the dataloder"""
    dataset=Flicker_dataset(root_dir,captions_file,transform)
    pad_id=dataset.vocabulary.string_to_index['<PAD>']
    dataloader=DataLoader(dataset=dataset,batch_size=batch_size,
    pin_memory=pin_memory,num_workers=num_workers,shuffle=shuffle,
    collate_fn=Collate(pad_idx=pad_id))
    return dataloader
"""root_dir=os.path.join(os.getcwd(),'data/Flicker8k_Dataset')
captions_file=os.path.join(os.getcwd(),"data/captions.txt")
loader=get_loader(root_dir=root_dir,captions_file=captions_file,transform=transform)
from tqdm import tqdm
for idx,(image,caption) in enumerate(tqdm(loader,total=len(loader),leave=False)):
    print(caption.shape)"""