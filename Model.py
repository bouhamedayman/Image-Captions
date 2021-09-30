import torch.optim as optim
import pytorch_lightning as pl
import torch
import torch.nn as nn
import  os
import torchvision.models as models
from Dataloader import get_loader,transform
class Encoder(nn.Module):
    def __init__(self,embed_size,train_cnn=False):
        super(Encoder,self).__init__()
        self.embed_size=embed_size #the size of the embedding vector
        self.train_cnn=train_cnn # if we will train the cnn network or not 
        self.model=models.inception.inception_v3(pretrained=True,aux_logits=False)
        self.model.fc=nn.Linear(self.model.fc.in_features,out_features=embed_size)#last layer size 
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout() #apply dropout and relu on last layer 
    def forward(self,images):
        out=self.model(images)
        for name,params in self.model.named_parameters():
            if ('bias' in name )or ('weight' in name):
                params.requires_grad=True #train only the bias and weight layers 
            else :
                params.requires_grad=self.train_cnn 
        return (self.dropout(self.relu(out))) #apply relu and dropout on the encoder output 



class Decoder(nn.Module):
    def __init__(self,embed_size,vocabulary_size,hidden_size,num_layers):
        super(Decoder,self).__init__()
        self.embed=nn.Embedding(num_embeddings=vocabulary_size,embedding_dim=embed_size)#the embedding of the caption 
        self.lstm= nn.LSTM(input_size=embed_size,hidden_size=hidden_size,num_layers=num_layers)#LSTM network 
        self.dropout=nn.Dropout()
        self.linear=nn.Linear(in_features=hidden_size,out_features=vocabulary_size)
    def forward(self,out,caption):
        embedding=self.dropout(self.embed(caption))
        res=torch.cat([out.unsqueeze(0),embedding],dim=0) #concat the encoder output with the embedding
        hiddens,_=self.lstm(res)
        return self.linear(hiddens) 

class RCNN(pl.LightningModule):
    def __init__(self,embed_size,vocab_size,hidden_size,num_layers,train_cnn) :
        super(RCNN,self).__init__()
        self.encoder=Encoder(embed_size=embed_size) 
        self.decoder=Decoder(embed_size,vocabulary_size=vocab_size,
        hidden_size=hidden_size,num_layers=num_layers)
    def forward(self,image,caption):#calculate RCNN output

        feature=self.encoder(image)
        out=self.decoder(feature,caption)
        return out
    def training_step(self,*batch): 
        criterion=nn.CrossEntropyLoss(ignore_index=0)#define loss function
        image=batch[0][0]#get the image 
        caption=batch[0][1]#get the caption 
        output=self(image,caption[:-1])
        loss_step=criterion(output.reshape(-1,output.shape[2]),caption.reshape(-1))#calculate loss
        self.log('loss_step',loss_step)
        return {'loss':loss_step}
    def configure_optimizers(self):
        parameters=self.parameters()
        optimizer=optim.Adam(params=parameters,lr=3e-4)#configure optimzer
        return optimizer
    def train_dataloader(self):
        root_dir=os.path.join(os.getcwd(),'data/Flicker8k_Dataset')
        captions_file=os.path.join(os.getcwd(),"data/captions.txt")
        dataloader=get_loader(root_dir=root_dir,captions_file=captions_file,transform=transform)
        return   dataloader #return the data loader 
      
        
    

    
        
 
#model=RCNN(20,2994,20,1,train_cnn=False)
#trainer=pl.Trainer(fast_dev_run=False,gpus=1)
#trainer.fit(model)
#b=torch.randint((30,2))
#model(torch.rand((2,3,299,299)),b)