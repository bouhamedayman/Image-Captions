import spacy
spacy_eng=spacy.load("en")
import pickle
#we create Vocabulary dataset so it can process the 
#text data and return a numerical representation of
#the sentences 
class Vocabulary:
    
    def __init__(self,freq_threshold=5) :#if a word is repeated more then the 
                                        #Threshold we will assign UNK to it
        self.index_to_string={0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}#idx to string dictionnary
        self.string_to_index={"<PAD>":0,"<SOS>":1,"<EOS>":2,"<UNK>":3}#string to index dictionnary 
        #EOS end of sentence ,SOS start of sentence,PAD padding,UNK unknown
        self.freq_threshold=freq_threshold
    def __len__(self):
        return (len(self.index_to_string))
    def tokenizer_eng(self,text):#tokenizing the sentence
        return [tok.text.lower() for tok in spacy_eng(text)]
    def build_vocabulary(self,sentences_list):
        frequencies={}#to count words frequnecies
        idx=4
        print('building vocabulary please wait ... \n')
    
        for sentence in sentences_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies.keys():
                    frequencies[word]=1
                else:
                    frequencies[word]+=1
                if frequencies[word]==self.freq_threshold:#each should appear at least
                                             # 5 time to include it to the dictionary 
                    self.string_to_index[word]=idx
                    self.index_to_string[idx]=word
                    idx+=1
                    if idx%1000==0:
                        print('building vocabulary please wait ... \n')
        
        with open('vocab.pkl','wb') as vocab:
            pickle.dump(self,vocab,pickle.HIGHEST_PROTOCOL)

        print("Vocabulary built  ")
    def numericalize(self,text):#return a numericalized text 
        tokenized_text=self.tokenizer_eng(text)
        num_text=[]
        for word in tokenized_text:#return 
            if word in self.string_to_index.keys():
                num_text.append(self.string_to_index[word])
            else: 
                num_text.append(self.string_to_index['<UNK>'])
        return  num_text


