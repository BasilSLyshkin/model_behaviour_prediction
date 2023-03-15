import numpy as np
from scipy.stats import multivariate_normal as mvnorm
from math import pi
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import DataLoader
import torch
from scripts.model import LanguageModel
from scripts.dataset import TextDataset
from tqdm.notebook import tqdm
from torch import nn
class GDA:
    
    def __init__(self, model_name, n_features : int = 256):
        self.n_features = 256
        self.model_name = model_name
        self.mu = None
        self.sigma = None
        self.model = None 
        
    def fit(self,X):
        test_set = TextDataset(X, 'all')
        loader = DataLoader(test_set, batch_size=1000, shuffle=False)

        self.model = self.load_checkpoint(X)
        
        embeds = torch.tensor([])
        for _,indices, lengths, _ in tqdm(loader):
            embeds = torch.cat([embeds,self.pass_through(indices, lengths)])
            
        self.mu = embeds.mean(0).numpy()
        self.sigma = np.cov(embeds.numpy(), rowvar=0)
    
    def predict(self, X, lengths):
        embeds = self.pass_through(X, lengths)
        x_minus_mu0 = embeds.cpu() - self.mu
        a = -(x_minus_mu0.T * np.matmul(np.linalg.inv(self.sigma), x_minus_mu0.T)).sum(0)
        b = -(self.n_features * np.log(2*pi) + np.linalg.det(self.sigma))
        return 0.5*(a+b)
    
    def pass_through(self,X, lengths):
        embeds = self.model.embedding(X)

        packed_embeds = pack_padded_sequence(
            input=embeds, 
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, _ = self.model.rnn(packed_embeds)

        output, _ = pad_packed_sequence(
            sequence=packed_output, 
            batch_first=True
        )
        return output.mean(1)

    
    def load_checkpoint(self, X):
        model_dict = {
            'rnn':nn.RNN,
            'lstm':nn.LSTM,
            'gru':nn.GRU
        }
        path = f'models/{self.model_name}.pth'
        checkpoint = torch.load(path)
        train_set = TextDataset(X, split = 'train')
        model = LanguageModel(train_set, rnn_type=model_dict[self.model_name])
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False

        model.eval()
        return model


