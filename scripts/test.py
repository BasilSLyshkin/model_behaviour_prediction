from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from scripts.model import LanguageModel
import pandas as pd
from scripts.gda import GDA
import torch
from scripts.dataset import TextDataset
    
def test(model: LanguageModel, gda : GDA, loader: DataLoader):
    device = next(model.parameters()).device
    model.eval()
    final = pd.DataFrame(columns = ['date','prediction','target', 'LogGDA uncertainty'])
    
    for date, indices, lengths, targets in tqdm(loader):
        indices = indices[:,:lengths.max()].to(device).long()
        unc = gda.predict(indices, lengths)
        pred = model(indices, lengths).argmax(1).cpu()
        batch = pd.DataFrame({
            'date':date,
            'prediction':pred.cpu(),
            'target':targets,
            'LogGDA uncertainty':unc
        })
        final = pd.concat([final,batch], axis = 0)
    return final
        
        