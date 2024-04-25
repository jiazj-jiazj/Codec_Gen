from torch.utils.data import Dataset, DataLoader  
  
class MyDataset(Dataset):  
    def __init__(self, data):  
        self.data = data  
  
    def __len__(self):  
        return len(self.data)  
  
    def __getitem__(self, index):  
        return self.data[index]  
  
data = range(100)  
dataset = MyDataset(data)  
dataloader = DataLoader(dataset, batch_size=10)  
print(len(DataLoader))
for batch in dataloader:  
    print(batch)  
