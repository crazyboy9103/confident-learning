from torch.utils.data import Dataset

class SubsetWithTransform(Dataset):
    def __init__(self, dataset, idxs, transform=None):
        self.dataset = dataset
        self.idxs = idxs
        self.transform = transform
        
    def __getitem__(self, index):
        image, target = self.dataset[index]
        if self.transform:
            image, target = self.transform(image, target)
        return image, target
        
    def __len__(self):
        return len(self.idxs)