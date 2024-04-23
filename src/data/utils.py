from torch.utils.data import Dataset

class SubsetWithTransform(Dataset):
    def __init__(self, dataset, idxs, transform=None):
        self.dataset = dataset
        self.idxs = idxs
        self.transform = transform
    
    # We want to iterate too
    def __iter__(self):
        for idx in self.idxs:
            image, target = self.dataset[idx]
            if self.transform:
                image, target = self.transform(image, target)

            yield image, target
                        
    def __getitem__(self, index):
        image, target = self.dataset[self.idxs[index]]
        if self.transform:
            image, target = self.transform(image, target)
        return image, target
        
    def __len__(self):
        return len(self.idxs)