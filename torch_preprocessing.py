import torch


def nanminmax(tensor, operation='min', dim=None, keepdim=False):
    if operation not in ['min','max']:
        raise ValueError("Operation must be 'min' or 'max'.")
    mask = torch.isnan(tensor)
    replacement = float('-inf') if operation == 'max' else float('inf')
    replacement = torch.tensor(replacement, dtype=tensor.dtype)
    tensor_masked = torch.where(mask, replacement, tensor)
    
    if operation == 'min':
        values, _ = torch.min(tensor_masked, dim=dim, keepdim=keepdim)
    else:
        values, _ = torch.max(tensor_masked, dim=dim, keepdim=keepdim)
        
    values = torch.where(torch.all(mask, dim=dim, keepdim=keepdim), torch.tensor(float('nan'), \
        dtype=tensor.dtype), values)
    return values


def nanstd(tensor, dim=None):
    mean = torch.nanmean(tensor, dim=dim, keepdim=True)
    deviations = tensor - mean
    squared_deviations = torch.square(deviations)
    nanvar = torch.nanmean(squared_deviations, dim=dim, keepdim=True)
    nanstd = torch.sqrt(nanvar)
    return nanstd


def zscore(tensor, dim=None):
    mean = torch.nanmean(tensor, dim=dim, keepdim=True)
    std = nanstd(tensor, dim=dim)
    zscore = (tensor - mean) / std
    return zscore


class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data):
        self.min = torch.min(data, dim=0).values
        self.max = torch.max(data, dim=0).values

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
