import torch


def nanminmax(tensor, operation='min', dim=None, keepdim=False):
    if operation not in ['min','max']:
        raise ValueError("Operation must be 'min' or 'max'.")
        
    mask = torch.isnan(tensor)
    replacement = float('-inf') if operation == 'max' else float('inf')
    replacement = torch.tensor(replacement, dtype=tensor.dtype, device=tensor.device)
    tensor_masked = torch.where(mask, replacement, tensor)
    
    if operation == 'min':
        values, _ = torch.min(tensor_masked, dim=dim, keepdim=keepdim)
    else:
        values, _ = torch.max(tensor_masked, dim=dim, keepdim=keepdim)
        
    values = torch.where(torch.all(mask, dim=dim, keepdim=keepdim), torch.tensor(float('nan'), \
        dtype=tensor.dtype, device=tensor.device), values)
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


def torch_nan_to_num(input_tensor, nan=0.0, posinf=None, neginf=None):
    output_tensor = torch.where(torch.isnan(input_tensor), torch.tensor(nan), input_tensor)
    
    if posinf is not None:
        output_tensor = torch.where(torch.isposinf(output_tensor), torch.tensor(posinf), output_tensor)
    
    if neginf is not None:
        output_tensor = torch.where(torch.isneginf(output_tensor), torch.tensor(neginf), output_tensor)
    
    return output_tensor
    

class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data):
        self.min = nanminmax(data, 'min', dim=0)
        self.max = nanminmax(data, 'max', dim=0)

    def transform(self, data):
        return (data - self.min) / (self.max - self.min + 1e-10)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    

class QuantileTransformer:
    def __init__(self, n_quantiles=100):
        self.n_quantiles = n_quantiles
        self.quantiles = None

    def fit(self, data):
        self.quantiles = torch.quantile(data, torch.linspace(0, 1, self.n_quantiles))

    def transform(self, data):
        sorted_data = torch.sort(data, dim=0).values
        ranks = torch.searchsorted(self.quantiles, sorted_data)
        transformed_data = ranks / (self.n_quantiles - 1)
        return transformed_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
