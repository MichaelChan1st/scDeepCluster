import h5py
import numpy as np
import scanpy as sc
from preprocess import read_dataset, normalize
class dataset_process():
    def __init__(self, data_file):
        assert data_file is not None
        data = h5py.File(data_file)
        x = np.array(data['X'])
        y = np.array(data['Y'])
        data.close()
        self.adata = sc.AnnData(x)
        self.adata.obs['Group'] = y

        self.adata = read_dataset(self.adata,
                             transpose=False,
                             test_split=False,
                             copy=True)

        self.adata = normalize(self.adata,
                          size_factors=True,
                          normalize_input=True,
                          logtrans_input=True)
    def store(self):
        self.adata.write("data/CITE_adata_file.h5ad")
    def retrieve(self):
        return sc.read("data/CITE_adata_file.h5ad")