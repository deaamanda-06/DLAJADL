import numpy as np

def muat():
    """Memuat data simulasi dummy dengan tipe data PyTorch-friendly."""
    np.random.seed(42)
    
    # Data Latih (Train)
    X_tr = np.random.randn(100, 3, 224, 224).astype(np.float32)
    y_tr = np.random.randint(0, 2, size=(100,)).astype(np.int64)
    
    # Data Validasi (Validation)
    X_va = np.random.randn(25, 3, 224, 224).astype(np.float32)
    y_va = np.random.randint(0, 2, size=(25,)).astype(np.int64)
    
    # Data Uji (Test)
    X_te = np.random.randn(25, 3, 224, 224).astype(np.float32)
    y_te = np.random.randint(0, 2, size=(25,)).astype(np.int64)
    
    return X_tr, y_tr, X_va, y_va, X_te, y_te