

# Dataset paths
DATA = {
    "train_csv": "data/train.csv",
    "val_csv": "data/val.csv",
    "train_dir": "data/train_images",
    "val_dir": "data/val_images",
}

# Dataloader parameters
DATALOADER = {
    "batch_size": 16,
    "num_workers": 4,
    "shuffle_train": True,
    "shuffle_val": False,
}

# Model parameters
MODEL = {
    "backbone": "resnet152",
    "num_classes": 14,
    "use_cache": True,
    "return_name": True,
}

# Training parameters
TRAINING = {
    "epochs": 60,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "num_workkers": 8, #update as CPU core number
    "early_stopping_patience": 6,
}

# Metrics to track
METRICS = {
    "multi_label": True,
    "compute_f1": True,
    "compute_precision": True,
    "compute_recall": True,
    "compute_auroc": True,
}
