config = {
    "batch_size": 32,                # smaller batch to avoid memory issues
    "image_size": 64,                # reduce from 96 → 64 to speed up
    "epochs": 10,                    # start with 10 for test run
    "temperature": 0.5,
    "lr": 3e-4,
    "dataset": "STL10",
    "model_name": "resnet18",
    "embedding_dim": 128,
    "projection_dim": 64
}
