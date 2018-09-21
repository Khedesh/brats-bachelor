APP_CONFIG = {
    'train': False,
    'test': False,
    'predict': True,

    'weights_file': 'weights.h5',
}

DATA_CONFIG = {
    'root_dir': '/Users/Khedesh/Desktop/Workspace/University/Project/Data/BRATS2015_Training/train/HGG',
    'batch_size': 256,
    'min_train_index': 0,
    'max_train_index': 4,
    'min_test_index': 5,
    'max_test_index': 6,
    'min_predict_index': 5,
    'max_predict_index': 6,
}

TRAIN_CONFIG = {
    'epochs': 1,
    'batch_size': 32,
}

TEST_CONFIG = {
    'batch_size': 32,
}

PREDICT_CONFIG = {
    'batch_size': 32,
}
