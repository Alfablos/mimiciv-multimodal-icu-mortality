from os import cpu_count

# deaths / total_stays in the training set
# needed due to heavily imbalanced label
loss_pos_weight = 936 / 6096

train_csv = '../ds_train.csv'
val_csv = '../ds_val.csv'
test_csv = '../ds_test.csv'

dataset_shuffle = True
num_workers = (cpu_count() // 2) - 2

hyperparameters = {
    'batch_size': 32,
    'epochs': 10,
    'dropout': 0.7,
    'learning_rate': 10e-5,
    'train_limit': 0.4
}