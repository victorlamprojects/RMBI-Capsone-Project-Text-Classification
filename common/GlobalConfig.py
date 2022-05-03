import itertools

random_seed = 42
# Max. sentence length
max_length=250
# Max. vector size - same as the one got by nlp.vocab.get_vector
vocab_size=300
batch_size = 16
num_epoch = 20
num_epoch_bert = 5
raw_data_size = 500
data_folder_path = f'./data/raw/{raw_data_size}'
hashtag_list_path = f'./data/hashtag_list.csv'
clean_folder_path = f'./data/clean_data/{raw_data_size}'
client_folder_path = f'./data/clients'
preprocessed_folder_path = f'./data/preprocessed_data/{raw_data_size}'
# Minimum number of words per post
threshold = 10
# Minimum number of posts
threshold_post = 300

kernel_sizes_map = []
kernel_sizes = [1,2,3,4,5]
for i in range(1,len(kernel_sizes)+1):
    kernel_sizes_map += list(itertools.combinations(kernel_sizes,i))
    
param_grid = {
    'hidden_size': (16, 256),
    'dropout_rate': (0.1, 0.5), 
    'learning_rate': (0.001, 0.01), 
    'l2_reg': (0.0001, 0.001),
    'num_filters': (2, 64),
    'pool_size': (2, 64),
    'num_mlp_layers': (1, 5),
    'num_cnn_layers': (1, 5),
    'num_lstm_layers': (1, 5),
    'kernel_index': (1,31),
    'sigmoid': (0, 1),
    'num_of_epochs': (5, 25)
}