from common.Libraries import *
from common.GlobalConfig import kernel_sizes_map

# Custom Dataset
class MyDataset(Dataset):
    def __init__(self, seq, y):
        assert len(seq) == len(y)
        self.seq = seq
        self.y = y
    def __getitem__(self, idx):
        return np.asarray(self.seq[idx]), self.y[idx]
    def __len__(self):
        return len(self.seq)
class BertDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)
def save_model(model, name):
    return torch.save(model, f"./best_model/{name}.pth")
def load_model(name):
    return torch.load(f"./best_model/{name}.pth")
def save_model_info(d, name):
    with open(f'./best_model/{name}.pkl', 'wb') as f:
        pickle.dump(dictionary, f)
def load_model_info(d, name):
    with open(f'./best_model/{name}.pkl', 'rb') as f:
        return pickle.load(f)
def get_loss_function(x):
    if x == 'BCE':
        def loss(pred, actual):
            loss_func = torch.nn.BCEWithLogitsLoss()
            correct_count = (pred.argmax(1) == actual.argmax(1)).sum().item()
            return (loss_func(pred, actual), correct_count)
        return loss
    else:
        def loss(pred, actual):
            loss_func = torch.nn.CrossEntropyLoss()
            correct_count = (pred.argmax(1) == actual).sum().item()
            return (loss_func(pred, actual), correct_count)
        return loss
def get_clean_params(params):
    nP = params.copy()
    nP['hidden_size'] = int(nP['hidden_size'])
    nP['num_filters'] = int(nP['num_filters'])
    nP['num_mlp_layers'] = int(nP['num_mlp_layers'])
    nP['num_cnn_layers'] = int(nP['num_cnn_layers'])
    nP['pool_size'] = int(nP['pool_size'])
    nP['num_of_epochs'] = int(nP['num_of_epochs'])
    nP['kernel_size'] = kernel_sizes_map[int(nP['kernel_index'])-1]
    if nP['sigmoid'] > 0.5:
        nP['sigmoid'] = True
    else:
        nP['sigmoid'] = False
    nP.pop('kernel_index', None)
    return nP
def check_valid_cnn_output_size(init_size, n_layers, kernel_sizes, pool_size,stride=1):
    for i in range(len(kernel_sizes)):
        o_size = init_size
        o_size = np.floor((o_size - kernel_sizes[i])/stride) + 1
        if o_size <= 0:
            return False
        for j in range(n_layers-1):
            o_size = np.floor((o_size - pool_size)/pool_size) + 1
            if o_size <= 0:
                return False
            o_size = np.floor((o_size - kernel_sizes[i])/stride) + 1
            if o_size <= 0:
                return False
    return True
def plot_confusion_matrix(y_pred, y_true, cat_list, title='Confusion Matrix'):
    cnf_matrix = confusion_matrix(y_true, y_pred, labels=cat_list)
    # Plot confusion matrix in a beautiful manner
    fig = plt.figure(figsize=(16, 14))
    ax= plt.subplot()
    sns.heatmap(cnf_matrix, annot=True, ax = ax, fmt = 'g')
    # labels, title and ticks
    ax.set_xlabel('Predicted', fontsize=20)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=90)
    ax.xaxis.set_ticklabels(cat_list, fontsize = 10)
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True', fontsize=20)
    ax.yaxis.set_ticklabels(cat_list, fontsize = 10)
    plt.yticks(rotation=0)

    plt.title(title, fontsize=20)
    plt.show()
def print_classification_report(y_pred, y_true, cat_list, title='Classification Report:'):
    print(title)
    print(classification_report(y_true, y_pred, target_names=cat_list))

def predict(model, test, index_to_tag, tag_to_cat):
    model.to(torch.device("cuda"))
    model.eval()
    y_true = []
    y_pred = []
    y_true_cat = []
    y_pred_cat = []
    with tqdm.tqdm(test) as t:
        for x, y in t:
            x_ = x.to(torch.device("cuda"))
            logits = model(x_)
            y_true = y_true + y.argmax(1).tolist()
            y_pred = y_pred + logits.argmax(1).tolist()
            y_true_cat = y_true_cat + [tag_to_cat.loc[index_to_tag['tag'][k], 'cat'] for k in y.argmax(1).tolist()]
            y_pred_cat = y_pred_cat + [tag_to_cat.loc[index_to_tag['tag'][k], 'cat'] for k in logits.argmax(1).tolist()]
    return y_true, y_pred, y_true_cat, y_pred_cat
def predict_bert(model, test, index_to_tag, tag_to_cat):
    model.to(torch.device("cuda"))
    model.eval()
    y_true = []
    y_pred = []
    y_true_cat = []
    y_pred_cat = []
    with tqdm.tqdm(test) as t:
        for x, y in t:
            x_ = x.to(torch.device("cuda"))
            logits = model(x_, None)[0]
            y_true = y_true + y.flatten().tolist()
            y_pred = y_pred + logits.argmax(1).tolist()
            y_true_cat = y_true_cat + [tag_to_cat.loc[index_to_tag['tag'][k], 'cat'] for k in y.flatten().tolist()]
            y_pred_cat = y_pred_cat + [tag_to_cat.loc[index_to_tag['tag'][k], 'cat'] for k in logits.argmax(1).tolist()]
    return y_true, y_pred, y_true_cat, y_pred_cat
def predict_word_vec(model, test):
    model.to(torch.device("cuda"))
    model.eval()
    y_true = []
    y_pred = []
    with tqdm.tqdm(test) as t:
        for x, y in t:
            x_ = x.to(torch.device("cuda"))
            logits = model(x_)
            y_true = y_true + y.argmax(1).tolist()
            y_pred = y_pred + logits.tolist()
    return y_true, y_pred