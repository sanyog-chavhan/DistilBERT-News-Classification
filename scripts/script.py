import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import argparse
import os
import pandas as pd


s3_path = 's3-path-to-dataset'
df = pd.read_csv(s3_path, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

df = df[['TITLE', 'CATEGORY']]

my_dict = {
    'e': 'Entertainment',
    'b': 'Business',
    'm': 'Health',
    't': 'Science'
}

def update_cat(x):
    return my_dict[x]

df['CATEGORY'] = df['CATEGORY'].apply(lambda x: update_cat(x))


encode_dict = {}

def encode_cat(x):
    if x not in encode_dict.keys():
        encode_dict[x] = len(encode_dict)
    return encode_dict[x]


df['ENCODED_CAT'] = df['CATEGORY'].apply(lambda x: encode_cat(x))

df = df.reset_index(drop=True)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class NewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        title = str(self.data.iloc[index, 0])
        title = " ".join(title.split())

        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.iloc[index, 2], dtype=torch.long)
        }

    def __len__(self):
        return self.len


train_size = 0.8
train_dataset = df.sample(frac=train_size, random_state=200)
test_dataset = df.drop(train_dataset.index).reset_index(drop=True)

train_dataset.reset_index(drop=True)

print(f"Full dataset: {df.shape}")
print(f"Train dataset: {train_dataset.shape}")
print(f"Test dataset: {test_dataset.shape}")

MAX_LEN=512
TRAIN_BATCH_SIZE=4
VALID_BATCH_SIZE=2

training_set = NewsDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = NewsDataset(test_dataset, tokenizer, MAX_LEN)

train_parameters = {
    'batch_size': TRAIN_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0
}

test_parameters = {
    'batch_size': VALID_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0
}

training_loader = DataLoader(training_set, **train_parameters)
testing_loader = DataLoader(testing_set, **test_parameters)


class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 4)


    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooled_output = hidden_state[:, 0]
        pooler = self.pre_classifier(pooled_output)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


def calculate_accuracy(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()

    '''
    This function calculates the number of correct predictions by comparing
    
    the predicted indices (big_idx) with the true targets.
    It sums up the number of matches and returns the count.
    Args:
        big_idx (torch.Tensor): The predicted indices from the model.
        targets (torch.Tensor): The true labels for the data.
    Returns:
        int: The number of correct predictions.
    Example:
        [0.1, 0.2, 0.3, 0.4]    # [0 0 0 1] target [0 0 0 1] = True
        [0.5, 0.6, 0.8, 0.7]    # [0 0 1 0] target [0 0 1 0] = True
        [0.9, 0.32, 0.56, 0.72]   # [1 0 0 0] target [1 0 0 0] = True
        [0.44, 0.78, 0.34, 0.12]    # [0 1 0 0] target [0 0 1 0] = False
        tensor([True, True, True, False])  # This would yield 3 correct predictions.
        .sum().item()  # This sums up the True values to get the count of correct predictions.
    batch size = 4
    '''
    return n_correct



def train(epoch, model, device, training_loader, optimizer, loss_function):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()

    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)
        outputs = model.forward(ids, mask)

        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calculate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _ % 5000 == 0:   # 100k / 4 = 25000, 1 epoch = 5 times
            loss_step = tr_loss / nb_tr_steps
            acc_step = (n_correct*100) / nb_tr_examples
            print(f"Training loss per 5000 steps: {loss_step}, Training accuracy per 5000 steps: {acc_step}")


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    epoch_acc = (n_correct*100) / nb_tr_examples
    print(f"Training loss per epoch {epoch} : {epoch_loss}, Training accuracy per epoch {epoch} : {epoch_acc}")

    return




def valid(epoch, model, testing_loader, device, loss_function):
    model.eval()
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    tr_loss = 0

    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            outputs = model.forward(ids, mask).squeeze()

            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calculate_accuracy(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if _ % 1000 == 0:
                loss_step = tr_loss / nb_tr_steps
                acc_step = (n_correct*100) / nb_tr_examples
                print(f"Validation loss per 1000 steps: {loss_step}, Validation accuracy per 1000 steps: {acc_step}")

    epoch_loss = tr_loss / nb_tr_steps
    epoch_acc = (n_correct*100) / nb_tr_examples
    print(f"Validation loss per epoch {epoch} : {epoch_loss}, Validation accuracy per epoch {epoch} : {epoch_acc}")
    return



def main():
    print("Start")

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_batch_size",type=int,default=4)
    parser.add_argument("--valid_batch_size",type=int,default=2)
    parser.add_argument("--learning_rate",type=float,default=5e-5)
    
    args = parser.parse_args()

    args.epochs
    args.train_batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBERTClass()

    model.to(device)

    LEARNING_RATE = 1e-05

    optimizer = torch.optim.Adam(params = model.parameters(), lr=LEARNING_RATE)

    loss_function = torch.nn.CrossEntropyLoss()

    # Train loop

    EPOCHS = 2

    for epoch in range(EPOCHS):
        print(f"starting epoch {epoch}")
        train(epoch, model, device, training_loader, optimizer, loss_function)
        valid(epoch, model, testing_loader, device, loss_function)

    output_dir = os.environ['SM_MODEL_DIR']
    output_model_file = os.path.join(output_dir, "pytorch_distilbert_news.bin")
    output_vocab_file = os.path.join(output_dir, "distilbert_news_classifier_vocab.bin")
    torch.save(model.state_dict(), output_model_file)
    tokenizer.save_vocabulary(output_vocab_file)


if __name__ == "__main__":
    main()