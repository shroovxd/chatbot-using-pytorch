if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

import json
import numpy as np


import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader


from nltk_utils import tokenize,bag_of_words,stem
from model import NeuralNet
#reading the intents json file
with open('intents.json','r') as f:
    intents = json.load(f)

all_words=[]
tag=[]
xy=[]

for i in intents['intents']:
    tags = i['tags']
    tag.append(tags)
    for pattern in i['patterns']:
        w=tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tags))

ignore_words = ['?','.','!',',']
all_words= [stem(i) for i in all_words if i not in ignore_words]

#training data
X_train=[]
Y_train =[]

for (pattern, tags) in xy:
    bag = bag_of_words(pattern, all_words)
    X_train.append(bag)

    label = tag.index(tags)
    Y_train.append(label)


X_train = np.array(X_train)
Y_train = np.array(Y_train)



class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.n_samples
    
   

batch_size = 4
hidden_size = 8
output_size = len(tag)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000




chat_dataset = ChatDataset()
train_loader = DataLoader(dataset=chat_dataset, batch_size=4, shuffle=True, num_workers=0)  


model = NeuralNet(input_size, hidden_size, output_size)
device = torch.device('cuda 'if torch.cuda.is_available() else 'cpu')

#optimisers and criterions - loss fns to reduce errors and activation fns to output based on the input - 0 and 1

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(num_epochs):
    for (words,labels) in train_loader:
        words = words.to(device,dtype=float)
        labels = labels.to(device)
        labels=labels.long()
        #forward
        output = model(words)
        loss = criterion(output,labels)
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 100 == 0 :
        print(f'epoch {epoch+1}/{num_epochs},loss={loss.item():.4f}')

print(f'final loss , loss={loss.item():.4f}')

data = {
    "model_state" : model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "tag": tag,
    "all_words":all_words

}

FILE = "data.pth"
torch.save(data,FILE)
print(f'training completed , file saved to {FILE}')

