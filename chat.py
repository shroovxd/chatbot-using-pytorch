import json
import torch
import random
from nltk_utils import bag_of_words,tokenize
from model import NeuralNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json','r') as f:
    intents = json.load(f)


FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words = data["all_words"]
tag = data["tag"]
model_state = data["model_state"]



model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

#chat starts
bot_name = "cha-cha"
print("let's chat, I am cha-cha. press quit to exit")

while True:
    sentence = input("You: ")
    if sentence.lower() == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)

    _, predicted = torch.max(output.data, 1)
    i= tag[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
            i = tag[predicted.item()]
            for intent in intents["intents"]:
                if i == intent["tags"]:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I don't understand.")


