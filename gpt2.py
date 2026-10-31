from torch import nn
import torch
import tiktoken
from model import GPT2, GPTConfig, Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":

    
    config = GPTConfig()
    model = GPT2(config)
    train = True
    if train==True:
        
        model.to("cuda")
        dataset = Dataset("tinyshakespeare.txt", sentence_size=516)
        torch.set_float32_matmul_precision("high")
        dataloader = DataLoader(dataset, batch_size=16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        model.train()
        losses = []
        
        for batch, (data, label) in enumerate(dataloader):
            torch.cuda.synchronize()
            initial_time = time.time()
            data = data.to("cuda")
            label = label.to("cuda")
            optimizer.zero_grad()
            output, loss = model(data, label)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            torch.cuda.synchronize()
            batch_time = time.time() - initial_time
            print(batch_time)
        torch.save(model.state_dict(), "gpt2_weights.pt")
        plt.plot(losses)
        plt.xlabel("Training step (batch)")
        plt.ylabel("Loss")
        plt.title("Training loss")
        plt.show()

    weights = torch.load(
    "gpt2_weights.pt",
    map_location="cpu",
    weights_only=True,
    )
    model.load_state_dict(weights)
    model.to("cuda")

    enc = tiktoken.get_encoding("gpt2")
    model.eval()
    sentence = "What, art thou  "
    tokens = torch.tensor(enc.encode(sentence), device="cuda").view(1, -1)  # Add batch dimension
    nb_tokens_to_predict = 4
    
    while (nb_tokens_to_predict>0):
        with torch.no_grad():
            output, _ = model(tokens)

        last_logit = output[:, -1, :]

        probabilities = torch.softmax(last_logit, dim=-1)
        top_probability_token = probabilities.argmax(dim=-1, keepdim=True)

        tokens = torch.cat((tokens, top_probability_token), dim=1)
        nb_tokens_to_predict-=1

    print(enc.decode(tokens[0].tolist()))
    