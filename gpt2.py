from torch import nn
import torch
import tiktoken
from model import GPT2, GPTConfig, Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

if __name__ == "__main__":

    
    config = GPTConfig(vocab_size = 50304)
    model = GPT2(config)
    train = True
    if train==True:
        
        model.to("cuda")
        model = torch.compile(model)
        sentence_size = 1024
        dataset = Dataset("tinyshakespeare.txt", sentence_size=sentence_size)
        torch.set_float32_matmul_precision("high")
        dataloader = DataLoader(dataset, batch_size=16, drop_last=True)
        nb_epochs = 1
        
        token_batch_size = 524_288 # paper says 0.5M but we needed a pretty number, i.e power of 2
        mini_batch_size = 16 
        assert  token_batch_size % (mini_batch_size * sentence_size) == 0

        batch_size = token_batch_size // (mini_batch_size * sentence_size)

        nb_steps_per_epoch = dataloader.__len__() // batch_size

        max_lr = 6e-4
        warmup_steps = 10
        min_lr = 6e-5
        total_steps = nb_epochs * nb_steps_per_epoch

        params = [p for p in model.parameters() if p.requires_grad]
        non_decay_params = [p for p in params if p.dim() < 2]
        decay_params = [p for p in params if p.dim() >= 2]
        params = [{"params": non_decay_params, "weight_decay": 0.0}, { "params": decay_params, "weight_decay": 0.1}]
        optimizer = torch.optim.AdamW(params, lr=max_lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
        warmup = LinearLR(optimizer=optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine_decay = CosineAnnealingLR(optimizer, T_max = total_steps - warmup_steps, eta_min=min_lr)
        scheduler = SequentialLR(optimizer, [warmup, cosine_decay], milestones=[warmup_steps])

        

        
        model.train()
        losses = []
        data_iter = iter(dataloader)
        for step in range(total_steps):
            torch.cuda.synchronize()
            initial_time = time.time()
            optimizer.zero_grad() 
            loss_accum = torch.zeros((), device="cuda")
            for mini_batch in range(batch_size):
                
                data, label = next(data_iter)
                data = data.to("cuda")
                label = label.to("cuda")
                    
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output, loss = model(data, label)
                    loss = loss/batch_size
                loss.backward()
                loss_accum += loss.detach()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss_accum.item())
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
    