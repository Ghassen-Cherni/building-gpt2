from torch import nn
import torch
import tiktoken
from model import GPT2, GPTConfig, ShardDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from pathlib import Path
import logging
import sys
from datetime import datetime
from evaluate_hellaswag import evaluate_hellaswag
from tqdm import tqdm


class ProgressLoggingHandler(logging.Handler):
    def emit(self, record):
        if getattr(record, "file_only", False):
            return
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:
            self.handleError(record)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    run_dir = Path(__file__).resolve().parent / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[ProgressLoggingHandler(), logging.FileHandler(run_dir / "training.log")],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger.info("Run logs: %s", run_dir)
    
    config = GPTConfig(vocab_size = 50304)
    model = GPT2(config)
    train = True
    if train==True:
        
        model.to("cuda")
        raw_model = model
        model = torch.compile(model)
        sentence_size = 1024
        dataset = ShardDataset(
            Path(__file__).resolve().parent / "edu_fineweb10B",
            sentence_size=sentence_size,
            split="train",
        )


        torch.set_float32_matmul_precision("high")
        mini_batch_size = 128
        dataloader = DataLoader(dataset, batch_size=mini_batch_size, drop_last=True)
        val_dataset = ShardDataset(
            Path(__file__).resolve().parent / "edu_fineweb10B",
            sentence_size=sentence_size,
            split="val",
        )
        val_dataloader = DataLoader(val_dataset, batch_size=64, drop_last=True)
        val_loss_steps = min(20, len(val_dataloader))
        if val_loss_steps == 0:
            raise ValueError("Validation dataset must contain at least one full batch")
        nb_epochs = 1
        
        token_batch_size = 524_288 # paper says 0.5M but we needed a pretty number, i.e power of 2
       
        assert  token_batch_size % (mini_batch_size * sentence_size) == 0

        batch_size = token_batch_size // (mini_batch_size * sentence_size)

        nb_steps_per_epoch = dataloader.__len__() // batch_size

        max_lr = 6e-4
        warmup_steps = 10
        min_lr = 6e-5
        total_steps = nb_epochs * nb_steps_per_epoch
        logger.info(
            "Training: %d shards, %d examples, %d optimizer steps, %d tokens/step; validation: %d batches/check",
            len(dataset.paths), len(dataset), total_steps, token_batch_size, val_loss_steps,
        )

        params = [p for p in model.parameters() if p.requires_grad]
        non_decay_params = [p for p in params if p.dim() < 2]
        decay_params = [p for p in params if p.dim() >= 2]
        params = [{"params": non_decay_params, "weight_decay": 0.0}, { "params": decay_params, "weight_decay": 0.1}]
        optimizer = torch.optim.AdamW(params, lr=max_lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
        warmup = LinearLR(optimizer=optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine_decay = CosineAnnealingLR(optimizer, T_max = total_steps - warmup_steps, eta_min=min_lr)
        scheduler = SequentialLR(optimizer, [warmup, cosine_decay], milestones=[warmup_steps])

        

        
        model.train()
        checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        losses = []
        data_iter = iter(dataloader)
        progress = tqdm(range(total_steps), desc="Training", unit="step", dynamic_ncols=True)
        latest_val_loss = None
        for step in progress:
            torch.cuda.synchronize()
            initial_time = time.time()
            optimizer.zero_grad() 
            loss_accum = torch.zeros((), device="cuda")
            for mini_batch in range(batch_size):
                
                try:
                    data, label = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    data, label = next(data_iter)
                data = data.to("cuda")
                label = label.to("cuda")
                    
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output, loss = model(data, label)
                    loss = loss/batch_size
                loss.backward()
                loss_accum += loss.detach()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            current_lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            losses.append(loss_accum.item())
            torch.cuda.synchronize()
            batch_time = time.time() - initial_time
            logger.info(
                "step %d/%d | time per batch: %.2fs | train loss: %.4f | lr: %.3e | grad norm: %.4f | tokens/s: %.0f",
                step, total_steps - 1, batch_time, loss_accum.item(), current_lr,
                norm.item(), token_batch_size / batch_time,
                extra={"file_only": True},
            )


            last_step = step == total_steps - 1
            if step % 250 == 0 or last_step:
                model.eval()
                val_iter = iter(val_dataloader)
                val_loss_accum = torch.zeros((), device="cuda")
                with torch.no_grad():
                    for _ in range(val_loss_steps):
                        val_data, val_label = next(val_iter)
                        val_data = val_data.to("cuda")
                        val_label = val_label.to("cuda")
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            _, val_loss = model(val_data, val_label)
                        val_loss_accum += val_loss.detach() / val_loss_steps
                logger.info("step %d | validation loss: %.4f | batches: %d", step, val_loss_accum.item(), val_loss_steps)
                latest_val_loss = val_loss_accum.item()
                model.train()

                if step > 0 and (step % 5000 == 0 or last_step):
                    checkpoint_path = checkpoint_dir / f"model_{step:05d}.pt"
                    torch.save({
                        "model": raw_model.state_dict(),
                        "config": vars(config),
                        "step": step,
                        "val_loss": val_loss_accum.item(),
                    }, checkpoint_path)
                    logger.info("Saved checkpoint: %s", checkpoint_path)

            metrics = {"loss": f"{losses[-1]:.4f}", "batch_s": f"{batch_time:.2f}", "lr": f"{current_lr:.2e}"}
            if latest_val_loss is not None:
                metrics["val_loss"] = f"{latest_val_loss:.4f}"
            progress.set_postfix(metrics, refresh=False)

        weights_path = Path(__file__).resolve().parent / "gpt2_weights.pt"
        torch.save(raw_model.state_dict(), weights_path)
        logger.info("Training finished; saved final weights: %s", weights_path)
        plt.plot(losses)
        plt.xlabel("Training step (batch)")
        plt.ylabel("Loss")
        plt.title("Training loss")
        plt.savefig(run_dir / "training_loss.png")
        plt.close()
        # Use the uncompiled model for variable-length benchmark examples.
        evaluate_hellaswag(raw_model, run_dir, device="cuda")

    # weights = torch.load(
    # "gpt2_weights.pt",
    # map_location="cpu",
    # weights_only=True,
    # )
    # model.load_state_dict(weights)
    # model.to("cuda")

    # enc = tiktoken.get_encoding("gpt2")
    # model.eval()
    # sentence = "What, art thou  "
    # tokens = torch.tensor(enc.encode(sentence), device="cuda").view(1, -1)  # Add batch dimension
    # nb_tokens_to_predict = 4
    
    # while (nb_tokens_to_predict>0):
    #     with torch.no_grad():
    #         output, _ = model(tokens)

    #     last_logit = output[:, -1, :]

    #     probabilities = torch.softmax(last_logit, dim=-1)
    #     top_probability_token = probabilities.argmax(dim=-1, keepdim=True)

    #     tokens = torch.cat((tokens, top_probability_token), dim=1)
    #     nb_tokens_to_predict-=1

    # print(enc.decode(tokens[0].tolist()))
    
