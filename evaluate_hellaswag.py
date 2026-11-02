"""Evaluate the project's trained model on HellaSwag's labeled validation split."""
import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def completion_scores(logits, tokens, mask):
    """Return summed and length-normalized next-token losses for each ending."""
    losses = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.size(-1)),
        tokens[:, 1:].reshape(-1),
        reduction="none",
    ).view(tokens.size(0), -1)
    ending_mask = mask[:, 1:]
    counts = ending_mask.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("Each candidate ending must contain at least one scored token")
    summed = (losses * ending_mask).sum(dim=1)
    return summed, summed / counts


@torch.no_grad()
def evaluate_hellaswag(model, output_dir, device="cuda", max_examples=None):
    from hellaswag import iterate_examples, render_example

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    started = time.perf_counter()
    correct = correct_norm = total = 0
    logger.info("Starting HellaSwag evaluation: labeled validation split, four candidate endings")
    try:
        with (output_dir / "hellaswag_predictions.jsonl").open("w") as predictions:
            for example in iterate_examples("val"):
                if max_examples is not None and total >= max_examples:
                    break
                _, tokens, mask, label = render_example(example)
                label = int(label)
                if not 0 <= label < 4:
                    raise ValueError("HellaSwag accuracy requires a labeled example")
                if tokens.size(1) > model.config.block_size:
                    raise ValueError("HellaSwag example exceeds the model's context length")
                tokens, mask = tokens.to(device), mask.to(device)
                autocast = torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else nullcontext()
                with autocast:
                    logits, _ = model(tokens)
                summed, normalized = completion_scores(logits, tokens, mask)
                predicted = summed.argmin().item()
                predicted_norm = normalized.argmin().item()
                total += 1
                correct += predicted == label
                correct_norm += predicted_norm == label
                predictions.write(json.dumps({
                    "id": example.get("ind"), "label": label,
                    "prediction": predicted, "prediction_normalized": predicted_norm,
                    "ending_losses": normalized.tolist(),
                }) + "\n")
                if total % 500 == 0:
                    logger.info("HellaSwag progress: %d examples | accuracy: %.4f | normalized accuracy: %.4f", total, correct / total, correct_norm / total)
        if total == 0:
            raise ValueError("No HellaSwag examples were evaluated")
        result = {
            "dataset": "HellaSwag", "split": "validation",
            "examples": total, "correct": correct, "correct_normalized": correct_norm,
            "accuracy": correct / total, "accuracy_normalized": correct_norm / total,
            "elapsed_seconds": time.perf_counter() - started,
        }
        (output_dir / "hellaswag_results.json").write_text(json.dumps(result, indent=2) + "\n")
        logger.info("HellaSwag complete: %d examples | accuracy: %.4f | normalized accuracy: %.4f | time: %.1fs", total, result["accuracy"], result["accuracy_normalized"], result["elapsed_seconds"])
        return result
    except Exception:
        logger.exception("HellaSwag evaluation failed; trained weights remain saved")
        raise
    finally:
        model.train(was_training)
