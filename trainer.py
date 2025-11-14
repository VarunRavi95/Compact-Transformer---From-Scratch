import torch
import torch.nn.functional as F
import torch.optim as optim
import math
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# Local imports
from config import ModelArgs, get_args
from model import Transformer, initialize_tokenizer_in_model
from data import prepare_dataset, load_datasets
from tokenizer import initialize_tokenizer

def _save_snapshot(model, optimizer, scheduler, epoch, step):
    """Save model checkpoint"""
    snapshot = {
        "MODEL_STATE": model.state_dict(),
        "OPTIMIZER_STATE": optimizer.state_dict(),
        "EPOCHS_RUN": epoch,
        "STEP_RUN": step
    }
    torch.save(snapshot, f"checkpoints/snapshot_{step}.pt")
    print(f"Epoch: {epoch} | Step: {step} | Snapshot saved.")

