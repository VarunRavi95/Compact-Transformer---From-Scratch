import argparse
from dataclasses import dataclass

def get_args():
    parser = argparse.ArgumentParser(description='Compact Transformer - Encoder - Decoder Transformer for Translation')

    # Required args for Model Architecture
    parser.add_argument('--block_size', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Training Batch Size')
    parser.add_argument('--embedding_dims', type=int, default=512, help='Model Embedding Dimensions')
    parser.add_argument('--no_of_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--no_of_decoder_layers', type=int, default=6, help='Number of decoder layers')

    # Vocab size
    parser.add_argument('--src_vocab_size', type=int, default=None, help='Source Vocabulary size (will be updated based on Tokenizer)')
    parser.add_argument('--tgt_vocab_size', type=int, default=None, help='Target Vocabulary size (will be updated based on Tokenizer)')

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=1, help='Number of Training Epochs')
    parser.add_argument('--max_lr', type=float, default=6e-4, help='Maximum Learning Rate')
    parser.add_argument('--weight_decay_optim', type=float, default=0.1, help='Weight Decay for Optimizer')
    parser.add_argument('--beta_1', type=float, default=0.9, help='Beta1 for optimizer')
    parser.add_argument('--beta_2', type=float, default=0.98, help='Beta2 for optimizer')
    parser.add_argument('--eps', type=float, default=1e-9, help='Epsilon for optimizer')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping value')

    # Regularization
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout Rate')
    parser.add_argument('--attn_dropout', type=float, default=0.1, help='Attention dropout rate')

    # System Config
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--use_liger', action='store_true', default=True, help='Use Liger Kernels for optimization')

    # Training schedule
    parser.add_argument('--warmup_iters', type=int, default=400, help='Wamup Iterations')
    parser.add_argument('--lr_decay_iters', type=int, default=10000, help='Learning Rate Decay iterations')
    parser.add_argument('--min_lr_ratio', type=float, default=0.1, help='Minimum learning rate as ratio of max_lr')
    parser.add_argument('--total_iters', type=int, default=10000, help='Total Training Iterations')
    parser.add_argument('--eval_iters', type=int, default=200, help='Evaluation Interval')
    parser.add_argument('--eval_check', type=int, default=200, help='Number of eval batches')
    parser.add_argument('--save_checkpoint_iter', type=int, default=500, help='Checkpoint save interval')

    # Gradient Accumulation
    parser.add_argument('--total_batch_size', type=int, default=524288, help='Total effective batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None, help='Gradient accumulation steps (will be auto calculated)')
    
    # HuggingFace token
    parser.add_argument('--hf_token', type=str, default='', help='HuggingFace Token')

    return parser.parse_args()


@dataclass
class ModelArgs:
    """Model Configuration Arguments"""
    block_size: int = 512
    batch_size: int = 32
    embedding_dims: int = 512
    no_of_heads: int = 8
    no_of_decoder_layers: int = 6

    src_vocab_size: int = None
    tgt_vocab_size: int = None

    epochs: int = 1
    max_lr: float = 6e-4
    weight_decay_optim: float = 0.1
    beta_1: float = 0.9
    beta_2: float = 0.98
    eps: float = 1e-9
    clip: float = 1.0

    dropout: float = 0.1
    attn_dropout: float = 0.1

    device: str = 'cpu'
    use_liger: bool = True

    # Training Schedule
    warmup_iters: int = 400
    lr_decay_iters: int = 10000
    min_lr_ratio: float = 0.1
    min_lr: float = None
    total_iters: int = 10000
    eval_iters: int = 200
    eval_check: int = 200
    save_checkpoint_iter: int = 500
    
    # Gradient Accumulation
    total_batch_size: int = 524288
    gradient_accumulation_steps: int = None
    
    # HuggingFace Token
    hf_token: str = ''

    def __post_init__(self):
        """Calculate derived parameters"""
        if self.gradient_accumulation_steps is None:
            self.gradient_accumulation_steps = self.total_batch_size // (self.batch_size * 1 * self.block_size)

        if self.min_lr is None:
            self.min_lr = self.min_lr_ratio * self.max_lr
