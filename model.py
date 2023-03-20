import jax
import jax.numpy as jnp
from tqdm import tqdm
from jax import Array
from jax.random import KeyArray
from flax import linen as nn
from flax.training import train_state


class TransformerDecoder(nn.Module):
    n_head: int
    n_embd: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: Array):
        mask = nn.make_causal_mask(jnp.ones((x.shape[0], x.shape[1])))
        x = x + \
            nn.SelfAttention(num_heads=self.n_head, dropout_rate=self.dropout_rate)(
                nn.LayerNorm()(x), mask=mask, deterministic=True)
        x = x + nn.Dense(features=self.n_embd)(nn.LayerNorm()(x))
        return x


class tinyGPT(nn.Module):
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    vocab_size: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x: Array):
        tok_embd = nn.Embed(num_embeddings=self.vocab_size,
                            features=self.n_embd)(x)
        pos_embd = nn.Embed(num_embeddings=self.block_size,
                            features=self.n_embd)(jnp.arange(x.shape[1]))
        x = tok_embd + pos_embd
        x = nn.Sequential([
            TransformerDecoder(
                n_head=self.n_head,
                n_embd=self.n_embd,
                dropout_rate=self.dropout_rate, )
            for _ in range(self.n_layer)]
        )(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(features=self.vocab_size)(x)
        return x


def generate(state: train_state.TrainState, idx: Array, max_new_token: int, block_size: int, key: KeyArray):
    nextkey = key
    for _ in tqdm(range(max_new_token)):
        idx_cond = idx[:, -block_size:]
        logits = state.apply_fn({'params': state.params}, idx_cond)
        logits = jnp.array(logits)[:, -1, :]
        key, nextkey = jax.random.split(nextkey)
        next_token = jax.random.categorical(
            key, logits, axis=-1).reshape(-1, 1)
        idx = jnp.concatenate((idx, next_token), axis=-1)
    return idx
