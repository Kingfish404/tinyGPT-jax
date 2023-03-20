import optax
import argparse
import jax
import jax.numpy as jnp

from rich import print
from rich.text import Text
from rich.panel import Panel
from rich.prompt import Prompt
from flax.training import checkpoints, train_state
from jax import jit
from jax.random import PRNGKey
from flax.training import checkpoints, train_state
from model import tinyGPT, generate
from dataset import TextDataset

dataset = TextDataset()
ckpt_dir = 'tmp/flax-checkpointing'
ckpt = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None)
config = ckpt['config']
model = tinyGPT(
    **config['model'],
    block_size=config['block_size'],
)
optimizer = optax.adamw(config['lr'])
state = train_state.TrainState.create(
    apply_fn=jit(model.apply),
    params=ckpt['model']['params'],
    tx=optimizer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reply-size", type=int, default=128)
    args = parser.parse_args()
    reply_size = args.reply_size
    nextkey = PRNGKey(0)

    while True:
        prompt = Prompt.ask(">>> ")
        if prompt == "exit":
            break
        key, nextkey = jax.random.split(nextkey)
        idx = jnp.array([dataset.encode(prompt)])
        generated_tokens = generate(
            state, idx, max_new_token=reply_size,
            block_size=config['block_size'], key=key)
        generated_text = dataset.decode(generated_tokens[0].tolist())
        generated_text = Text(generated_text, style="green")
        print(Panel(generated_text, width=120, title="Reply"))


if __name__ == "__main__":
    main()
