import os
import shutil
import random
import jax
import jax.numpy as jnp
import optax
import fire
from tqdm import tqdm
from jax import jit, value_and_grad, Array
from jax.random import PRNGKey
from flax import linen as nn
from flax.training import checkpoints, train_state
from model import tinyGPT, generate
from dataset import TextDataset
from typing import Callable
from functools import partial


def main(
        epochs: int = 5000, lr: float = 1e-4, batch_size: int = 128,
        block_size: int = 128, n_embd: int = 512, n_head: int = 4, n_layer: int = 4,
        dropout_rate: float = 0.1,
        train_ratio: float = 0.9, train_samples: int = 100, eval_every: int = 100):
    dataset = TextDataset(train_ratio=train_ratio, train_samples=train_samples)
    config = {
        "train_ratio": dataset.train_ratio,
        "train_samples": dataset.train_samples,
        "batch_size": batch_size,
        "epochs": epochs,
        "eval_every": eval_every,
        "lr": lr,
        "block_size": block_size,
        "model": {
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "dropout_rate": dropout_rate,
            "vocab_size": dataset.vocab_size,
        }
    }

    nextkey = PRNGKey(random.randint(0, 1024))
    key, nextkey = jax.random.split(nextkey)
    optimizer = optax.adamw(config['lr'])
    model = tinyGPT(
        **config['model'],
        block_size=config['block_size'],
    )
    x1, y1 = dataset.get_batch(
        batch_size=config['batch_size'], block_size=config['block_size'], key=key)
    params = model.init(key, x1)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params['params'],
        tx=optimizer)

    print(model.tabulate(key, x1))
    ckpt_dir = 'tmp/flax-checkpointing'
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    @partial(jit, static_argnames=('apply_fn',))
    def loss_fn(params, batch, y, apply_fn: Callable):
        logits = apply_fn({'params': params}, batch)
        loss = optax.softmax_cross_entropy(
            logits, nn.one_hot(y, config['model']['vocab_size']))
        return loss.mean()

    @jit
    def train(state: train_state.TrainState, batch: Array, y: Array):

        loss, grads = value_and_grad(loss_fn)(
            state.params, batch, y, apply_fn=state.apply_fn)
        state = state.apply_gradients(grads=grads)
        return state, loss

    for i in tqdm(range(config['epochs'])):
        key, nextkey = jax.random.split(nextkey)
        batch, y = dataset.get_batch(batch_size=config['batch_size'],
                                     block_size=config['block_size'], key=key)

        state, train_loss = train(
            state=state, batch=batch, y=y)

        if i % config['eval_every'] == 0:
            ckpt = {'model': state, 'config': config, 'data': [x1]}
            checkpoints.save_checkpoint(ckpt_dir=ckpt_dir,
                                        target=ckpt,
                                        step=i,
                                        overwrite=False,
                                        keep=2)

            batch, y = dataset.get_batch(type='eval', batch_size=config['batch_size'],
                                         block_size=config['block_size'], key=key)
            eval_loss = loss_fn(state.params, batch, y,
                                apply_fn=state.apply_fn)
            logits = state.apply_fn({'params': state.params}, batch)
            next_token = jax.random.categorical(
                key, jnp.array(logits), axis=-1)
            print("\ntrain loss: %s, eval loss: %s" % (train_loss, eval_loss))
            print("encode: {%s}\ndecode: {%s}" %
                  (dataset.decode(batch[0].tolist()), dataset.decode(next_token[0].tolist())))

    if False:
        key, nextkey = jax.random.split(nextkey)
        print("decode full: {%s}" % dataset.decode(
            generate(
                state, jnp.zeros((1, 1), dtype=jnp.int32),
                128, block_size=config['block_size'], key=key)[0].tolist()))


if __name__ == '__main__':
    fire.Fire(main)
