import jax
import jax.numpy as jnp
from os import path
from jax import Array
from jax.random import PRNGKey, KeyArray


class TextDataset():

    def __init__(self, train_ratio=0.9, train_samples=100):
        text = ""
        base_path = 'data/'
        source = [
            # english
            'war-and-peace.txt',
            'the-hunchback-of-notre-dame.txt',
            'tiny-shakespeare.txt',
            # chinese
            'a-dream-of-red-mansions.txt',
            'water-margin.txt',
            'three-kingdoms.txt',
            'the-journey-to-the-west.txt',
        ]
        for s in source:
            with open(path.join(base_path, s), 'r') as f:
                text += f.read()

        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.chars = chars
        self.train_ratio = train_ratio
        self.train_samples = train_samples

        data = jnp.array(self.encode(text), dtype=jnp.int32)
        base_size = int((train_ratio * len(data)) // train_samples)
        eval_base_size = int((len(data)*(1-train_ratio)) // train_samples)
        self.train_data = jnp.concatenate([
            data[i:i + base_size]
            for i in range(0, len(data), len(data) // train_samples)
        ]).flatten()
        self.eval_data = jnp.concatenate([
            data[i + base_size:i + base_size + eval_base_size]
            for i in range(0, len(data), len(data) // train_samples)
        ]).flatten()
        del data

    def encode(self, x): return [self.stoi[c] for c in x]
    def decode(self, x): return ''.join([self.itos[c] for c in x])

    def get_batch(self, batch_size: int, block_size: int, key: KeyArray, type="train"):
        if type == "train":
            data = self.train_data
        else:
            data = self.eval_data
        idx = jax.random.randint(key, (batch_size,), 0, len(data) - block_size)
        batch = jnp.stack([data[i:i + block_size] for i in idx])
        y = jnp.stack([data[i + 1:i + block_size + 1] for i in idx])
        return batch, y


if __name__ == '__main__':
    dataset = TextDataset(train_samples=10)
    print(dataset.vocab_size)
    print(dataset.train_data.shape)
    print(dataset.eval_data.shape)
    print("train data:")
    print(dataset.decode(dataset.train_data[:100].tolist()))
    print("eval data:")
    print(dataset.decode(dataset.eval_data[:100].tolist()))
