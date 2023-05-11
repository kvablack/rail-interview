import randomname
import random
import numpy as np

PROMPTS = [randomname.get_name() for _ in range(64)]


def get_prompts(batch_size):
    return random.choices(PROMPTS, k=batch_size)


def get_reward(prompts, samples):
    labels = np.array([hash("_" + p) % 256 for p in prompts])
    return -np.abs(samples - labels)


class Model:
    def __init__(self):
        self.params = np.random.uniform(0, 256, 1024)

    def sample(self, prompts):
        input_ids = np.array([hash(p) % 1024 for p in prompts])
        mean = self.params[input_ids]
        std = np.array([hash(p) % 1023 for p in prompts]) / 256
        return np.random.normal(mean, std)

    def train_step(self, prompts, samples, advantages):
        input_ids = np.array([hash(p) % 1024 for p in prompts])
        mean = self.params[input_ids]
        self.params[input_ids] += advantages * (samples - mean)
