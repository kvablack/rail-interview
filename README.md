# rail-interview

## Setup
```
git clone https://github.com/kvablack/rail-interview
cd rail-interview
pip install -e .
```
Feel free to create a virtualenv/conda env as well.

## Your Task
I've set up a fake "machine learning" problem that learns to output particular values from text inputs using the policy gradient algorithm.

During each epoch, the model takes a batch of text prompts as input and produces a "sample" for each prompt (in our case, a sample is just a scalar value). It evaluates the samples using a reward function and then updates the model using the policy gradient algorithm.

The policy gradient algorithm needs an _advantage_ for each sample. The advantage is a scalar value that represents how much better (or worse) the sample is than average. Right now, the advantages are just calculated using the mean and standard deviation of the rewards from the entire batch.

The problem is that different prompts might have different reward distributions (in fact, I've designed it this way). Not taking this into account significantly increases the noise of the gradient estimates and makes learning difficult.

Your task is to implement **per-prompt stat tracking**. This means that you need to calculate the mean and standard deviation of the rewards for each prompt independently. Each batch may not have enough instances of a prompt to calculate good statistics, so you need to track these statistics across epochs. Then, you should use these statistics to calculate the advantage for each sample.

The stats change over time, so you don't want to keep really old rewards around. For each prompt, you should keep a buffer of the last `stat_tracking_buffer_size` rewards and use these to calculate the stats. Also, at the beginning of training, there won't be enough rewards per prompt to calculate accurate stats. Until there are `stat_tracking_min_count` rewards in the buffer for a given prompt, you should keep using the full-batch mean and standard deviation to calculate the advantages for that prompt.

## Testing
Run `scripts/train.py` with default arguments. Without per-prompt stat-tracking, you should achieve an average reward of around -50 after 1000 epochs. With per-prompt stat tracking, you should be able to achieve an average reward of around -10.

## What I'm Looking For
Quality over quantity! Don't just rush on to the bonuses below once it works. Make sure you write:

- Good, clean, well-formatted code. Imagine other people are going to be using this code for a long time, and posibility extending it.
- Docstrings!!!!!
- Good encapsulation (you probably shouldn't put everything in `train.py`, you can create new files if you want)

## Bonuses
In no particular order:

- Save the per-prompt stats during each epoch to `FLAGS.logdir` in some reasonable format. Write another script to load the stats and plot them over time for each prompt.
- Write another version of per-prompt stat tracking that uses an exponential moving average of rewards instead of a buffer. Compare the performance of the two versions (may require tweaking the EMA decay hyperparameter).
- Look at `rail_interview/learning.py` and explain how I set up the problem: what is the "model", what is the reward function, and how is this a policy gradient algorithm? (You don't have to write this down, just be prepared to explain it.)