from absl import app, flags
from rail_interview.learning import get_prompts, get_reward, Model

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_epochs", 1000, "Number of epochs to train for.")
flags.DEFINE_integer("sample_batch_size", 256, "Batch size for sampling.")
flags.DEFINE_integer("train_batch_size", 16, "Batch size for training.")
flags.DEFINE_integer(
    "stat_tracking_buffer_size",
    32,
    "Number of samples per prompt in the stat tracking buffer.",
)
flags.DEFINE_integer(
    "stat_tracking_min_count",
    16,
    "Minimum number of samples in the stat tracking buffer before we start using it.",
)
flags.DEFINE_string("logdir", "logs", "Directory to write logs to.")


def main(_):
    # our "machine learning" model (holds the parameters)
    model = Model()

    for epoch in range(FLAGS.num_epochs):
        # get `batch_size` random prompts
        prompts = get_prompts(FLAGS.sample_batch_size)

        # sample from the model
        samples = model.sample(prompts)

        # compute rewards
        rewards = get_reward(prompts, samples)

        # compute advantages using mean/std of entire batch
        # TODO: replace this with per-prompt stat tracking
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # do a "train step" on our model
        model.train_step(prompts, samples, advantages)

        print(f"Epoch {epoch}: {rewards.mean():.2f} ± {rewards.std():.2f}")


if __name__ == "__main__":
    app.run(main)
