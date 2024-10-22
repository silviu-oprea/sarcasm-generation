import random


def sample_list(lst):
    n_el = random.randint(0, len(lst) - 1)
    els = random.sample(lst, n_el)
    return els


def select_strategy(obs):
    """Randomly select a failure strategy of the expectation.

    In the current implementation, pragmatic insincerity is embedded by feeding
    the expectation to the commonsense generator, where the expectation is
    incongruous to the input event.
    """
    sample_obs = dict()
    for time, time_obs in obs.items():
        sample_time_obs = sample_list(time_obs)
        sample_obs[time] = sample_time_obs

    return sample_obs
