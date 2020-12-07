import pathlib
import sys
import os
import torch

comet_path = str(pathlib.Path(__file__).absolute().parent / 'comet')
sys.path.append(comet_path)

import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive
import utils.utils as utils


opt, state_dict = interactive.load_model_file(
    'commonsense_builders/comet/pretrained_models/atomic_pretrained_model.pickle'
)

if opt.data.get("maxe1", None) is None:
    opt.data.maxe1 = 17
    opt.data.maxe2 = 35
    opt.data.maxr = 1
data_loader, text_encoder = interactive.load_data(
    'atomic', opt,
    'commonsense_builders/comet/data/atomic/processed/generation',
    'commonsense_builders/comet/model'
)
n_ctx = data_loader.max_event + data_loader.max_effect
n_vocab = len(text_encoder.encoder) + n_ctx
model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)
cfg.device = "cpu"

prior_rels = {'xIntent', 'xNeed', 'xAttr'}
posterior_rels = {'xReact', 'xWant', 'xEffect', 'oReact', 'oEffect', 'oWant'}

def build_commonsense(expectation, category='all', sampling='beam-3'):
    """Generates commonsense relation objects for several relation types.

    The subject of the relation is the subject of the event that the expectation
    alludes to.

    Args:
        expectation (str): An event that alludes to the input event. It is the
            event that was expected and failed. The failure is deduced from the
            fact that the input event happened.

    Returns:
        dict(
            prior -> list(tuple(R, list(O))),
            posterior -> list(tuple(R, list())))
        ):
            If S is the subject of the relation (the subject of the input
            event), the output are tuples of prior and posterior relations R and
            objects O, such that, for each O, and for each R, it is the case
            that R(S, O).
            These relations reflect if-then inferences.
    """
    sampler = interactive.set_sampler(opt, sampling, data_loader)
    outputs = interactive.get_atomic_sequence(
        expectation, model, sampler, data_loader, text_encoder, category,
        should_print=False
    )

    priors = []
    posteriors = []

    for R in outputs.keys():
        beams = [b for b in outputs[R]['beams'] if b != 'none']
        if len(beams) > 0:
            if R in prior_rels:
                priors.append((R, beams))
            elif R in posterior_rels:
                posteriors.append((R, beams))

    return dict(priors=priors, posteriors=posteriors)


if __name__ == "__main__":
    expectations = [
        'Ben lost the marathon',
        'Ben is happy to take out the trash'
    ]
    for exp in expectations:
        objects = build_commonsense(exp)
        print(objects['priors'])
        print(objects['posteriors'])
        print()