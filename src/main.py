import logging

from slib import logs

from expectation_extractors.pattern_negation_extractor import extract_expectations
from commonsense_builders.comet_builder import build_commonsense
from strategy_selectors.random_selector import select_strategy
from generators.pattern_generator import generate_response

logger = logs.create_logger('main')


def generate_sarcastic_responses(event, n_resp):
    """Generates a list of sarcastic responses to the input event.

    Args:
        event (str): An event, i.e. a reference to an action performed by an
            actor, such as "Ben won the marathon".
        n_resp (int): The number of sarcastic responses to generate.

    Returns:
        list[tuple[expectation, strategy, response]]: A list of sarcastic
            responses to the input event, along with the expectation and the
            (expectation) failure strategy. The latter two can be used to
            generate an explanation as to why the response is sarcastic.
    """
    resp = []

    logger.info('Extracting expectations')
    exps = extract_expectations(event, use_antonyms=True)
    logger.info('Extracted %d expectations', len(exps))

    for it_num, exp in enumerate(exps, start=1):
        logger.info('Expectation %s / %s: building commonsense',
                    it_num, len(exps))
        objects = build_commonsense(exp)

        logger.info('Expectation %s / %d: '
                    'selecting strategy and generating responses',
                    it_num, len(exps))
        for _ in range(n_resp):
            strategy = select_strategy(objects)
            text = generate_response(exp, strategy)
            resp.append((exp, strategy, text))

    return resp


EVENT = 'Ben won the marathon'
responses = generate_sarcastic_responses(EVENT, 1)
for response in responses:
    expectation, strategy, texts = response
    print('Event:', EVENT)
    print('\tExpectation:', expectation)
    print('\tStrategy:', strategy)
    print('\tText:', '. '.join(texts))