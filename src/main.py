import argparse
import json
import logging

from typing import List

from max import ExplainableSarcasticResponse
from max import (
    expectation_extractors, commonsense_builders, strategy_selectors,
    response_generators
)

logger = logging.create_logger('main')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event_file_path",
        type=str,
        help=(
            "Optional. A text file containing one event per line. An event is "
            "a text that describes an action performed by someone, e.g. "
            '"I won the marathon". If not preset will enter interactive mode.'
        )
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        help=(
            "Optional. A json file where to save the output sarcastic "
            "responses, each with the failed expectation and the failure "
            "strategy. Required if event_file_path is set."
        )
    )
    args = parser.parse_args()
    if args.event_file_path is not None:
        assert args.output_file_path is not None, (
            "If event_file_path is provided, output_file_path is also needed"
        )
    return args


def generate_sarcastic_responses(
    event: str, num_responses: int = 1
) -> List[ExplainableSarcasticResponse]:
    """Generates a list of sarcastic responses to the input event.

    Example:
      for the event="I ran out of characters"
      one response might be "Yay! Good job not knowing how to write."

    Args:
        event (str): An event, i.e. a reference to an action performed by an
            actor, such as "I ran out of characters"
        num_responses (int): The number of sarcastic responses to generate.

    Returns:
        `List[max.SarcasticResponse]`: A list of sarcastic  responses to the
            input event. Each response is accompanied by the failed expectation
            and the failure strategy. The latter two can be used to generate an
            explanation as to why the response is sarcastic.
    """
    response_lst = []

    logger.info("Extracting expectations")
    expectation_lst = expectation_extractors.pattern_negation(
        event, use_antonyms=True
    )
    logger.info(f"Extracted {len(expectation_lst)} expectations")

    for it_num, failed_expectation in enumerate(expectation_lst):
        logger.info(
            f"Expectation {it_num} / {len(expectation_lst)}: "
            "building commonsense"
        )
        object_lst = commonsense_builders.comet(expectation_lst)

        logger.info(
            f"Expectation {it_num} / {len(expectation_lst)}: "
            'selecting strategy and generating responses'
        )
        for _ in range(num_responses):
            failure_strategy = strategy_selectors.random(object_lst)
            response_text = response_generators.pattern(
                failed_expectation, failure_strategy
            )
            response_lst.append(ExplainableSarcasticResponse(
                response_text, failed_expectation, failure_strategy
            ))
    return response_lst


def main_batch(event_file_path, output_file_path):
    with open(event_file_path, 'r', encoding='utf-8') as in_fp, \
         open(output_file_path, 'w', encoding='utf-8') as out_fp:
        for line in in_fp:
            event = line.strip()
        response = generate_sarcastic_responses(event, num_responses=1)[0]
        serial = {
            "event": event,
            "response_text": response.response_text,
            "failed_expectation": response.failed_expectation,
            "failure_strategy": response.failure_strategy
        }
        out_fp.write(json.dumps(serial) + "\n")


def main_interactive():
    raise NotImplementedError


if __name__ == "__main__":
    args = parse_args()

    if args.event_file_path is not None:
        main_batch(args.event_file_path)
    else:
        main_interactive()
