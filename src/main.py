import argparse
import json
import logging

from typing import List

from max import (
    ExplainableSarcasticResponse, PatternNegationExpectationExtractor,
    CommonsenseBuilderResponse, CometCommonsenseBuilder,
    PatternResponseGenerator,
    SarcasmGenerator
)


logger = logging.getLogger('main')
sarcasm_generator_logger = logging.getLogger('sarcasm_generator')


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


def main_batch(sarcasm_generator, event_file_path, output_file_path):
    with open(event_file_path, 'r', encoding='utf-8') as in_fp, \
         open(output_file_path, 'w', encoding='utf-8') as out_fp:
        for line in in_fp:
            event = line.strip()
            logger.info(f"Processing event: {event}")
            responses = sarcasm_generator.generate_responses(
                event, num_responses=1
            )[0]
            json.dump([r.to_json() for r in responses], out_fp, indent=2)


def main_interactive(sarcasm_generator):
    raise NotImplementedError


def init_logger(logger):
    logger.setLevel("INFO")
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
    "{asctime} - {name} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def main(args):
    expectation_extractor = PatternNegationExpectationExtractor.default()
    commonsense_builder = CometCommonsenseBuilder.default()
    response_generator = PatternResponseGenerator.default()
    sarcasm_generator = SarcasmGenerator(
        expectation_extractor, commonsense_builder, response_generator
    )

    if args.event_file_path is not None:
        logger.info("Entering batch mode")
        main_batch(
            sarcasm_generator, args.event_file_path, args.output_file_path
        )
    else:
        logger.info("Entering interactive mode")
        main_interactive(sarcasm_generator)


if __name__ == "__main__":
    for l in [logger, sarcasm_generator_logger]:
        init_logger(l)
    args = parse_args()
    main(args)
