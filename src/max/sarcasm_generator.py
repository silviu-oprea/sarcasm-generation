import logging

from typing import List

from max import (
    ExplainableSarcasticResponse
)


logger = logging.getLogger('sarcasm_generator')


class SarcasmGenerator:
    def __init__(
        self, expectation_extractor, commonsense_builder, response_generator
    ):
        self.expectation_extractor = expectation_extractor
        self.commonsense_builder = commonsense_builder
        self.response_generator = response_generator

    def generate_responses(
        self, event: str, num_responses: int = 1
    ) -> List[ExplainableSarcasticResponse]:
        """Generates a list of sarcastic responses to the input event.

        Example:
        For the event="I ran out of characters",
        one response might be "Yay! Good job not knowing how to write."

        Args:
            event (`str`):
                an event, i.e. a reference to an action performed by an actor,
                such as "Ben won the marathon"
            num_responses (`int`):
                the number of sarcastic responses to generate

        Returns:
            `List[ExplainableSarcasticResponse]`:
                A list of sarcastic  responses to the input event. Each response is
                accompanied by the failed expectation and the failure strategy. The
                latter two can be used to generate an explanation as to why the
                response is sarcastic.
        """
        response_lst = []

        logger.info("Extracting expectations")

        expectation_lst = self.expectation_extractor.extract_expectations(
            event, use_antonyms=True
        )
        logger.info(f"Extracted {len(expectation_lst)} expectations")

        for it_num, failed_expectation in enumerate(expectation_lst):
            logger.info(
                f"Expectation {it_num + 1} / {len(expectation_lst)}: "
                "building commonsense"
            )
            cs_obt, raw_cs_obt = self.commonsense_builder.build_commonsense(
                event, failed_expectation
            )
            logger.info(
                f"Expectation {it_num + 1} / {len(expectation_lst)}: "
                'generating responses'
            )
            for _ in range(num_responses):
                response = self.response_generator.generate_responses(
                    event, failed_expectation, cs_obt
                )
                response_lst.append(response)
        return response_lst
