from dataclasses import dataclass


@dataclass
class ExplainableSarcasticResponse(object):
    """
    Output class for the sarcasm generation pipeline.

    Args:
        response_text (`str`)

        failed_expectation (`???`)

        failure_strategy (`???`)

    """

    response_text: str
    failed_expectation: ???
    failure_strategy: ???

    def to_json(self):
        pass