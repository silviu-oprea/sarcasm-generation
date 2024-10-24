import pathlib
import unittest

from max.expectation_extractors.pattern_negation_extractor import (
    PatternNegationExpectationExtractor
)


event_and_expectations = [
    ('Ben does not win marathons', [
        'Ben wins marathons', 'Ben does win marathons'
    ]),
    ('Ben did not win marathons', [
        'Ben won marathons', 'Ben did win marathons'
    ]),

    ('Ben does win marathons', [
        'Ben does not win marathons'
    ]),
    ('Ben did win marathons', [
        'Ben did not win marathons'
    ]),

    ('Ben is winning marathons', [
        'Ben is not winning marathons'
    ]),
    ('Ben is not winning marathons', [
        'Ben is winning marathons'
    ]),
    ('Ben was winning marathons', [
        'Ben was not winning marathons'
    ]),
    ('Ben was not winning marathons', [
        'Ben was winning marathons'
    ]),

    ('Ben wins marathons', [
        'Ben does not win marathons'
    ]),
    ('Ben won marathons', [
        'Ben did not win marathons'
    ]),

    ('I ran out of characters', [
        'I did not run out of characters'
    ])
]

class TestPatternNegationExpectationExtractor(unittest.TestCase):
    def setUp(self):
        antonyms_ts_path = (
            pathlib.Path(__file__).absolute().parent.parent.parent
            / 'resources' / "antonyms.tsv"
        )
        self.extractor = PatternNegationExpectationExtractor(antonyms_ts_path)

    def test_extract_expectations(self):
        for event, target_expectations in event_and_expectations:
            pred_expectations = self.extractor.extract_expectations(event)
            self.assertListEqual(pred_expectations, target_expectations)


if __name__ == '__main__':
    unittest.main()
