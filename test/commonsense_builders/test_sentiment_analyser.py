import pathlib
import unittest

from max.commonsense_builders.sentiment_analyser import (
    SentimentAnalyser
)


class TestSentimentAnalyser(unittest.TestCase):
    def setUp(self):
        self.analyser = SentimentAnalyser.default()

    def test_get_sentiment(self):
        self.assertEqual(
            self.analyser.get_sentiment("Thank you very much"),
            "positive"
        )
        self.assertEqual(
            self.analyser.get_sentiment("I am very tired"),
            "negative"
        )


if __name__ == '__main__':
    unittest.main()
