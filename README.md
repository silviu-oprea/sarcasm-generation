# Sarcasm Generation

python -m spacy download en_core_web_sm

- test for pattern negation extractor

## Software workflow

The entry point is main.generate_sarcastic_response(event). It takes as input an
event, i.e. a reference to an action performed by an actor, such as "Ben won the
marathon".

To produce a sarcastic comment, the following happens:

1. Compute expectation;
2. Compute prior and posterior objects;
3. Choose a failure strategy and embed pragmatic insincerity;
4. Generate sarcastic comment.

### Compute expectation

Different strategies for computing the expectation, given the input event,
are implemented under /expectation_extractors.

In the current implementation, the expectation is the negation of the input
event. For instance, if the input is P="Ben won the marathon" the expectation
can be one of E="Ben lost the marathon" or E="Ben was 2nd in the marathon".
