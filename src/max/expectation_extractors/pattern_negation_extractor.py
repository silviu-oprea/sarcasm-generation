import pathlib
import spacy
from pyinflect import getInflection


sp = spacy.load('en_core_web_sm')
sp.tokenizer.rules = {
    '___': [{65: '___'}],
    **{key: value for key, value in sp.tokenizer.rules.items()}
}

antonyms = {}
res_path = pathlib.Path(__file__).absolute().parent.parent.parent / 'resources'
with open(pathlib.Path(res_path) / 'antonyms.txt', 'r') as fp:
    for line in fp:
        toks = line.strip().split('\t')
        subjects = toks[0].split(',')
        objects = toks[1].split(',')
        for su in subjects:
            for ob in objects:
                antonyms[su] = ob
                antonyms[ob] = su


def pos_match(sent, pattern):
    for word, word_patterns in zip(sent[1:], pattern):
        if not isinstance(word_patterns, list):
            word_patterns = [word_patterns]

        one_matches = False
        for pattern in word_patterns:
            if all([getattr(word, key) == val for key, val in pattern.items()]):
                one_matches = True
        if not one_matches:
            return False
    return True


def extract_expectations(event, use_antonyms=False):
    """Builds possible expectations that failed such that the event happened.

    In the current rule-based implemetation, expectations are phrases that are
    roughly opposite in meaning to the input event. For instance, if the event
    is "Ben won the marathon", possivle expectations are "Ben did not win the
    marathon" or "Ben lost the marathon".

    Args: event (str): An event, i.e. a reference to an action performed by an
        actor, such as "Ben won the marathon". use_antonyms (bool): Whether to
        use antonyms in the negation process.

    Returns: list(str): A list of possible expectations.
    """
    sent = sp(event)
    expectations = []

    info = [(w.text, w.pos_, w.tag_) for w in sent]
    if (sent[0].pos_, sent[0].tag_) != ('PROPN', 'NNP'):
        raise Exception(f'Unable to extract expectation from {info}')
    subj = sent[0]

    # (AUX VBZ | AUX VBD) (not | n't) (VERB VB)
    # e.g. Ben does (AUX VBZ) not win marathons ->
    #          Ben wins marathons, Ben does win marathons,
    #          Ben does not loose marathons
    #      Ben did (AUX VBD) not win marathons ->
    #          Ben won marathons, Ben did win marathons,
    #          Ben did not loose marathons
    if pos_match(sent, [
        [{'pos_': 'AUX', 'tag_': 'VBZ'}, {'pos_': 'AUX', 'tag_': 'VBD'}],
        [{'text': 'not'}, {'text': "n't"}],
        {'pos_': 'VERB', 'tag_': 'VB'}
    ]):
        aux = sent[1]
        verb = sent[3]
        rest = [w.text for w in sent[4:]]

        # Ben wins marathons, Ben won marathons
        infls = getInflection(verb.lemma_, tag=aux.tag_)
        if infls is not None:
            toks = [subj.text, infls[0]] + rest
            expectations.append(' '.join(toks))

        # Ben does win marathons, Ben did win marathons
        toks = [subj.text, aux.text, verb.lemma_] + rest
        expectations.append(' '.join(toks))

        # Ben does not loose marathons, Ben did not loose marathons
        if use_antonyms and verb.lemma_ in antonyms:
            ant = antonyms[verb.lemma_]
            toks = [subj.text, aux.text, 'not', ant] + rest
            expectations.append(' '.join(toks))

    # (AUX VBZ | AUX VBD) (VERB VB)
    # e.g. Ben does (AUX VBZ) win marathons ->
    #          Ben does not win marathons, Ben loses marathons,
    #          Ben does loose marathons
    #          use_antonyms: Ben looses marathons
    #      Ben did (AUX VBD) win marathons ->
    #          Ben did not win marathons, Ben lost marathons,
    #          Ben did loose marathons
    elif pos_match(sent, [
        [{'pos_': 'AUX', 'tag_': 'VBZ'}, {'pos_': 'AUX', 'tag_': 'VBD'}],
        {'pos_': 'VERB', 'tag_': 'VB'}
    ]):
        aux = sent[1]
        verb = sent[2]
        rest = [w.text for w in sent[3:]]

        # Ben does not win marathons, Ben did not win marathons
        toks = [subj.text, aux.text, 'not', sent[2].lemma_] + rest
        expectations.append(' '.join(toks))

        if use_antonyms and verb.lemma_ in antonyms:
            # Ben does loose marathons, Ben did loose marathons
            toks = [subj.text, aux.text, antonyms[verb.lemma_]] + rest
            expectations.append(' '.join(toks))

            # Ben loses marathons, Ben lost marathons
            infls = getInflection(antonyms[verb.lemma_], tag=aux.tag_)
            if infls is not None:
                toks = [subj.text, infls[0]] + rest
                expectations.append(' '.join(toks))

    # (is AUX VBZ | AUX VBD) (not)
    # e.g. Ben is (AUX VBZ) not winning marathons ->
    #          Ben is winning marathons
    #      Ben was (AUX VBD) not winning marathons ->
    #          Ben was winning marathons
    elif pos_match(sent, [
        [{'text': 'is', 'pos_': 'AUX', 'tag_': 'VBZ'},
         {'text': 'was', 'pos_': 'AUX', 'tag_': 'VBD'}],
        [{'text': 'not'}, {'text': "n't"}]
    ]):
        aux = sent[1]
        rest = [w.text for w in sent[3:]]
        toks = [subj.text, aux.text] + rest
        expectations.append(' '.join(toks))

    # (is AUX VBZ | AUX VBD) (not)
    # e.g. Ben is (AUX VBZ) winning marathons ->
    #          Ben is not winning marathons
    #      Ben was (AUX VBD) winning marathons ->
    #          Ben was winning marathons
    elif pos_match(sent, [
        [{'text': 'is', 'pos_': 'AUX', 'tag_': 'VBZ'},
         {'text': 'was', 'pos_': 'AUX', 'tag_': 'VBD'}],
    ]):
        aux = sent[1]
        rest = [w.text for w in sent[2:]]
        toks = [subj.text, aux.text, 'not'] + rest
        expectations.append(' '.join(toks))

    # (VERB VBZ | VERB VBD)
    # e.g. Ben wins (VERB VBZ) marathons ->
    #          Ben does not win marathons, Ben loses marathons
    #      Ben won (VERB VBD) marathons ->
    #          Ben did not win marathons, Ben lost marathons
    elif pos_match(sent, [
        [{'pos_': 'VERB', 'tag_': 'VBZ'}, {'pos_': 'VERB', 'tag_': 'VBD'}]
    ]):
        verb = sent[1]
        rest = [w.text for w in sent[2:]]

        # Ben does not win marathons, Ben did not win marathons
        aux = 'does' if verb.tag_ == 'VBZ' else 'did'
        toks = [subj.text, aux, 'not', verb.lemma_] + rest
        expectations.append(' '.join(toks))

        if use_antonyms and verb.lemma_ in antonyms:
            infls = getInflection(antonyms[verb.lemma_], tag=verb.tag_)
            if infls is not None:
                toks = [subj.text, infls[0]] + rest
                expectations.append(' '.join(toks))

    else:
        raise Exception(f'Unable to extract expectation from {info}')

    return expectations


if __name__ == "__main__":
    events = [
        'Ben does not win marathons',
        'Ben did not win marathons',

        'Ben does win marathons',
        'Ben did win marathons',

        'Ben is winning marathons',
        'Ben is not winning marathons',
        'Ben was winning marathons',
        'Ben was not winning marathons',

        'Ben wins marathons',
        'Ben won marathons',
    ]

    for event in events:
        exp = extract_expectations(event, use_antonyms=True)
        print(event, '--->', exp)
