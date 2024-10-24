import csv
import pathlib

import lemminflect
import spacy

from .extractor import ExpectationExtractor


class PatternNegationExpectationExtractor(ExpectationExtractor):
    def __init__(self, antonyms_tsv_path):
        self.spacy_processor = self._init_spacy()
        self.word_to_antonym = self._read_antonyms_tsv(antonyms_tsv_path)

    def _init_spacy(self):
        spacy_processor = spacy.load('en_core_web_sm')
        spacy_processor.tokenizer.rules = {
            '___': [{65: '___'}],
            **{
                key: value
                for key, value in spacy_processor.tokenizer.rules.items()
            }
        }
        return spacy_processor

    def _read_antonyms_tsv(self, antonyms_tsv_path):
        word_to_antonym = {}
        with open(antonyms_tsv_path, "r", encoding="utf-8") as fp:
            reader = csv.DictReader(
                fp, fieldnames=["word_csv", "antonym_csv"], delimiter="\t"
            )
            for doc in reader:
                word_lst = [
                    word.strip()
                    for word in doc["word_csv"].split(",")
                ]
                antonym_lst = [
                    antonym.strip()
                    for antonym in doc["antonym_csv"].split(",")
                ]
                for word in word_lst:
                    for antonym in antonym_lst:
                        word_to_antonym[word] = antonym
                        word_to_antonym[antonym] = word
        return word_to_antonym

    def extract_expectations(self, event, use_antonyms=False):
        """Given the event, compute failed expectations.

        In the current rule-based implemetation, expectations are propositions that
        are roughly opposite in meaning to the input event. For instance, if the
        event is "Ben won the marathon", possible expectations are "Ben did not win
        the marathon" or "Ben lost the marathon".

        Args:
            event (`str`):
                an event, i.e. a reference to an action performed by an actor, such
                as "Ben won the marathon"
            use_antonyms (`bool`):
                whether to use antonyms in the process of constructing the a phrase
                that expresses a meaning opposite to the event

        Returns:
            List(`str`):
                a list of possible expectations, such as
                ["Ben did not win the marathon", "Ben lost the marathon"].
        """
        sp_obt = self.spacy_processor(event)
        expectations = []

        log_summary = [(w.text, w.pos_, w.tag_) for w in sp_obt]
        # if (sp_obt[0].pos_, sp_obt[0].tag_) != ('PROPN', 'NNP'):
        #     raise ValueError(
        #         f'Unable to extract expectation from {log_summary}'
        #     )

        if pos_match(sp_obt, [
            [{'pos_': 'AUX', 'tag_': 'VBZ'}, {'pos_': 'AUX', 'tag_': 'VBD'}],
            [{'text': 'not'}, {'text': "n't"}],
            {'pos_': 'VERB', 'tag_': 'VB'}
        ]):
            # (AUX VBZ | AUX VBD) (not | n't) (VERB VB)
            # Example: Ben does (AUX VBZ) not win marathons ->
            #             Ben wins marathons, Ben does win marathons,
            #             Ben does not loose marathons
            #         Ben did (AUX VBD) not win marathons ->
            #             Ben won marathons, Ben did win marathons,
            #             Ben did not loose marathons
            expectations.extend(self._match_aux_not_verb(sp_obt, use_antonyms))
        elif pos_match(sp_obt, [
            [{'pos_': 'AUX', 'tag_': 'VBZ'}, {'pos_': 'AUX', 'tag_': 'VBD'}],
            {'pos_': 'VERB', 'tag_': 'VB'}
        ]):
            # (AUX VBZ | AUX VBD) (VERB VB)
            # Example: Ben does (AUX VBZ) win marathons ->
            #              Ben does not win marathons, Ben loses marathons,
            #              Ben does loose marathons
            #              use_antonyms: Ben looses marathons
            #          Ben did (AUX VBD) win marathons ->
            #              Ben did not win marathons, Ben lost marathons,
            #              Ben did loose marathons
            expectations.extend(self._match_aux_verb(sp_obt, use_antonyms))
        elif pos_match(sp_obt, [
            [{'text': 'is', 'pos_': 'AUX', 'tag_': 'VBZ'},
            {'text': 'was', 'pos_': 'AUX', 'tag_': 'VBD'}],
            [{'text': 'not'}, {'text': "n't"}]
        ]):
            # (is AUX VBZ | AUX VBD) (not)
            # e.g. Ben is (AUX VBZ) not winning marathons ->
            #          Ben is winning marathons
            #      Ben was (AUX VBD) not winning marathons ->
            #          Ben was winning marathons
            expectations.extend(self._match_aux_not(sp_obt))
        elif pos_match(sp_obt, [
            [{'text': 'is', 'pos_': 'AUX', 'tag_': 'VBZ'},
            {'text': 'was', 'pos_': 'AUX', 'tag_': 'VBD'}],
        ]):
            expectations.extend(self._match_aux_not_2(sp_obt))
        elif pos_match(sp_obt, [
            [{'pos_': 'VERB', 'tag_': 'VBZ'}, {'pos_': 'VERB', 'tag_': 'VBD'},
            {'pos_': 'VERB', 'tag_': 'VBP'}]
        ]):
            # (VERB VBZ | VERB VBD)
            # e.g. Ben wins (VERB VBZ) marathons ->
            #          Ben does not win marathons, Ben loses marathons
            #      Ben won (VERB VBD) marathons ->
            #          Ben did not win marathons, Ben lost marathons
            expectations.extend(self._match_verb(sp_obt, use_antonyms))
        else:
            pass
            # raise Exception(
            #     f'Unable to extract expectation from {log_summary}'
            # )

        return expectations

    def _match_aux_not_verb(self, sp_obt, use_antonyms):
        """(AUX VBZ | AUX VBD) (not | n't) (VERB VB)

        Example: Ben does (AUX VBZ) not win marathons ->
                     Ben wins marathons, Ben does win marathons,
                     Ben does not loose marathons
                 Ben did (AUX VBD) not win marathons ->
                     Ben won marathons, Ben did win marathons,
                     Ben did not loose marathons
        """
        subj = sp_obt[0]
        aux = sp_obt[1]
        verb = sp_obt[3]
        rest = [w.text for w in sp_obt[4:]]
        expectations = []

        # Ben wins marathons, Ben won marathons
        infls = lemminflect.getInflection(verb.lemma_, tag=aux.tag_)
        if infls is not None:
            toks = [subj.text, infls[0]] + rest
            expectations.append(' '.join(toks))

        # Ben does win marathons, Ben did win marathons
        toks = [subj.text, aux.text, verb.lemma_] + rest
        expectations.append(' '.join(toks))

        # Ben does not loose marathons, Ben did not loose marathons
        if use_antonyms and verb.lemma_ in self.word_to_antonym:
            ant = self.word_to_antonym[verb.lemma_]
            toks = [subj.text, aux.text, 'not', ant] + rest
            expectations.append(' '.join(toks))
        return expectations

    def _match_aux_verb(self, sp_obt, use_antonyms):
        """(AUX VBZ | AUX VBD) (VERB VB)
        Example: Ben does (AUX VBZ) win marathons ->
                     Ben does not win marathons, Ben loses marathons,
                     Ben does loose marathons
                     use_antonyms: Ben looses marathons
                 Ben did (AUX VBD) win marathons ->
                     Ben did not win marathons, Ben lost marathons,
                     Ben did loose marathons

        """
        subj = sp_obt[0]
        aux = sp_obt[1]
        verb = sp_obt[2]
        rest = [w.text for w in sp_obt[3:]]
        expectations = []

        # Ben does not win marathons, Ben did not win marathons
        toks = [subj.text, aux.text, 'not', sp_obt[2].lemma_] + rest
        expectations.append(' '.join(toks))

        if use_antonyms and verb.lemma_ in self.word_to_antonym:
            # Ben does loose marathons, Ben did loose marathons
            toks = [subj.text, aux.text, self.word_to_antonym[verb.lemma_]] + rest
            expectations.append(' '.join(toks))

            # Ben loses marathons, Ben lost marathons
            infls = lemminflect.getInflection(
                self.word_to_antonym[verb.lemma_], tag=aux.tag_
            )
            if infls is not None:
                toks = [subj.text, infls[0]] + rest
                expectations.append(' '.join(toks))
        return expectations

    def _match_aux_not(self, sp_obt):
        """(is AUX VBZ | AUX VBD) (not)
        Example: Ben is (AUX VBZ) not winning marathons ->
                     Ben is winning marathons
                 Ben was (AUX VBD) not winning marathons ->
                     Ben was winning marathons
        """
        subj = sp_obt[0]
        aux = sp_obt[1]
        rest = [w.text for w in sp_obt[3:]]
        toks = [subj.text, aux.text] + rest
        expectations = []
        expectations.append(' '.join(toks))
        return expectations

    def _match_aux_not_2(self, sp_obt):
        """(is AUX VBZ | AUX VBD) (not)
        Example: Ben is (AUX VBZ) winning marathons ->
                     Ben is not winning marathons
                 Ben was (AUX VBD) winning marathons ->
                     Ben was winning marathons

        """
        subj = sp_obt[0]
        aux = sp_obt[1]
        rest = [w.text for w in sp_obt[2:]]
        toks = [subj.text, aux.text, 'not'] + rest
        expectations = []
        expectations.append(' '.join(toks))
        return expectations

    def _match_verb(self, sp_obt, use_antonyms):
        """(VERB VBZ | VERB VBD)
        Example: Ben wins (VERB VBZ) marathons ->
                     Ben does not win marathons, Ben loses marathons
                 Ben won (VERB VBD) marathons ->
                     Ben did not win marathons, Ben lost marathons

        """
        subj = sp_obt[0]
        verb = sp_obt[1]
        rest = [w.text for w in sp_obt[2:]]
        expectations = []

        # Ben does not win marathons, Ben did not win marathons
        aux = 'does' if verb.tag_ == 'VBZ' else 'did'
        toks = [subj.text, aux, 'not', verb.lemma_] + rest
        expectations.append(' '.join(toks))

        if use_antonyms and verb.lemma_ in self.word_to_antonym:
            infls = lemminflect.getInflection(
                self.word_to_antonym[verb.lemma_], tag=verb.tag_
            )
            if infls is not None:
                toks = [subj.text, infls[0]] + rest
                expectations.append(' '.join(toks))
        return expectations

    @classmethod
    def default(cls):
        antonyms_tsv_path = (
            # /src/max/expectation_extractors/ -> /
            pathlib.Path(__file__).absolute().parent.parent.parent.parent
            / 'resources' / "antonyms.tsv"
        )
        return cls(antonyms_tsv_path)


def pos_match(sp_obt, pattern):
    """Check if a given pattern matches the provided spacy output object.
    """
    for word, word_patterns in zip(sp_obt[1:], pattern):
        if not isinstance(word_patterns, list):
            word_patterns = [word_patterns]

        one_matches = False
        for pattern in word_patterns:
            if all([getattr(word, key) == val for key, val in pattern.items()]):
                one_matches = True
        if not one_matches:
            return False
    return True
