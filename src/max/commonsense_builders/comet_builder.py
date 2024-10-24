import pathlib
import sys

from typing import Tuple

import spacy
import torch
from spacy.lang.en.stop_words import STOP_WORDS

from max import CommonsenseBuilderResponse
from .sentiment_analyser import SentimentAnalyser
from .builder import CommonsenseBuilder


# This should be re-engineered
COMET_PATH = pathlib.Path(__file__).absolute().parent / 'comet'
sys.path.append(str(COMET_PATH))


from src.data import data
from src.data import config
from src.interactive import functions
from utils import utils


STOP_WORDS.add('stay')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
config.device = DEVICE


class CometCommonsenseBuilder(CommonsenseBuilder):
    def __init__(
        self, model, data_loader, text_encoder, valid_relation_types, opt,
        spacy_processor, sentiment_analyser
    ):
        self.model = model
        self.data_loader = data_loader
        self.text_encoder = text_encoder
        self.valid_relation_types = valid_relation_types
        self.opt = opt
        self.spacy_processor = spacy_processor
        self.sentiment_analyser = sentiment_analyser

    def build_commonsense(
        self,
        event: str,
        failed_expectation: str = None,
        sampling: str = 'beam-10'
    ) -> Tuple[CommonsenseBuilderResponse, CommonsenseBuilderResponse]:
        """Generates commonsense relation objects for a set of predefined
        relation types.

        Args:
            event (`str`):
                an event, i.e. a reference to an action performed by an actor,
                such as "Ben won the marathon"
            failed_expectation (`str`):
                An event that is incongruous to the input event. This event has
                failed since the input event happened.
            rel_type (`str`):
                The type of the commonsense if-then relations to consider.
                Objects of this relation will be inferred. In the current
                implementation, this can be one of 'xIntent', 'xNeed', 'xAttr',
                'xWant', 'xReact', 'xWant', 'xEffect'.
            sampling (`str`):
                the sampling algorithm to be used by the commonsense generator

        Returns:
            `Tuple[CommonsenseBuilderResponse, CommonsenseBuilderResponse]`:
                Two instances of `CommonsenseBuilderResponse`. Within one
                instance, both `event_obts` and `failed_expectation_obts`
                are dictionaries from a relation type to a list of commonsense
                objects.
                Why two instances? The first one is postprocessed; the second
                one is raw, as returned by COMET. We usually use the first one.
                The second one is for debugging purposes.
        """
        event_cs = self.build_comet_commonsense(event, sampling)

        # "raw" here refers to "without the postprocessing applied below in
        # remove_comet_overlap"
        raw_event_cs = event_cs.copy()

        if failed_expectation is not None:
            exp_cs = self.build_comet_commonsense(
                failed_expectation, sampling
            )
            raw_exp_cs = exp_cs.copy()
        else:
            exp_cs = None
            raw_exp_cs = None

        event_cs, exp_cs = self.remove_comet_overlap(event_cs, exp_cs)

        if failed_expectation is not None and exp_cs is not None:
            return (
                CommonsenseBuilderResponse(
                    event_obts=event_cs, failed_expectation_obts=exp_cs
                ),
                CommonsenseBuilderResponse(
                    event_obts=raw_event_cs, failed_expectation_obts=raw_exp_cs
                )
            )
        else:
            return (
                CommonsenseBuilderResponse(event_obts=event_cs),
                CommonsenseBuilderResponse(event_obts=raw_event_cs)
            )

    def build_comet_commonsense(self, input, sampling):
        sampler = functions.set_sampler(self.opt, sampling, self.data_loader)
        outputs = functions.get_atomic_sequence(
            input, self.model, sampler, self.data_loader, self.text_encoder,
            list(self.valid_relation_types)
        )
        outputs = {
            relation_type: relation_obt['beams']
            for relation_type, relation_obt in outputs.items()
            if relation_type in self.valid_relation_types
        }
        return outputs

    # ======================================================================== #
    # Warning: the code that follows is rather tedious.

    def preproc_obt(self, obt, R):
        toks = obt.split()
        if toks[0] in ['to', 'personx']:
            toks = toks[1:]
        if len(toks) > 1 and toks[0] == 'person' and toks[1] == 'x':
            toks = toks[2:]
        if len(toks) > 0 and toks[-1] == '.':
            toks = toks[:-1]
        if len(toks) > 0 and R not in ['xAttr', 'xReact']:
            toks = self.spacy_processor(' '.join(toks))
            vb = toks[0].lemma_ \
                if toks[0].lemma_ != '-PRON-' and toks[0].pos_ == 'VERB' \
                else toks[0].text
            toks = [vb] + [t.text for t in toks[1:]]
            # toks = [t.lemma_ if t.lemma_ != '-PRON-' and t.pos_ == 'VERB' else t.text for t in toks]
        for i in range(len(toks)):
            if toks[i] == 'their':
                toks[i] = 'your'
            elif toks[i] == 'they':
                toks[i] = 'you'
        if len(toks) > 0 and toks[0] == 'be':
            toks = ['none']
        return ' '.join(toks)

    def sents_contradict(self, s1, s2):
        return (
            self.sentiment_analyser.get_sentiment(s1)
            != self.sentiment_analyser.get_sentiment(s2)
        )

    def remove_comet_overlap(self, in_cs, exp_cs=None):
        # Preprocess and dedupe individually.
        in_cs = {
            R: obts_unique([
                self.preproc_obt(obt, R)
                for obt in obts
                if obt != 'none' and len(obt.strip()) > 0])
            for R, obts in in_cs.items()
        }
        if exp_cs is not None and len(exp_cs['xAttr']) == 0:
            exp_cs = None
        if exp_cs is not None:
            exp_cs = {
                R: obts_unique([
                    self.preproc_obt(obt, R)
                    for obt in obts
                    if obt != 'none' and len(obt.strip()) > 0])
                for R, obts in exp_cs.items()
            }

        if exp_cs is not None:
            # Get suspect objects.
            common_cs = {
                R: obts_inters(in_cs[R], exp_cs[R]) for R in in_cs.keys()
            }
            # in_cs = {R: [obt for obt in obts if not obt_in(obt, common_cs[R])]
            #          for R, obts in in_cs.items()}
            exp_cs = {
                R: [obt for obt in obts if not obt_in(obt, common_cs[R])]
                for R, obts in exp_cs.items()
            }

        if exp_cs is not None and len(exp_cs['xAttr']) == 0:
            exp_cs = None
        # Reference sentences built from xAttr obts.
        in_ref_sent = gen_sentence('xAttr', in_cs['xAttr'][:5])
        if exp_cs is not None:
            exp_ref_sent = gen_sentence('xAttr', exp_cs['xAttr'][:5])

        # Remove obts that contradict with xAttr obts.
        in_cs = {
            R: [
                obt
                for obt in obts
                if not self.sents_contradict(
                    in_ref_sent, gen_sentence(R, [obt])
                )]
            for R, obts in in_cs.items()
        }
        if exp_cs is not None:
            exp_cs = {
                R: [
                    obt
                    for obt in obts
                    if not self.sents_contradict(
                        exp_ref_sent, gen_sentence(R, [obt])
                    )]
                for R, obts in exp_cs.items()
            }

        # Remove obts that are duplicated across relations,
        # in the priority order below.
        in_acc = set()
        exp_acc = set()
        for R in ['xAttr', 'xIntent', 'xNeed', 'xReact', 'xWant', 'xEffect']:
            in_cs[R] = obts_diff(in_cs[R], in_acc)
            in_acc.update(in_cs[R])
            if exp_cs is not None:
                exp_cs[R] = obts_diff(exp_cs[R], exp_acc)
                exp_acc.update(exp_cs[R])

        return in_cs, exp_cs

    @classmethod
    def default(cls):
        valid_relation_types = {
            'xIntent', 'xNeed', 'xAttr', 'xWant', 'xReact', 'xWant', 'xEffect'
        }
        opt, state_dict = functions.load_model_file(str(
            COMET_PATH / 'pretrained_models' / 'atomic_pretrained_model.pickle'
        ))
        if opt.data.get("maxe1", None) is None:
            opt.data.maxe1 = 17
            opt.data.maxe2 = 35
            opt.data.maxr = 1
        data_loader, text_encoder = functions.load_data(
            'atomic', opt
        )
        n_ctx = data_loader.max_event + data_loader.max_effect
        n_vocab = len(text_encoder.encoder) + n_ctx
        model = functions.make_model(opt, n_vocab, n_ctx, state_dict)
        model = model.to(DEVICE)
        spacy_processor = spacy.load('en_core_web_sm')
        sentiment_analyser = SentimentAnalyser.default()
        return cls(
            model, data_loader, text_encoder, valid_relation_types, opt,
            spacy_processor, sentiment_analyser
        )


def obt_eq(o1, o2):
    o1 = ' '.join([tok for tok in o1.split() if tok not in STOP_WORDS])
    o2 = ' '.join([tok for tok in o2.split() if tok not in STOP_WORDS])
    return o1 == o2\
        or o1.startswith(o2) or o2.startswith(o1)\
        or o1.endswith(o2) or o2.endswith(o1)


def obt_in(obt, other_obts):
    return any(obt_eq(obt, other_obt) for other_obt in other_obts)


def obts_inters(obts1, obts2):
    return [obt for obt in obts1 if obt_in(obt, obts2)]


def obts_diff(obts1, obts2):
    return [obt for obt in obts1 if not obt_in(obt, obts2)]


def obts_unique(obts):
    unique = []
    for obt in obts:
        if obt != 'none' and len(obt) > 0 and not obt_in(obt, unique):
            unique.append(obt)
    return unique


def and_join(obts):
    if len(obts) == 1:
        return obts[0]

    prefix = ', '.join(obts[:-1])
    suffix = ' and ' + obts[-1]
    return prefix + suffix


def gen_sentence(R, obts):
    if R == 'xIntent':
        return 'He wanted to ' + and_join(obts) + '.'
    elif R == 'xNeed':
        return 'He decided to ' + and_join(obts) + '.'
    elif R == 'xAttr':
        return 'He is a ' + and_join(obts) + ' person.'
    elif R == 'xWant':
        return 'He wants to ' + and_join(obts) + '.'
    elif R == 'xReact':
        return 'He feels ' + and_join(obts) + '.'
    else: # R == 'xEffect':
        return 'He ' + and_join(obts) + '.'
