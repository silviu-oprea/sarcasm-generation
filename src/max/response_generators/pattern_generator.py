import random

import spacy

from .generator import ResponseGenerator
from max import ExplainableSarcasticResponse


nlp = spacy.load('en_core_web_sm')


class PatternResponseGenerator(ResponseGenerator):
    def __init__(self, patterns, valid_relation_types):
        self.patterns = patterns
        self.valid_relation_types = valid_relation_types

    def generate_responses(self, event, failed_expectation, cs_obt):
        """Generate explainable sarcastic responses from the give commonsense
        object.

        Args:
            failed_expectation (`str`):
                An event that is incongruous to the input event. This event has
                failed since the input event happened.
            cs_obt (`CommonsenseBuilderResponse`):
                Instance of CommonsenseBuilderResponse from which the
                explainable sarcastic response will be constructed.

        Return:
            `ExplainableSarcasticResponse`:
                An object containing the relation, subject and object;
                the norm violated (which, in this implementation is always
                the maxim of quality), the failed expectation, and the
                response texts.
        """
        responses = []
        def _generate_responses_for_target(target, pattern_name):
            for relation_type in self.valid_relation_types:
                if len(target[relation_type]) == 0:
                    continue

                # consider the first relation object
                relation_object = target[relation_type][0]
                generator = self.patterns[pattern_name][relation_type]
                texts = generator(relation_object)

                responses.append(ExplainableSarcasticResponse(
                    event=event,
                    failed_expectation=failed_expectation,
                    relation_type=relation_type,
                    relation_subject="event",
                    relation_object=relation_object,
                    norm_violated="maxim of quality",
                    response_texts=texts
                ))
        _generate_responses_for_target(cs_obt.event_obts, "event")
        if cs_obt.failed_expectation_obts is not None:
            _generate_responses_for_target(
                cs_obt.failed_expectation_obts, "failed_expectation"
            )
        return responses

    @classmethod
    def default(cls):
        patterns = dict(
            event=dict(
                xNeed=gen_xNeed_complete,
                xAttr=gen_xAttr_complete,
                xReact=gen_xReact_complete,
                xEffect=gen_xEffect_complete
            ),
            failed_expectation=dict(
                xNeed=lambda obt: gen_xNeed_complete(obt, P=False),
                xAttr=lambda obt: gen_xAttr_complete(obt, P=False),
                xReact=lambda obt: gen_xReact_complete(obt, P=False),
                xEffect=lambda obt: gen_xEffect_complete(obt, P=False)
            )
        )
        valid_relation_types = ['xNeed', 'xAttr', 'xReact', 'xEffect']
        return cls(patterns, valid_relation_types)


interjs = ['Yay!', 'Brilliant!']
compls = ['Good job', 'Well done']
suff_intens = ['for sure']
intens = ['very']


def gen_xNeed_complete(obt, P=True):
    # Interjection and compliment:
    #   [interj: Yay!] [compliment: Good job] (P ? not :)
    #   training for the marathon.
    interj = random.choice(interjs)
    compl = random.choice(compls)
    obt_ger = get_inflection(obt, 'VBG', context='I')
    u1 = f"{interj} {compl}{' not ' if P else ' '}{obt_ger}."

    # Intensifier and compliment:
    #   You (P ? didn't :) (P ? train : trained) for the
    #     marathon, that's [inten: for sure]. [compliment: Good job!]
    suff_inten = random.choice(suff_intens)
    compl = random.choice(compls)
    if P:
        # say that precondition of P is not the case
        u2 = f"You didn't {obt}, that's {suff_inten}. {compl}!"
    else:
        # say that precondition of Q is the case
        obt_past = get_inflection(obt, 'VBD', context='I')
        u2 = f"You {obt_past}, that's {suff_inten}. {compl}!"

    return [u1, u2]


def gen_xAttr_complete(obt, P=True):
    # Interjection and intensifier:
    #   [Yay!] You're (P ? not :) [inten: very] athletic,
    #   that's [inten: for sure].
    interj = random.choice(interjs)
    inten = random.choice(intens)
    suff_inten = random.choice(suff_intens)
    u1 = f"{interj} You're{' not ' if P else ' '} {inten} {obt}, that's {suff_inten}."

    # Interjection and compliment
    # [interj: Yay!] [inten: Good job] (P ? not :) being athletic.
    interj = random.choice(interjs)
    compl = random.choice(compls)
    u2 = f"{interj} {compl}{' not ' if P else ' '}being {obt}."

    # Intensifier and compliment
    # You're not a [inten: very] athelic person, that's
    #   [inten: for sure ]. [compliment: Good job!]
    interj = random.choice(interjs)
    suff_inten = random.choice(suff_intens)
    u3 = f"{interj} You're{' not ' if P else ' '}a very {obt} person, " + \
        f"that's {suff_inten}."

    # return random.choice([u1, u2, u3])
    return [u1, u2, u3]


def gen_xReact_complete(obt, P=True):
    # Intensifier and interjection
    # You're not feeling [inten: very] happy right now, that's
    #   [inten: for sure]. [interj: Yay!]
    inten = random.choice(intens)
    suff_inten = random.choice(suff_intens)
    interj = random.choice(interjs)
    u1 = f"You're{' not ' if P else ' '}feeling {inten} {obt} right now, " + \
        f"that's {suff_inten}. {interj}"

    return [u1]


def gen_xEffect_complete(obt, P=True):
    obt_inf = get_inflection(obt, 'VB', context='he')

    # Intensifier and interjection
    # You're not [inten: really] going to become famous right now,
    #   that's [inten: for sure]. [interj: Yay!]
    interj = random.choice(interjs)
    inten = 'really'
    suff_inten = random.choice(suff_intens)
    u1 = f"You're{' not ' if P else ' '}{inten} going to {obt_inf} " + \
        f"right now, that's {suff_inten}. {interj}"
    # return random.choice([u1, u2])
    return [u1]


def get_inflection(obt, tag, context=''):
    toks = []
    found_verb = False

    for tok in nlp(context + ' ' + obt):
        if tok.pos_ in ['VERB', 'AUX'] and found_verb is False:
            found_verb = True
            toks.append(tok._.inflect(tag))
        else:
            toks.append(str(tok))
    return ' '.join(toks[len(context.split()):])