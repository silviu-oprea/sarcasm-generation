import nltk
import random
from nltk.corpus import sentiwordnet as swn

try:
    nltk.data.find('corpora/sentiwordnet')
except LookupError:
    nltk.download('sentiwordnet')
# list(swn.senti_synsets('happy', 'a'))[0].pos_score

# Example xIntent objects, given event P:
# - P=Ben won the marathon: to loose, to give up 1st place
# - P=Ben enjoyed the party: to be alone, to stay home
xIntent = [
    lambda obt: f'And you really tried {obt}, didn\'t you?'
]

# Example xNeed objects, given event P:
# - P=Ben won the marathon: to ignore training, to not practice
# - P=Ben enjoyed the party: to be antisocial, to not socialise
xNeed = [
    lambda obt: 'Good thing you decided ' + obt
]

# Example attributes, given event P:
# - P=Ben won the marathon: defeated, unskilled, incompetent
# - P=Ben enjoyed the party: unsocial, sad, careless
xAttr = [
    lambda obt: 'Right, you must be slightly ' + obt,
    lambda obt: 'You must be a bit ' + obt,
    lambda obt: f'You must be somewhat {obt}, are you not?',

    lambda obt: 'Oh, you ' + obt + ' person!',
    lambda obt: 'You must be ' + obt,

    lambda obt: f'You must be quite {obt}!',
    lambda obt: 'You must be very ' + obt,
    lambda obt: 'You must be terribly ' + obt,
    lambda obt: 'You are definitely ' + obt
]

# xEffect: Given event P, PersonX then:
# - P=Ben won the marathon: cries, get defeated
# - P=Ben enjoyed the party: get yelled at, get drunk, loose friends
xEffect=[ # personx then...
    lambda obt: f"You'll definitely {obt} now!",
    lambda obt: f"Don't worry if you you'll {obt} now. It's OK.",
    lambda obt: f"Ah! I think you'll {obt} now."
]

# Example attributes, given event P:
# - P=Ben won the marathon: to try harder, to not give up.
# - P=Ben enjoyed the party: to go home, to sleep
xWant=[
    lambda obt: f'You probably want to {obt}, and I support you!',
    lambda obt: f"It's understandable if you'd like to {obt} now..."
]

# Example attributes, given event P:
# - P=Ben won the marathon: sad, defeated, disappointed
# - P=Ben enjoyed the party: sad, bad, tired
xReact=[
    lambda obt: f"I'm sorry you feel so {obt}...",
    lambda obt: f'Oh, you are feeling quite {obt}, are you not?'
]

# Example attributes, given event P:
# - P=Ben won the marathon: give personx some tea
# - P=Ben enjoyed the party: drive personx home
oEffect=[
    lambda obt: f"Your friends will {obt} now."
]

# Example attributes, given event P:
# - P=Ben won the marathon: to provide consolation
# - P=Ben enjoyed the party: to help you get through
oWant=[
    lambda obt: f"Your friends probably want to {obt} now."
]

# Example attributes, given event P:
# - P=Ben won the marathon: disappointed, sad, upset
# - P=Ben enjoyed the party: to
oReact=[
    lambda obt: f"Your friends must be feeling {obt} now..."
]

rel2pats = dict(xIntent=xIntent, xNeed=xNeed, xAttr=xAttr,
                xReact=xReact, xWant=xWant, xEffect=xEffect,
                oReact=oReact, oWant=oWant, oEffect=oEffect)

def preprocess(obts):
    """Just some weird hacks for now. Will have a preprocessor for each relation
    type.
    """
    pre_obts = []
    for obt in obts:
        toks = obt.split()
        if toks[0] == 'to':
            toks = toks[1:]
        for i in range(len(toks)):
            if toks[i] == 'personx':
                toks[i] = 'you'
        pre_obts.append(' '.join(toks))
    return ', '.join(pre_obts[:-1]) + ' and ' + pre_obts[-1]


def generate_response(expectation, strategy):
    """Generates a sarcastic response given the expectation failure strategy.

    Args:
        exp (str): Expectation. Provided to extract the subject S.
        strategy (dict(
            priors -> list(tuple(T, list(O))),
            posteriors -> list(tuple(T, list(O)))
        )):
            Tuples of prior and posterior relations R and objects O, such that,
            for each O, and for each R, it is the case that R(S, O).
    """

    comments = []
    for rel, obts in strategy['priors']:
        obts = preprocess(obts)
        pat = random.choice(rel2pats[rel])
        comments.append(pat(obts))

    for rel, obts in strategy['posteriors']:
        obts = preprocess(obts)
        pat = random.choice(rel2pats[rel])
        comments.append(pat(obts))

    return comments
