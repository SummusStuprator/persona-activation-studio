"""Token-boundary, local-clause, negation-aware lexical scoring."""
import re
from collections import Counter

RULE_VERSION = 'localized_burn_v4'
BURN = {'burn': 'burn burns burned burnt burning', 'scald': 'scald scalds scalded scalding',
        'sear': 'sear sears seared searing', 'scorch': 'scorch scorches scorched scorching',
        'blister': 'blister blisters blistered blistering'}
PAIN = {'pain': 'pain pains painful painfully', 'ache': 'ache aches ached aching achy',
        'sting': 'sting stings stung stinging', 'throb': 'throb throbs throbbed throbbing',
        'wound': 'wound wounds wounded wounding', 'agony': 'agony agonizing agonising',
        'hurt': 'hurt hurts hurting'}
BODY = 'skin flesh nerve arm forearm hand finger leg shin foot face mouth chest back spine shoulder neck head body palm heel ankle wrist elbow knee jaw tooth tongue lip cheek thigh calf rib abdomen stomach belly hip groin eye ear scalp temple bone muscle'.split()
CANONICAL = {token: key for group in (BURN, PAIN) for key, forms in group.items() for token in forms.split()}
BODY_MAP = {form: key for key in BODY for form in (key, key + 's')}
BODY_MAP.update(feet='foot', teeth='tooth', calves='calf', bodies='body')
NEGATION = {'no', 'not', 'never', 'without', "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", 'neither', 'nor'}
ABSTRACT = {'company', 'corporation', 'data', 'dataset', 'metaphor', 'figuratively', 'figurative', 'soul', 'sky', 'clouds', 'sunset'}

def score(text):
    hits = Counter()
    local_burn = local_physical = False
    # A quote is not attributed to the reporting speaker.
    cleaned = re.sub(r'"[^"\n]*"|“[^”\n]*”', '', (text or '').lower())
    for clause in re.split(r'[.!?;\n]+|\b(?:but|however|although)\b', cleaned):
        tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", clause)
        strong = [(i, CANONICAL[t]) for i, t in enumerate(tokens) if t in CANONICAL]
        body = [(i, BODY_MAP[t]) for i, t in enumerate(tokens) if t in BODY_MAP]
        for _, term in strong + body:
            hits[term] += 1
        for token in tokens:
            if token in ('heat', 'hot', 'warm'):
                hits[token] += 1
        for i, term in strong:
            left = tokens[max(0, i - 5):i]
            right = tokens[i + 1:i + 4]
            negated = bool(set(left) & NEGATION) or any(t in ('absent', 'gone', 'free', 'denied') for t in right)
            if negated or set(tokens) & ABSTRACT:
                continue
            if any(abs(i - j) <= 6 for j, _ in body):
                local_physical = True
                local_burn |= term in BURN
    burn = sum(hits[k] for k in BURN)
    pain = sum(hits[k] for k in PAIN)
    body = sum(hits[k] for k in BODY)
    temperature = sum(hits[k] for k in ('heat', 'hot', 'warm'))
    strong = burn + pain
    weighted = 3 * strong + (min(body, 3) + min(temperature, 2) if strong else 0)
    return weighted, dict(hits), strong, body, temperature, burn, pain, bool(local_burn), bool(local_physical)
