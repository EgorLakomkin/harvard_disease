from text_utils import nltk_pos_tag_routine

__author__ = 'UM'


def pos_tag_processor(observations, pos_tagger):
    token_sequence = [ obs['token'] for obs in observations ]
    pos_tag_sequence = pos_tagger( token_sequence )
    for (obs, ( token, lemma,pos_tag ) ) in zip(observations, pos_tag_sequence) :
        obs['lemma'] = lemma
        obs['pos'] = pos_tag


def nltk_pos_tag_processor(observations,  *args, **kwargs):
    return pos_tag_processor( observations, pos_tagger= nltk_pos_tag_routine )