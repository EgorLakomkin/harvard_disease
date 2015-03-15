import nltk
from nltk.tokenize.punkt import PunktWordTokenizer,PunktSentenceTokenizer
__author__ = 'UM'
import re

def nltk_pos_tag_routine(tokens):
    pos_tags =  nltk.pos_tag( [t for t in tokens] )
    for (t, p) in pos_tags:
        yield (t,t,p)

def span_tokenize(sentence, tokens):
    last_token_pos = 0
    for t in tokens:
        current_token_pos = sentence.find( t, last_token_pos )
        if current_token_pos == -1:
            raise Exception("Span Tokenize cannot find token")
        last_token_pos = current_token_pos + len(t)
        yield current_token_pos, current_token_pos + len(t) - 1

def pre_process(string):
    return re.sub(r'([\.,\-\\/\'\\*"+():"])', r' \1 ', string)

def base_tokenize(sentence, tokenizer = PunktWordTokenizer().tokenize ):

    tokens = tokenizer(sentence)
    if tokens is not None and len(tokens) > 0:
        res_tokens = []
        for t_start, t_end in span_tokenize( sentence, tokens ):
            t = sentence[ t_start : t_end + 1 ]

            res_tokens.append( ( t, t_start, t_end ) )
        return res_tokens
    else:
        return None

if __name__ == "__main__":
    print pre_process("(ATH)")