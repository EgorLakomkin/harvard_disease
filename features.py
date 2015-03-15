__author__ = 'UM'

CONJUNCTION_DELIMITER = '%/%'


on_char_features = {'(' : 'OpenParen', ')' : 'CloseParen', '[' : 'OpenBracket', ']' : 'CloseBracket', '/' : 'BackSlash',
                    ',' : 'Comma', ':' : 'Colon', '.' : 'FullStop', '*' : 'Star', '%' : 'Percent', '+' : 'Plus',
                    '-' : 'Hyphen'}

SPECIAL_LIST = []

def extract_suffix(token, size = 2):
    return token[ -size : ]

def extract_prefix(token, size = 2):
    return token[ : size]

def get_orthographic_features(token):
    T = (
        'AllUpper', 'AllDigit', 'AllSymbol',
        'AllUpperDigit', 'AllUpperSymbol', 'AllDigitSymbol',
        'AllUpperDigitSymbol',
        'InitUpper',
        'AllLetter',
        'AllAlnum', 'AllLowerCase', 'EndsLowerCase', 'EndsUpperCase', 'MixCase'
        )
    R = set(T)

    for i in range(len(token)):
        c = token[i]
        if c.isupper():
            R.discard('AllDigit')
            R.discard('AllSymbol')
            R.discard('AllDigitSymbol')
            R.discard('AllLowerCase')
        elif c.isdigit() or c in (',', '.'):
            R.discard('AllUpper')
            R.discard('AllSymbol')
            R.discard('AllUpperSymbol')
            R.discard('AllLetter')
        elif c.islower():
            R.discard('AllUpper')
            R.discard('AllDigit')
            R.discard('AllSymbol')
            R.discard('AllUpperDigit')
            R.discard('AllUpperSymbol')
            R.discard('AllDigitSymbol')
            R.discard('AllUpperDigitSymbol')
        else:
            R.discard('AllUpper')
            R.discard('AllDigit')
            R.discard('AllUpperDigit')
            R.discard('AllLetter')
            R.discard('AllAlnum')

        if i == 0 and not c.isupper():
            R.discard('InitUpper')

        if i == len( token ) - 1 and c.isupper():
            R.discard('EndsLowerCase')

        if i == len( token ) - 1 and c.islower():
            R.discard('EndsUpperCase')

    if 'AllUpper' in R or 'AllLowerCase' in R:
        R.discard('MixCase')

    for tag in T:
        if tag in R:
            yield tag

def counting_features(sequence, i):
    types = set()

    token = sequence[i]['token']

    if len( token ) == 1 and token.isupper():
        types.add('SingleCapitalLetter')

    if len( token ) == 2 and token.isupper():
        types.add('TwoCapitalLetter')

    if len( token ) == 3  and token.isupper():
        types.add('ThreeCapitalLetter')

    if len( token ) > 3  and token.isupper():
        types.add('MoreThreeCapitalLetter')

    if len( token ) == 1 and token.isdigit():
            types.add('SingleDigit')

    if len( token ) == 2 and token.isdigit():
        types.add('TwoDigit')

    if len( token ) == 3  and token.isdigit():
        types.add('ThreeDigit')

    if len( token ) > 3  and token.isdigit():
        types.add('MoreThreeDigit')

    if len( token ) == 1:
        types.add('OneCharacterToken')
    if len( token ) == 2:
        types.add('TwoCharacterToken')
    if len( token ) in [3,4,5]:
        types.add('ThreeTillFiveCharacterToken')
    if len(token) >  5:
        types.add('MoreThenFixeCharacterToken')
    #add length features

    for type in types:
        yield type

def yield_n_gram( token, n_gram_size = 3 ):
    for i in range( len( token ) ):
        possible_n_gram = token[ i : i + n_gram_size ]
        if len(possible_n_gram) == 3:
            yield possible_n_gram


def morphology_features(sequence, i):


    current_token = sequence[i]['token']
    current_token_lower = current_token.lower()

    suffix_2 = extract_suffix(current_token_lower, size = 2)
    suffix_3 = extract_suffix(current_token_lower, size = 3)
    suffix_4 = extract_suffix(current_token_lower, size = 4)

    prefix_2 = extract_prefix(current_token_lower, size = 2)
    prefix_3 = extract_prefix(current_token_lower, size = 3)
    prefix_4 = extract_prefix(current_token_lower, size = 4)

    yield "morph_suffix_2" + "=" + suffix_2
    yield "morph_suffix_3" + "=" + suffix_3
    yield "morph_suffix_4" + "=" + suffix_4

    for j in range(-2,2):
        if j != 0 and i + j >=0 and i + j < len(sequence):
            for size in [2,3,4]:
                window_token = sequence[i + j]['token']
                window_token_lower = window_token.lower()
                prefix_window_j = extract_prefix( window_token_lower  , size = size)
                suffix_window_j =  extract_suffix( window_token_lower  , size = size)
                yield "morph_prefix_window" + str(j) + "_size" + str(size)+ "=" + prefix_window_j
                yield "morph_suffix_window" + str(j) + "_size" + str(size)+ "=" + suffix_window_j

    yield "morph_prefix_2" + "=" + prefix_2
    yield "morph_prefix_3" + "=" + prefix_3
    yield "morph_prefix_4" + "=" + prefix_4

    for type in get_orthographic_features( current_token ):
        yield "morph_type="+ type

    for j in range(-2,2):
        if j != 0 and i + j >=0 and i + j < len(sequence):
            token_in_window = sequence[i + j]['token']
            for type in get_orthographic_features( token_in_window ):
                yield "morph_type=@POS_{0}_{1}".format( str( j), type )

    for type in counting_features( sequence, i ):
        yield "morph_count=" + type

    for j in range(-2,2):
        if j != 0 and i + j >=0 and i + j < len(sequence):
            for type in counting_features( sequence, i + j ):
                yield "morph_count=@POS_{0}_{1}".format( str( j ), type )

    #add character n-grams

    for n_gram in yield_n_gram( current_token_lower,n_gram_size= 3 ):
        yield "n_char_gram_3=" + n_gram

    for n_gram in yield_n_gram( current_token_lower, n_gram_size= 4 ):
        yield "n_char_gram_4=" + n_gram

    for n_gram in yield_n_gram( current_token_lower, n_gram_size= 5 ):
        yield "n_char_gram_5=" + n_gram



    if current_token_lower in SPECIAL_LIST:
        yield "greek_token"

    if current_token in on_char_features:
        yield "special={}".format( on_char_features[ current_token ] )



def baseline_features(sequence, i):

    word = sequence[i]['token']
    lemma= sequence[i]['lemma']
    pos = sequence[i]['pos']

    if i == 0:
        yield'__BOS__'
    elif i == len(sequence) - 1:
        yield '__EOS__'

    yield "lemma=" + lemma.lower()
    yield "pos=" + pos
    yield "word=" + word.lower()

    if i > 0:
        prev_word, prev_lemma, prev_pos = sequence[i-1]['token'], sequence[i-1]['lemma'], sequence[i-1]['pos']
        yield "lemma-1=" + prev_lemma
        yield "pos-1=" + prev_pos
        yield "word-1=" + prev_word.lower()
    if i > 1:
        prev_prev_word, prev__prev_lemma, prev_prev_pos = sequence[i-2]['token'], sequence[i-2]['lemma'] ,sequence[i-2]['pos']
        yield "lemma-2=" + prev__prev_lemma
        yield "pos-2=" + prev_prev_pos
        yield "word-2=" + prev_prev_word.lower()

    if i + 1 < len(sequence):
        next_word, next_lemma, next_pos = sequence[ i + 1 ][ 'token' ],sequence[ i + 1 ][ 'lemma' ],sequence[ i + 1 ][ 'pos' ]
        yield "lemma+1=" + next_lemma
        yield "pos+1=" + next_pos
        yield "word+1=" + next_word.lower()

    if i + 2 < len(sequence):
        next_next_word, next_next_lemma, next_next_pos = sequence[ i + 2 ]['token'], sequence[ i + 2 ]['lemma'], sequence[ i + 2 ]['pos']
        yield "lemma+2=" + next_next_lemma
        yield "pos+2=" + next_next_pos
        yield "word+2=" + next_next_word.lower()

    #adding bigram features
    for j in xrange(-2,2):
        if i+j>=0 and i+j + 1 < len(sequence):
            bi_gram = sequence[ i + j ]['lemma'].lower() + "&&" + sequence[ i + j + 1 ]['lemma'].lower()
            yield "bi_gram={}".format( bi_gram )

def conjunction_features(sequence, i):
    templates = [ (-3, -1), (-2,-1),(-1, 0), (0, 1), ]
    for w1, w2 in templates:
        if w1 + i >=0 and w1 + i < len(sequence) and w2 + i >= 0 and w2+i < len(sequence):
            w1_word, w1_lemma, w1_pos = sequence[ w1 + i ]['token'], sequence[ w1 + i ]['lemma'] ,sequence[ w1 + i ]['pos']
            w2_word, w2_lemma, w2_pos = sequence[ w2 + i ]['token'],sequence[ w2 + i ]['lemma'],sequence[ w2 + i ]['pos']
            yield "POS={0}@{1}{2}POS_{3}@{4}".format( w1_pos, str(w1), CONJUNCTION_DELIMITER, w2_pos, str(w2) )
            yield "LEMMA={0}@{1}{2}LEMMA_{3}@{4}".format( w1_lemma.encode('utf-8'), str(w1), CONJUNCTION_DELIMITER, w2_lemma.encode('utf-8'), str(w2) )


default_pipeline = [ baseline_features,morphology_features, conjunction_features ]

def default_word2features(sent, i):
    for feature_func in default_pipeline:
       for feature in feature_func(sent, i):
           yield feature

def pipeline_feature_extractor(sequence, i, lst_feature_extractors):
    for feature_func in lst_feature_extractors:
        for f in feature_func(sequence, i):
            if isinstance(f, basestring):
                yield f, 1.0
            elif isinstance(f, tuple):
                k,v = f
                yield k, v
            else:
                raise Exception("Unknown feature {}".format( f ))