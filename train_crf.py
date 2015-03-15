from functools import partial
import random
from nltk import PunktWordTokenizer
import pycrfsuite
from pycrfsuite._pycrfsuite import ItemSequence
from data_utils import yield_annotated_data, EXPERT_TRAIN_FILE, get_tokens_labels
from features import pipeline_feature_extractor, default_word2features
from processors import nltk_pos_tag_processor
from text_utils import pre_process, base_tokenize
from utils import compose_feature_vector
import os
__author__ = 'UM'

params = {
        'c1' : 0.01,
        'c2' : 0.1,
        'max_iterations' : 400,
        'feature.possible_transitions' : 1,
        'feature.possible_states' : 1
    }

MODELS_DIR = './models/'
FORWARD_MODEL_TYPE = 'fw'
BACKWARD_MODEL_TYPE = 'bw'

TEST_FILE = './data/test_file.xml'

def train_crf_model( data_generator, params, feature_extractor,
                     processors, output_model, model_type = FORWARD_MODEL_TYPE, holdout = False ):
    for bioc_info in data_generator:
        text = bioc_info['text']
        annotations = bioc_info['annotations']
        observations, lbls = get_tokens_labels( text, annotations, PunktWordTokenizer().tokenize )
        #additional processors Pos tag, word clusters, etc
        for processor in processors:
            processor( observations )
        #generating features from observations
        features = [ compose_feature_vector(observations, i, feature_extractor) for i in range(len(observations)) ]
        features = ItemSequence(features)
        #backward model
        if model_type == BACKWARD_MODEL_TYPE:
            sent_features = list(reversed( features ))
            lbls = list(reversed( lbls ))

        if holdout:
            #pick the group
            rnd_value = random.random()
            if rnd_value > 0.8:
                group_num = 1 # validate
            else:
                group_num = 0
        if holdout:
            trainer.append( features, lbls, group_num )
        else:
            trainer.append( features, lbls )

    print "Started training"
    if holdout:
       trainer.train( output_model, 1 ) # for validation
    else:
       trainer.train(output_model)
    print "Saving model to file {}".format( output_model )


def yield_annotations_tokens_idx(tokens, predictions):
    start_entity_idx = -1
    entity_buffer = []
    for (idx, ((token, start_idx, end_idx), pred)) in enumerate( zip(tokens, predictions ) ):
        if pred == 'B':
            start_entity_idx = idx
            entity_buffer.append(token)
        if pred == 'I' and start_entity_idx >= 0:
            entity_buffer.append(token)
        if pred == 'O' and start_entity_idx >= 0:
            end_entity_idx = idx
            yield start_entity_idx,end_entity_idx
            start_entity_idx = -1
            entity_buffer = []

def get_mapping(original_text):
    pre_processed_text = pre_process(original_text)
    mapping = {}

    matched_idx = 0
    i = 0
    while i < len(pre_processed_text):
        preprocessed_text_ch = pre_processed_text[i]
        if matched_idx < len(original_text):
            if preprocessed_text_ch == original_text[matched_idx]:
                mapping[i] = matched_idx
                matched_idx += 1
                i+= 1
            else:
                if preprocessed_text_ch == ' ':
                    i+= 1
                else:
                    raise Exception("weird")
        else:
            if preprocessed_text_ch == ' ':
                i += 1
    if len(mapping.values()) != len(original_text):
        raise Exception("not correct mapping")
    return mapping

def get_predictions_model(tagger, bioc_info, tokenizer, processors, feature_extractor, model_type = FORWARD_MODEL_TYPE):

    text = bioc_info['text']
    if len(text.strip()) == 0:
        print ">>>>>>>>>>>>>empty text"
        return
    pre_processed_text = pre_process(text)

    mapping = get_mapping(text)


    tokens = base_tokenize( pre_processed_text, tokenizer )
    observations = [ {'token' : t} for (t,_,_) in tokens]
    for processor in processors:
            processor( observations )
    #generating features from observations
    features = [ compose_feature_vector(observations, i, feature_extractor) for i in range(len(observations)) ]
    features = ItemSequence(features)
    if model_type == BACKWARD_MODEL_TYPE:
        features = list(reversed( features ))
    predictions = tagger.tag( features )
    if model_type == BACKWARD_MODEL_TYPE:
        predictions = list(reversed(predictions))
    #extract entities here


    for ann_token_start_idx, ann_token_end_idx in yield_annotations_tokens_idx(  tokens, predictions  ):
        token_start, token_start_pos_start, token_start_pos_end = tokens[ ann_token_start_idx ]
        not_found_correct = False
        while ann_token_end_idx >= 0:
            token_end, token_end_pos_start, token_end_pos_end = tokens[ ann_token_end_idx ]
            if token_end in ['.', '(', ',', '-', ':']:
                ann_token_end_idx -= 1
                if ann_token_end_idx < 0:
                    not_found_correct = True
                    break
            else:
                break
        if ann_token_end_idx < ann_token_start_idx or not_found_correct:
            continue

        offset_annotation = mapping[ token_start_pos_start ]
        correct_end_idx = mapping[token_end_pos_end]
        length = correct_end_idx - offset_annotation + 1
        correct_offset = offset_annotation + bioc_info['passage_offset']
        print text[offset_annotation: offset_annotation + length ]
        yield bioc_info['document_id'], correct_offset, length

    #return result

def post_process_annotations(id, offset, length, bioc_data):
    return id, offset, length

if __name__ == "__main__":
    #mapping = get_mapping("(1.2.3.4)")
    train = True
    test = True
    result_file = 'result.csv'
    #
    # 1. Training
    #
    processors = [ nltk_pos_tag_processor ]
    feature_extractors = [default_word2features]

    feature_extractor = partial(pipeline_feature_extractor, lst_feature_extractors = feature_extractors)
    expert_model_file = os.path.join(MODELS_DIR, 'crf_expert.bin')
    if train:
        trainer = pycrfsuite.Trainer(verbose=True)
        print "Using params {}".format( params)
        holdout = True
        trainer.set_params( params )


        train_data = yield_annotated_data( EXPERT_TRAIN_FILE )
        train_crf_model( train_data, params, feature_extractor, processors,
                         expert_model_file,FORWARD_MODEL_TYPE, holdout = True )
    #
    # 2. Applying model
    #
    #load model

    if test:
        with open('result_file', 'w') as f:
            tagger = pycrfsuite.Tagger()
            tagger.open(expert_model_file)

            for bioc_info in yield_annotated_data( TEST_FILE ):
                #get predictions
                for id, offset, length in get_predictions_model(tagger, bioc_info,  PunktWordTokenizer().tokenize,
                                                     processors, feature_extractor ):
                    f.write("{},{},{}\n".format( id, str(offset), str(length) ))
