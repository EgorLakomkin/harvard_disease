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

def get_predictions_model(model, text, tokenizer, processors, feature_extractor, model_type = FORWARD_MODEL_TYPE):
    text = pre_process(text)
    tokens = [t for (t, _,_) in base_tokenize( text, tokenizer )]
    observations = [ {'token' : t} for t in tokens]
    for processor in processors:
            processor( observations )
    #generating features from observations
    features = [ compose_feature_vector(observations, i, feature_extractor) for i in range(len(observations)) ]
    features = ItemSequence(features)
    if model_type == BACKWARD_MODEL_TYPE:
        features = list(reversed( features ))
    predictions = model.tag( features )
    if model_type == BACKWARD_MODEL_TYPE:
        predictions = list(reversed(predictions))
    #extract entities here

    #return result

if __name__ == "__main__":
    train = False
    #
    # 1. Training
    #
    expert_model_file = os.path.join(MODELS_DIR, 'crf_expert.bin')
    if train:
        trainer = pycrfsuite.Trainer(verbose=True)
        print "Using params {}".format( params)
        holdout = True
        trainer.set_params( params )

        processors = [ nltk_pos_tag_processor ]
        feature_extractors = [default_word2features]

        feature_extractor = partial(pipeline_feature_extractor, lst_feature_extractors = feature_extractors)
        train_data = yield_annotated_data( EXPERT_TRAIN_FILE )
        train_crf_model( train_crf_model, params, feature_extractor, processors,
                         os.path.join(MODELS_DIR, 'crf_expert.bin'),FORWARD_MODEL_TYPE, holdout = True )
    #
    # 2. Applying model
    #
    #load model
    model = None
    for bioc_info in yield_annotated_data( TEST_FILE ):
        text = bioc_info['text']
        #get predictions
        annotations = get_predictions_model(model, text,  PunktWordTokenizer().tokenize,
                                             feature_extractor, processors)
