from nltk import PunktWordTokenizer
from text_utils import base_tokenize, pre_process

__author__ = 'UM'
from bioc import BioCReader

EXPERT_TRAIN_FILE = './data/expert1_bioc.xml'

def yield_annotated_data(xml_file):
    """

    :param xml_file:
    :return:
    """
    bioc_reader = BioCReader(xml_file)
    bioc_reader.read()
    for document in bioc_reader.collection.documents:
        document_id = document.id
        for passage in document.passages:
            text = passage.text
            passage_offset = int(passage.offset)
            annotations = passage.annotations
            for ann in annotations:
                for location in ann.locations:
                    location.offset = int(location.offset) - passage_offset
                    location.length = int(location.length)
            yield {'text' :text, 'annotations' : annotations, 'passage_offset' : passage_offset,
                   'document_id' : document_id}

def get_token_indices(tokens, offset, length):
    start_idx = -1
    end_idx = -1
    found = False
    for (t_idx, (token, t_start, t_end)) in enumerate(tokens):
        if t_start == offset:
            start_idx = t_idx
            found = True
        else:
            if found == True:
                if t_end == offset + length - 1:
                    end_idx = t_idx
    if not found:
        raise Exception("not found annotation")
    if found and end_idx == -1:
        raise Exception("not found ending")
    return start_idx, end_idx

def get_tokens_labels( text, annotations, tokenizer_func ):
    tokens = []
    labels = []
    last_processed_pos = 0
    for annotation in annotations:
        for location in annotation.locations:
            offset = location.offset
            length = location.length
            if offset > last_processed_pos:
                #firstly extract not annotated region befor annotations
                not_annotated_region = pre_process(text[ last_processed_pos: offset])
                if len(not_annotated_region.strip()) > 0:
                    not_annotated_tokens = [t for (t, _,_) in base_tokenize( not_annotated_region, tokenizer_func )]
                    tokens.extend( not_annotated_tokens )
                    labels.extend( ['O'] * len(not_annotated_tokens) )



            annotation_region = pre_process(text[offset: offset + length])
            annotation_tokens =  [t for (t, _,_) in base_tokenize( annotation_region, tokenizer_func )]
            tokens.extend( annotation_tokens )
            annotation_labels = ['B'] + ['I'] * (len(annotation_tokens) - 1)
            labels.extend( annotation_labels )
            last_processed_pos = offset + length

    if last_processed_pos < len(text) - 1:
        not_annotated_region = pre_process(text[ last_processed_pos: len(text)  ])
        not_annotated_tokens = [t for (t, _,_) in base_tokenize( not_annotated_region, tokenizer_func )]
        tokens.extend( not_annotated_tokens )
        labels.extend( ['O'] * len(not_annotated_tokens) )

    if len(tokens) != len(labels):
        raise Exception("not correct")
    return [{'token' : t} for t in tokens], labels


if __name__ == "__main__":
    for bioc_info in yield_annotated_data( EXPERT_TRAIN_FILE ):
        text = bioc_info['text']
        annotations = bioc_info['annotations']
        tokens, lbls = get_tokens_labels( text, annotations, PunktWordTokenizer().tokenize )
        print tokens, lbls