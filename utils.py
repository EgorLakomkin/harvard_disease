__author__ = 'UM'



def compose_feature_vector(observation, i, feature_extractor):
    res_vector = {}
    for k,v in feature_extractor(observation, i):
        res_vector[k] = v
    return res_vector