import pandas as pd
import numpy as np
import re as re

from features.base import Feature, get_arguments, generate_features

Feature.dir = 'features'



if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_feather('./data/input/train.feather')
    test = pd.read_feather('./data/input/test.feather')

    generate_features(globals(), args.force)