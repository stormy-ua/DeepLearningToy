import re
from PIL import Image
from io import BytesIO
import base64
import numpy as np
from skimage import color
from skimage import io
import scipy.misc as sp
import matplotlib.pyplot as plt
import dill as pickle

import sys
sys.path.append('/Users/kirill/Documents/Projects/DeepLearningToy/src')

from pydeeptoy.networks import *
from pydeeptoy.optimizers import *
from pydeeptoy.losses import *


def recognize_digit(base64image):
    image_data = re.sub('^data:image/.+;base64,', '', base64image)
    im = Image.open(BytesIO(base64.b64decode(image_data)))
    image_array = np.asarray(im)
    image_array = np.sum(image_array, axis=2)
    image_array = sp.imresize(image_array, (28, 28))
    plt.imshow(image_array, cmap='gray')

    estimator = pickle.load(open("digit.recognizer.mlp.estimator", "rb"))
    digit = estimator.predict(image_array.reshape(-1))
    return digit
