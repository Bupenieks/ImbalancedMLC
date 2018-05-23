import os
from keras.optimizers import SGD
from losses import match_loss
###################
#     CONFIG      #
###################

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Model parameters
params = """
epochs = 500
steps_per_epoch = 100
val_steps = 50
learning_rate = 1e-2
momentum = 0.9
batch_size = 24
nesterov = True
reg_const = 1e-4

# Learning Rate Decay Parameters
monitor='val_loss'
factor = 0.5
patience = 2
epsilon = 0.01
min_lr = 1e-6

shuffle = False
balanced_mini_batches = True

# Training parameters
loss = match_loss('categorical_crossentropy')
metrics = ['accuracy']
optimizer = SGD(lr=learning_rate, momentum=momentum, nesterov=nesterov)

output_folder = 'Balanced_Batches_Categorical_Crossentropy'
train_annotations_file = 'annotations/train.json'
val_annotations_file = 'annotations/val.json'
"""

exec(params)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
with open(os.path.join(output_folder, 'params.json'), 'w') as f:
    f.write(params)

# Load model weights
LOAD_WEIGHTS = False
WEIGHT_PATH = '.'

SLACK_WEBHOOK_URLS = [
    'https://hooks.slack.com/services/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', # Ben
    #'https://hooks.slack.com/services/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', # Agastya
]

slack_end_msg = "{} finished training.".format(output_folder.strip('/').strip('.'))
slack_start_msg = "Started training {}.".format(output_folder.strip('/').strip('.'))

###################
###################

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

import sys, keras, random, time, json#, pylab
sys.path.append(os.path.join(os.getcwd(), '..'))

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.xception import Xception
from keras.layers import Activation, Dense, GlobalAveragePooling2D, Reshape
from keras.models import Model
import keras.backend as K
from keras.utils import to_categorical
import numpy as np
import skimage.io as io
from collections import defaultdict
import xml.etree.ElementTree as ET
from PIL import Image
from train_utils.utils import augment_image, create_per_class_metrics
import requests

from train_utils.callbacks import BatchTimer, SaveCustomMetrics, SlackCallback
from keras.applications.xception import preprocess_input

# Get Categories
all_categories = []
with open('annotations/categories.txt', 'r') as c:
    all_categories = [x.strip('\n') for x in c.readlines()]
NUM_CLASSES = len(all_categories)
print '{} Classes'.format(NUM_CLASSES)

# Init data generator and create binary y vectors
def load_fn_x(x):
    img = Image.open(x)
    img = img.resize((224,224))
    img = augment_image(img)
    return np.array(img)

def multihot(y):
    to_index = lambda i: all_categories.index(i)
    y_new = map(to_index, y)
    maxxed = np.max(to_categorical(y_new, NUM_CLASSES), axis=0)
    return np.array([maxxed, 1-maxxed]).T


# Create data generator
def create_generator(x, y, batch_size=32, load_fn_x=Image.open, load_fn_y=lambda y:y, shuffle=True):
    if len(x) != len(y):
        raise ValueError("Length of x and y are not equal.")

    x_data, y_data = x, y
    if shuffle:
        zipped = zip(x,y)
        random.shuffle(zipped)
        x_data, y_data = zip(*zipped) # Unzip

    index = 0
    while True:
        batch_x, batch_y = [], []
        for _ in range(batch_size):
            try:
                index += 1
                index %= len(x_data)
                img = load_fn_x(x_data[index])
                if np.array(img).shape == (224,224):
                    continue

                batch_x.append(img)
                batch_y.append(load_fn_y(y_data[index]))

            except IOError as e:
                print e
                continue
#         batch_y = list(np.expand_dims(np.array(batch_y).T, axis=-1))
#         batch_x = [preprocess_input(np.array(x).astype(float)) for x in batch_x]
        yield (preprocess_input(np.array(batch_x).astype(float)), np.array(batch_y).astype(float))

def create_balanced_generator(x_data, y_data, batch_size=32, load_fn_x=Image.open, load_fn_y=lambda y:y, shuffle=True):
    assert len(x_data) == len(y_data)
    assert batch_size >= NUM_CLASSES
           
    label_to_image_map = defaultdict(list)
    for instance, labels in zip(x_data,y_data):
        for l in labels:
            label_to_image_map[l].append((instance, labels))

                    
    labels = label_to_image_map.keys()
            
    while True:
        random.shuffle(labels)
        batch_x, batch_y = [], []
        used_instances = set()
        for i in range(batch_size):
            l = labels[i % len(labels)]      
            try:
                rand_x, rand_y = random.choice(label_to_image_map[l])
                while rand_x in used_instances:
                    rand_x, rand_y = random.choice(label_to_image_map[l])
                    
                used_instances.add(rand_x)

                img = load_fn_x(rand_x)
                if np.array(img).shape == (224,224):
                    continue

                batch_x.append(img)
                batch_y.append(load_fn_y(rand_y))

            except IOError as e:
                print e
                continue
                
        yield (preprocess_input(np.array(batch_x).astype(float)), np.array(batch_y).astype(float))
     
    
def train():
    # Get train val data
    with open(train_annotations_file, 'r') as train_json, open('annotations/val.json', 'r') as val_json:
        train_to_cat_map = json.load(train_json)
        val_to_cat_map = json.load(val_json)

    # Create metrics
    exec(create_per_class_metrics(NUM_CLASSES))
    metrics.extend(a)

    # Build and compile model
    xception = Xception(
        include_top=False,
        weights='imagenet',
        input_shape=None)

    dense_input = GlobalAveragePooling2D()(xception.output)
    dout = Dense(NUM_CLASSES*2, activation='linear', name="fc_output")(dense_input)
    dout = Reshape((NUM_CLASSES, 2), name='dense_output')(dout)
    dout = Activation('softmax')(dout)

    model = Model([xception.input], [dout])

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        reg_const=reg_const
    )

    if LOAD_WEIGHTS:
        model.load_weights(WEIGHT_PATH)

    # Select correct generator
    train_gen_func = create_generator
    x_train_load = load_fn_x
    y_train_load = multihot
    
    if balanced_mini_batches:
        train_gen_func = create_balanced_generator
        
    # Create generators
    x_train, y_train = zip(*train_to_cat_map.items())
    train_data_gen = train_gen_func(
        x_train,
        y_train,
        batch_size,
        load_fn_x=x_train_load,
        load_fn_y=y_train_load,
        shuffle=shuffle
    )

    x_val, y_val = zip(*val_to_cat_map.items())
    val_data_gen = create_generator(
        x_val,
        y_val,
        batch_size,
        load_fn_x=load_fn_x,
        load_fn_y=multihot,
        shuffle=shuffle
    )

    # Custom callbacks
    batch_timer = BatchTimer(output_folder)
    custom_logger = SaveCustomMetrics(output_folder)
    slack_call = SlackCallback(SLACK_WEBHOOK_URLS, slack_start_msg, slack_end_msg)

    # LR Decay
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=factor,
        patience=patience,
        verbose=1,
        epsilon=epsilon,
        min_lr=min_lr
    )

    model.fit_generator(
        generator=train_data_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=val_steps,
        verbose=0,
        folder=output_folder,
 	callbacks=[batch_timer, reduce_lr, slack_call],
        overwrite=True
    )
        
try:
    train()
except BaseException as e:
    print e
    for url in SLACK_WEBHOOK_URLS:
        r = requests.post(url, json={'text': 'Train script failed with exception {}'.format(e)})
        print(r.status_code, r.reason)
