from imgaug import augmenters as iaa
from PIL import Image
import numpy as np
# The following example shows an augmentation sequence that might be useful for many common experiments. It applies:
# crops and affine transformations to images, 
# flips some of the images horizontally, 
# adds a bit of noise and 
# blur and also changes the contrast as well as brightness.

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order


# Apply the augmentation to an image
def augment_image(img):
    '''
    Input type: PIL image
    output type: PIL image
    '''
    img_arr = np.array(img)
    batch_aug = seq.augment_images([img_arr])
    aug_img = Image.fromarray(batch_aug[0])
    
    return aug_img


def create_per_class_metrics(num_classes):
    funcs = '''
a = []
    '''
    for index in range(num_classes):
        s = '''
def TP_index(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true[:,:,0] * y_pred[:,:,0], 0, 1)), axis=0)[index]

def TN_index(y_true, y_pred):
    return K.sum(K.round(K.clip((1-y_true[:,:,0]) * (1-y_pred[:,:,0]), 0, 1)), axis=0)[index]

def FN_index(y_true, y_pred):
    return K.sum(K.round(K.clip((y_true[:,:,0])*(1-y_pred[:,:,0]), 0, 1)), axis=0)[index]

def FP_index(y_true, y_pred):
    return K.sum(K.round(K.clip((1-y_true[:,:,0])*(y_pred[:,:,0]), 0, 1)), axis=0)[index]

a.extend([TP_index, TN_index, FN_index, FP_index])
        '''
        funcs += (s.replace("index", str(index)))
    return funcs
