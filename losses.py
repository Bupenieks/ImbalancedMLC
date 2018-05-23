import keras.backend as K 

def balanced_crossentropy(y_true, y_pred, alpha=0.5):
	"""
	y_true: (None, num_classes, 2)
	y_pred: (None, num_classes, 2)
	"""
	y_pred = K.maximum(K.minimum(y_pred, 0.9999), 1e-4)
	multiply = -y_true*K.log(y_pred)
	pos_loss = K.sum(multiply[:,:,0], axis=1)/(K.sum(y_true[:,:,0], axis=1)+1e-4)
	neg_loss = K.sum(multiply[:,:,1], axis=1)/(K.sum(y_true[:,:,1], axis=1)+1e-4)
	return pos_loss*alpha + neg_loss*(1-alpha)

def weighted_loss(y_true, y_pred, alpha=0.5):
	"""
	y_true: (None, num_classes, 2)
	y_pred: (None, num_classes, 2)
	"""
	y_pred = K.maximum(K.minimum(y_pred, 0.9999), 1e-4)
	multiply = -y_true*K.log(y_pred)
	divisor = K.sum(y_true[:,:,0] + y_true[:,:,1], axis=1) + 1e-4
	pos_loss = K.sum(multiply[:,:,0], axis=1)/divisor
	neg_loss = K.sum(multiply[:,:,1], axis=1)/divisor
	return 2*(pos_loss*alpha + neg_loss*(1-alpha))

def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true[:,:,0])
    y_pred_f = K.flatten(y_pred[:,:,0])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.):
	y_pred = K.clip(y_pred, 1e-4, 0.9999)
	inverse_y_true = y_true[:,:,1]
	inverse_y_pred = y_pred[:,:,1]
	y_pred = y_pred[:,:,0]
	y_true = y_true[:,:,0]
	mul = y_true*y_pred
	inverse_mul = inverse_y_pred*inverse_y_true
	pos_loss = ((1-mul)**gamma)*(1-alpha)*K.log(y_pred)*y_true
	neg_loss = ((1-inverse_mul)**gamma)*(alpha)*(K.log(inverse_y_pred))*inverse_y_true
	return -K.sum(pos_loss + neg_loss, axis=1)


def match_loss(name):
	if name == 'categorical_crossentropy':
		return 'categorical_crossentropy'
	elif name == 'dice_coef':
		return dice_coef_loss
	elif name == 'weighted':
		return weighted_loss
	elif name == 'balanced':
		return balanced_crossentropy
	elif name == 'focal':
		return focal_loss
	else:
		raise Exception('Not Supported Loss Fn {}'.format(name))
