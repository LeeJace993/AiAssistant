from keras.models import Sequential
import numpy as np
from timeit import timeit
import tensorflow as tf
from keras.preprocessing import image
import tqdm
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import pathlib

img_rows, img_cols = 128, 128
# output image dimensions
label_rows, label_cols = 128, 128
# 创建Unet网络
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(128, 128, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
#model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding='same'))
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['binary_accuracy'])
model.summary()
def load_image_mask(
    image_path='./TongeImageDataset/groundtruth/images',
    mask_path='./TongeImageDataset/groundtruth/mask1/'
    ):
    X = []
    y = []
    path_dir = pathlib.Path(image_path)
    imgs = sorted(list(path_dir.glob("*.bmp")))
    for i in tqdm.tqdm(range(299)):
        x_1 = image.load_img(image_path + "/" + str(i+1) + ".bmp", target_size=(128, 128))
        x_1 = image.img_to_array(x_1)
        x_1 = np.expand_dims(x_1, axis=0)
        X.append(x_1)
    for i in tqdm.tqdm(range(299)):
        y_1 = image.load_img(mask_path + "/" + str(i+1) + ".bmp", target_size=(128, 128))
        y_1 = image.img_to_array(y_1)
        y_1 = np.expand_dims(y_1, axis=0)
        y.append(y_1)
    return X, y
X, y = load_image_mask()
X_1 = np.concatenate(X, axis=0)
y_1 = np.concatenate(y, axis=0)
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

X_1_gray = rgb2gray(X_1)
y_1_gray = rgb2gray(y_1)

X_1_gray = np.expand_dims(X_1_gray, axis=-1)
y_1_gray = np.expand_dims(y_1_gray, axis=-1)

np.save("./utils/Image1_gray.npy", X_1_gray)
np.save("./utils/Mask1_gray.npy", y_1_gray)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
print(X_1_gray)
print(y_1_gray.shape)
model.fit(X_1_gray, y_1_gray, batch_size=2, epochs=2500)
model.save("./utils/model_2_1.model")