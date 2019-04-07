# 在 CIFAR10 小型图像数据集上利用数据增强训练一个简单的 CNN 网络。

使用 TensorFlow 内部数据增强 API，利用 LambdaLayer 将 ImageGenerator 替换为嵌入式的 AugmentLayer，在GPU上更快。

** `ImageGenerator`(IG) vs `AugmentLayer`(AL) 的评测结果，两者都使用 augmentation 2D:**

(backend = Tensorflow-GPU, Nvidia Tesla P100-SXM2)

Epoch no. | IG %Accuracy   | IG Performance | AL %Accuracy  | AL Performance
---------:|---------------:|---------------:|--------------:|--------------:
1         | 44.84          | 15 ms/step     | 45.54         | 358 us/step
2         | 52.34          |  8 ms/step     | 50.55         | 285 us/step
8         | 65.45          |  8 ms/step     | 65.59         | 281 us/step
25        | 76.74          |  8 ms/step     | 76.17         | 280 us/step
100       | 78.81          |  8 ms/step     | 78.70         | 285 us/step

设置: horizontal_flip = True


Epoch no. | IG %Accuracy   | IG Performance | AL %Accuracy  | AL Performance
---------:|---------------:|---------------:|--------------:|--------------:
1         | 43.46          | 15 ms/step     | 42.21         | 334 us/step
2         | 48.95          | 11 ms/step     | 48.06         | 282 us/step
8         | 63.59          | 11 ms/step     | 61.35         | 290 us/step
25        | 72.25          | 12 ms/step     | 71.08         | 287 us/step
100       | 76.35          | 11 ms/step     | 74.62         | 286 us/step

设置: rotation = 30.0


(`ImageGenerator` 和 `AugmentLayer` 的转角处理和旋转精度均略有不同。)


```python
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, Lambda, MaxPooling2D
from keras import backend as K
import os

if K.backend() != 'tensorflow':
    raise RuntimeError('This example can only run with the '
                       'TensorFlow backend, '
                       'because it requires TF-native augmentation APIs')

import tensorflow as tf


def augment_2d(inputs, rotation=0, horizontal_flip=False, vertical_flip=False):
    """在 2D 数据上应用加性增强。

    # 参数
      rotation: 浮点数，旋转的度数 (0 <= rotation < 180)，
          例如， 3 表示图像在 (-3.0, 3.0) 度范围内随机旋转。
      horizontal_flip: 布尔值，是否允许随机水平翻转，
          例如， True 代表 50% 的几率水平翻转图像。
      vertical_flip: 布尔值，是否允许随机垂直翻转，
          例如， True 代表 50% 的几率垂直翻转图像。

    # 返回
      增强后的输入数据，其尺寸元原数据相同。
    """
    if inputs.dtype != tf.float32:
        inputs = tf.image.convert_image_dtype(inputs, dtype=tf.float32)

    with tf.name_scope('augmentation'):
        shp = tf.shape(inputs)
        batch_size, height, width = shp[0], shp[1], shp[2]
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)

        transforms = []
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)

        if rotation > 0:
            angle_rad = rotation * 3.141592653589793 / 180.0
            angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
            f = tf.contrib.image.angles_to_projective_transforms(angles,
                                                                 height, width)
            transforms.append(f)

        if horizontal_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            shape = [-1., 0., width, 0., 1., 0., 0., 0.]
            flip_transform = tf.convert_to_tensor(shape, dtype=tf.float32)
            flip = tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1])
            noflip = tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])
            transforms.append(tf.where(coin, flip, noflip))

        if vertical_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            shape = [1., 0., 0., 0., -1., height, 0., 0.]
            flip_transform = tf.convert_to_tensor(shape, dtype=tf.float32)
            flip = tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1])
            noflip = tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])
            transforms.append(tf.where(coin, flip, noflip))

    if transforms:
        f = tf.contrib.image.compose_transforms(*transforms)
        inputs = tf.contrib.image.transform(inputs, f, interpolation='BILINEAR')
    return inputs


batch_size = 32
num_classes = 10
epochs = 100
num_predictions = 20
save_dir = '/tmp/saved_models'
model_name = 'keras_cifar10_trained_model.h5'

# 数据，切分为训练集和测试集：
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 将类向量转换为二进制类矩阵。
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Lambda(augment_2d,
                 input_shape=x_train.shape[1:],
                 arguments={'rotation': 8.0, 'horizontal_flip': True}))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# 初始化 RMSprop 优化器
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

# 保存模型和权重
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# 评估模型
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
```
