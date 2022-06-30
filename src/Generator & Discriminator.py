from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model

def downsample(filters, batch_norm=True):
  
  result = Sequential()
  initializer = tf.random_normal_initializer(0, 0.02)
  
 
  result.add(Conv2D(filters=filters,strides=2, kernel_size=4, padding='same',
                    kernel_initializer=initializer, use_bias=not batch_norm))
  #Batch
  if batch_norm:
    result.add(BatchNormalization())
  #Activation LeakyRelu
  result.add(LeakyReLU())
  return result

def upsample(filters, dropout=True):
  result = Sequential()
  initializer = tf.random_normal_initializer(0, 0.02)
  
  #Conv
  result.add(Conv2DTranspose(filters=filters,strides=2, kernel_size=4, padding='same',
                             kernel_initializer=initializer, use_bias=False))
  #Batch
  if dropout:
    result.add(Dropout(0.5))
  #Activation LeakyRelu
  result.add(ReLU())
  return result


def Generator():
  
  initializer = tf.random_normal_initializer(0, 0.02)
  
  inputs = Input(shape=[None, None, 3]) # (b, 256, 256, 64)
  
  down_stack = [
      downsample(64, batch_norm=False), # (b, 128, 128, 64)
      downsample(128), # (b, 64, 64, 128)
      downsample(256), # (b, 32, 32, 256)
      downsample(512), # (b, 16, 16, 512)
      downsample(512), # (b, 8, 8, 512)
      downsample(512), # (b, 4, 4, 512)
      downsample(512), # (b, 2, 2, 512)
      downsample(512)  # (b, 1, 1, 512)
  ]
  
  up_stack = [
      upsample(512), # (b, 2, 2, 1024)
      upsample(512), # (b, 4, 4, 1024)
      upsample(512), # (b, 8, 8, 1024)
      upsample(512, dropout=False), # (b, 16, 16, 1024)
      upsample(256, dropout=False), # (b, 32, 32, 512)
      upsample(128, dropout=False), # (b, 64, 64, 256)
      upsample(64, dropout=False), # (b, 128, 128, 128)

  ]

  last = Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer, 
                         activation='tanh')

  x = inputs
  s = []
  concat = Concatenate()
  for enc in down_stack:
    x = enc(x)
    s.append(x)
  s = reversed(s[:-1])
    
  for dec, sk in zip(up_stack, s):
    x = dec(x)
    x = concat([x, sk])    
 
  output = last(x)
  
  return Model(inputs=inputs, outputs=output)


def Discriminator():
  real_input = Input(shape=[None, None, 3], name="real_image")
  fake_input = Input(shape=[None, None, 3], name="fake_image")
  
  con = concatenate([real_input, fake_input])
  
  initializer = tf.random_normal_initializer(0, 0.02)
  
  dec1 = downsample(64, batch_norm=False)(con)
  dec2 = downsample(128)(dec1)
  dec3 = downsample(128)(dec2)
  dec4 = downsample(128)(dec3)
  
  output = Conv2D(filters=1, kernel_size=4, strides=1, kernel_initializer=initializer, padding='same')(dec4)
  return Model(inputs=[real_input, fake_input], outputs=output)