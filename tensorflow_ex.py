import tensorflow as tf

# Tensor 생성
tensor = tf.random.uniform(shape=(5, 60), minval=0, maxval=1)

print(tensor)
print("Shape:", tensor.shape)
