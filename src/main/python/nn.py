import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, padding: str="SAME"):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)


def max_pool(x, l):
    return tf.nn.max_pool(x, ksize=[1, l, 1, 1], strides=[1, 1, 1, 1], padding='VALID')


def write_4dmat(path: str, mat):
    with open(path, "w") as f:
        for i in range(mat.size):
            f.write(str(mat.flatten(order='C')[i]))
            if i != mat.size:
                f.write(", ")
            if (i+1) % mat.shape[-1] == 0:
                f.write("&\n")
