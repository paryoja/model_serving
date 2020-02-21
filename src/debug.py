import tensorflow as tf

x = ["test.jpg\n"]
list_ds = tf.data.Dataset.from_tensor_slices(x)


def debug_path(file_path):
    img = tf.io.read_file(file_path)

    return img


ds = list_ds.map(debug_path)

for d in ds.take(1):
    print(d)
