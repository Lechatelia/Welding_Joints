import tensorflow as tf
import numpy as np
from skimage import transform, io, color
import os

test_pb_path = 'final_preds.pb'
raw_root = 'test_images/'

IMAGE_HEIGHT = 448
IMAGE_WIDTH = 336

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()

    with open(test_pb_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        input_x = sess.graph.get_tensor_by_name("img_in:0")
        out_put = sess.graph.get_tensor_by_name("final_preds:0")

        inputs = []
        for img in os.listdir(raw_root):
            img_path = os.path.join(raw_root, img)
            print (img)
            t_img = io.imread(img_path)
            t_img = transform.resize(t_img, (448, 336))
            inputs.append(np.array(t_img))
        inputs = np.stack(inputs, axis=0) / 256.
                
        print('input_shape:', inputs.shape)
        preds = sess.run(out_put, feed_dict={input_x: inputs})
        
        print (preds)
