import tensorflow as tf

IMAGE_HEIGHT = 448
IMAGE_WIDTH = 336

inputs = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='img_in')
# preds = tf.reduce_mean(inputs, axis=[1, 2, 3], name='final_preds')
# preds = tf.contrib.layers.avg_pool2d(inputs, [IMAGE_HEIGHT, IMAGE_WIDTH], stride=1,
#     padding='VALID')
# preds = tf.identity(preds[:, 0, 0, :], name='final_preds')
# preds = tf.nn.avg_pool(inputs, [1, IMAGE_HEIGHT, IMAGE_WIDTH, 1], [1, 1, 1, 1],
#     padding='VALID', name='final_preds')
preds, _ = tf.nn.moments(inputs, axes=[1, 2])
preds = tf.identity(preds, name='final_preds')

sess = tf.Session()

output_node_names = 'final_preds'

output_txt_name = 'try1.pbtxt'
output_pb_name = 'try1.pb'

tf.train.write_graph(sess.graph_def,"./",output_txt_name,as_text=True)
output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names.split(","))
tf.train.write_graph(output_graph_def,"./",output_pb_name,as_text=False)
