import tensorflow as tf
import os
import sys
import resnet


slim = tf.contrib.slim

"""
model_dir = 'test_pb/'
model_name = 'mymodel.pb'
source_model = model_dir + 'model.ckpt-0'
"""
model_dir = 'logs_0715'
source_model = os.path.join(model_dir, 'model_ckpt-38900')

IMAGE_HEIGHT = 448
IMAGE_WIDTH = 336
NUM_RESIDUAL_BLOCKS = 5

def freeze_mobilenet(meta_file):

    tf.reset_default_graph()
    
    inputs = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='img_in')
    logits = resnet.inference(inputs, NUM_RESIDUAL_BLOCKS, reuse=False)
    output = tf.identity(logits, name='final_preds')
    output_node_names = 'final_preds'
    
    output_txt_name = output_node_names + '.pbtxt'
    output_pb_name = output_node_names + '.pb'
        
    rest_var = slim.get_variables_to_restore()
    with tf.Session() as sess:
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        
        saver = tf.train.Saver(rest_var)
        saver.restore(sess, meta_file)
  
        tf.train.write_graph(sess.graph_def,"./",output_txt_name,as_text=True)
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names.split(","))
        tf.train.write_graph(output_graph_def,"./",output_pb_name,as_text=False)
        
freeze_mobilenet(source_model)


