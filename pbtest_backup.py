import tensorflow as tf
import numpy as np
from skimage import transform, io, color
import os

test_pb_path = './final_output_stage3.pb'
raw_root = '/home/robot/Project/CPM_backup/data/test_pic/00008.bmp'
root = './'
def resize_to_imgsz(hm, img):
    """ Create Tensor for joint position prediction

    :param  hm:     Assuming input of shape (sz, w, h, c)
    :param  img:    Assuming input of shape (sz, w, h, c)
    :return:
    """
#    assert len(hm.shape) == 4 and len(img.shape) == 4
    ret_map = np.zeros((hm.shape[0], img.shape[1], img.shape[2], hm.shape[-1]), np.float32)
    for n in range(hm.shape[0]):
        for c in range(hm.shape[-1]):
            ret_map[n,:,:,c] = transform.resize(hm[n,:,:,c], img.shape[1: 3])
    return ret_map


with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()

    with open(test_pb_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        input_x = sess.graph.get_tensor_by_name("input/img_in:0")
        out_put = sess.graph.get_tensor_by_name("CPM/final_output_stage3:0")

        
        t_img = io.imread(raw_root)
        """
        #input_list.append(cv2.resize(t_img, (in_size_w, in_size_h)))
        _input = np.array(t_img)

        _input =_input.reshape((1, 560, 2448, 3))
        """
        t_img = transform.resize(t_img, (280, 1224))
        t_img = color.rgb2gray(t_img)
        _input = np.array(t_img)
        _input = _input.reshape((1, 280, 1224, 1))
        
        print('input_shape:', _input.shape)
        # pred_map = sess.run(out_put, feed_dict={input_x: _input / 255.0})[:, -1]
        pred_map = sess.run(out_put, feed_dict={input_x: _input})
        print('output_shape:', pred_map.shape)
        print('output_dtype:', pred_map.dtype)
        print('output_nbytes:', pred_map.nbytes)
        print (pred_map[0, 0, 0, 0])
        print (pred_map[0, 0, 0, 1])
        print (pred_map[0, 14, 109, 0])
        print (pred_map[0, 14, 109, 1])
        print (np.where(pred_map==np.max(pred_map)))
        print (np.max(pred_map[0, :, :, :]))
         
        r_pred_map = resize_to_imgsz(np.expand_dims(pred_map[0], 0),
                                     np.expand_dims(_input[0], 0))
        print(r_pred_map.shape)


        print (r_pred_map[0, 0, 0, 0])
        print (r_pred_map[0, 0, 0, 1])
        print (r_pred_map[0, 113, 675, 0])
        print (r_pred_map[0, 113, 675, 1])

        
       
        v_pred_map = np.squeeze(r_pred_map, axis=0)
        saveName = os.path.join(root, 'final_pb' + '-'+ '.jpg')
        io.imsave(saveName,
                  (v_pred_map[:, :, 0] * 255.).astype(np.uint8))

        """
        v_pred_map = np.sum(r_pred_map, axis=3)

        saveName = os.path.join(root, 'final_pb' + '-'+ '.jpg')
        io.imsave(saveName,
                  (np.sum(v_pred_map, axis=0) * 255.0).astype(np.uint8))
         """        
        
