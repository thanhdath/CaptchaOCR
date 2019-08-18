import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from tensorflow.python.training import monitored_session
import sys

GRAPH_PB_PATH = sys.argv[1]
with tf.Session() as sess:
    print("load graph")
    with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
        graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    image_placeholder = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
    output = tf.get_default_graph().get_tensor_by_name('AttentionOcr_v1/predicted_chars:0')
    import time
    stime = time.time()
    o = sess.run(output, {image_placeholder:np.random.randint(0, 10, size=(32, 50,200,3))})
    print(o)
    print(time.time()-stime)
