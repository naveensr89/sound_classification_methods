from __future__ import print_function

import os
import numpy as np
from scipy.io import wavfile
import six
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

flags = tf.app.flags

flags.DEFINE_string('checkpoint', 'vggish_model.ckpt',
                    'Path to the VGGish checkpoint file.')

flags.DEFINE_string('pca_params', 'vggish_pca_params.npz',
                    'Path to the VGGish PCA parameters file.')

FLAGS = flags.FLAGS

cur_dir = os.path.dirname(os.path.abspath(__file__))

FLAGS.checkpoint = os.path.join(cur_dir, FLAGS.checkpoint)


class AudiosetEmbedding:

    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph)
            # Define the model in inference mode, load the checkpoint, and
            # locate input and output tensors.
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(self.sess, FLAGS.checkpoint)
            self.features_tensor = self.sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)
            self.embedding_tensor = self.sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME)

    def get_audioset_embeddings(self, samples, sr):
        examples_batch = vggish_input.waveform_to_examples(samples, sr)

        with self.graph.as_default():
            # Run inference and postprocessing.
            [embedding_batch] = self.sess.run(
                [self.embedding_tensor],
                feed_dict={self.features_tensor: examples_batch})
        return embedding_batch


if __name__ == "__main__":
    a = AudiosetEmbedding()
    b = np.random.rand(44100, 2)
    c = a.get_audioset_embeddings(b, 44100)
    print(c.shape)
