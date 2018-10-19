import tensorflow as tf
import sys
sys.path.append("../")
from tcn import TemporalConvNet


class TCN(object):
    def __init__(self, input_size, output_size, num_channels, seq_len, emb_size, kernel_size=2, clip_value=1.0):
        self.input_size = input_size
        self.output_size = output_size
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.emb_size = emb_size
        self.clip_value = clip_value

        self._build()
        self.saver = tf.train.Saver()

    def _build(self):
        self.x = tf.placeholder(tf.int32, shape=(None, None), name='input_chars')
        self.y = tf.placeholder(tf.int32, shape=(None, None), name='next_chars')
        self.lr = tf.placeholder(tf.float32, shape=None, name='lr')
        self.eff_history = tf.placeholder(tf.int32, shape=None, name='eff_history')
        self.dropout = tf.placeholder_with_default(0., shape=())
        self.emb_dropout = tf.placeholder_with_default(0., shape=())

        embedding = tf.get_variable('char_embedding', shape=(self.output_size, self.emb_size), dtype=tf.float32,
                                initializer=tf.random_uniform_initializer(-0.1, 0.1))
        inputs = tf.nn.embedding_lookup(embedding, self.x)

        self.tcn = TemporalConvNet(self.num_channels, stride=1, kernel_size=self.kernel_size, dropout=self.dropout)
        outputs = self.tcn(inputs)
        reshaped_outputs = tf.reshape(outputs, (-1, self.emb_size))
        logits = tf.matmul(reshaped_outputs, embedding, transpose_b=True)

        logits_shape = tf.concat([tf.shape(outputs)[:2], (tf.constant(self.output_size),)], 0)
        logits = tf.reshape(logits, shape=logits_shape)
        eff_logits = tf.slice(logits, [0,self.eff_history,0], [-1, -1, -1])
        eff_labels = tf.slice(self.y, [0,self.eff_history], [-1, -1])
        CE_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=eff_labels, logits=eff_logits)
        self.loss = tf.reduce_mean(CE_loss)

        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        gvs = optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -self.clip_value, self.clip_value), var) for grad, var in gvs]
        self.train_op = optimizer.apply_gradients(capped_gvs)


