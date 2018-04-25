import tensorflow as tf
import sys
sys.path.append("../")
from tcn import TemporalConvNet


class TCN(object):
    def __init__(self, input_size, output_size, num_channels, seq_len, emb_size, kernel_size=2, dropout=0.2, emb_dropout=0.2, clip_value=1.0):
        self.input_size = input_size
        self.output_size = output_size
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.emb_size = emb_size
        self.clip_value = clip_value
        self.tcn = TemporalConvNet(num_channels, stride=1, kernel_size=kernel_size, dropout=dropout)

        self._build()
        self.saver = tf.train.Saver()

    def _build(self):
        self.x = tf.placeholder(tf.int32, shape=(None, self.seq_len), name='input_chars')
        self.y = tf.placeholder(tf.int32, shape=(None, self.seq_len), name='next_chars')
        self.lr = tf.placeholder(tf.float32, shape=None, name='lr')
        self.eff_history = tf.placeholder(tf.int32, shape=None, name='eff_history')

        embedding = tf.get_variable('char_embedding', shape=(self.output_size, self.emb_size), dtype=tf.float32,
                                initializer=tf.random_uniform_initializer(-0.1, 0.1))
        inputs = tf.nn.embedding_lookup(embedding, self.x)

        outputs = self.tcn(inputs)
        reshaped_outputs = tf.reshape(outputs, (-1, self.emb_size))
        logits = tf.matmul(reshaped_outputs, embedding, transpose_b=True)

        logits = tf.reshape(logits, shape=(-1, self.seq_len, self.output_size))
        eff_logits = tf.slice(logits, [0,self.eff_history,0], [-1, -1, -1])
        eff_labels = tf.slice(self.y, [0,self.eff_history], [-1, -1])
        CE_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=eff_labels, logits=eff_logits)
        self.loss = tf.reduce_mean(CE_loss)

        optimizer = tf.train.AdamOptimizer(self.lr)
        gvs = optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -self.clip_value, self.clip_value), var) for grad, var in gvs]
        self.train_op = optimizer.apply_gradients(capped_gvs)


