import tensorflow as tf
import tensorflow.contrib.eager as tfe

EOS = '<eos>'


class Embedding(tf.keras.Model):
    def __init__(self, V, d):
        super(Embedding, self).__init__()
        self.W = tfe.Variable(tf.random_uniform(minval=-1.0, maxval=1.0, shape=[V, d]))

    def call(self, word_indexes):
        return tf.nn.embedding_lookup(self.W, word_indexes)


def create_dataset(sentences_file, vocab_table, batch_size, eos=EOS):
    # Create a Text Line dataset, which returns a string tensor
    dataset = tf.data.TextLineDataset(sentences_file)

    # Convert to a list of words..
    dataset = dataset.map(lambda sentence: tf.string_split([sentence]).values)

    # Create target words right shifted by one, append EOS, also return size of each sentence...
    dataset = dataset.map(lambda words: (words, tf.concat([words[1:], [eos]], axis=0), tf.size(words)))

    # Lookup words, word->integer, EOS->1
    dataset = dataset.map(lambda src_words, tgt_words, num_words: (
    vocab_table.lookup(src_words), vocab_table.lookup(tgt_words), num_words))

    # [None] -> src words, [None] -> tgt_words, [] length of sentence
    dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=([None], [None], []))
    return dataset


class StaticRNN(tf.keras.Model):
    def __init__(self, h, cell):
        super(StaticRNN, self).__init__()
        if cell == 'lstm':
            self.cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=h)
        elif cell == 'gru':
            self.cell = tf.nn.rnn_cell.GRUCell(num_units=h)
        else:
            self.cell = tf.nn.rnn_cell.BasicRNNCell(num_units=h)

    def call(self, word_vectors, num_words):
        word_vectors_time = tf.unstack(word_vectors, axis=1)
        outputs, final_state = tf.nn.static_rnn(cell=self.cell, inputs=word_vectors_time, sequence_length=num_words,
                                                dtype=tf.float32)
        return outputs


class LanguageModel(tf.keras.Model):
    def __init__(self, V, d, h, cell):
        super(LanguageModel, self).__init__()
        self.word_embedding = Embedding(V, d)
        self.rnn = StaticRNN(h, cell)
        self.output_layer = tf.keras.layers.Dense(units=V)

    def call(self, datum):
        word_vectors = self.word_embedding(datum[0])
        rnn_outputs_time = self.rnn(word_vectors, datum[2])

        # We want to convert it back to shape batch_size x TimeSteps x h
        rnn_outputs = tf.stack(rnn_outputs_time, axis=1)
        logits = self.output_layer(rnn_outputs)
        return logits


def loss_fun(model, datum):
    logits = model(datum)
    mask = tf.sequence_mask(datum[2], dtype=tf.float32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=datum[1]) * mask
    return tf.reduce_sum(loss) / tf.cast(tf.reduce_sum(datum[2]), dtype=tf.float32)