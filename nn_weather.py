import tensorflow as tf
import reader as r

class WeatherModel(object):

    def __init__(self,config,is_training):
        size = config.hidden_size
        keep_prob = config.keep_prob

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)

        attn_cell = lstm_cell

        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        #Return zero-filled state tensor, with data type
        self._initial_state = cell.zero_state(config.batch_size, tf.float32)

        inputs = tf.get_variable("data", [config.batch_size, size], dtype=tf.float32)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)
        #For testing
        self.inputs = inputs

        state = self._initial_state
        #TODO- unroll
        (cell_output, state) = cell(inputs, state)
        output = cell_output
        softmax_w = tf.get_variable("softmax_w", [size, 3], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [3], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        self.logits=logits

def main(_):
    print("Hello")

    config = Config()
    input_train,output_train = r.getRandomTrainBatch(config.batch_size)


    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-0.1,0.1)
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                model = WeatherModel(is_training=True, config=config)

        sess = tf.Session()
        init = tf.global_variables_initializer();
        sess.run(init)
        print(sess.run(model.logits,{model.inputs:input_train}))


class Config(object):
    init_scale = 0.1
    keep_prob = 0.5
    num_layers = 1
    hidden_size = 24
    batch_size = 2

if __name__ == "__main__":
    tf.app.run()