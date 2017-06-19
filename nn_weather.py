import tensorflow as tf
import reader as r

class WeatherModel(object):

    def __init__(self,config,is_training):
        size = config.hidden_size
        num_steps = config.num_steps

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

        input_data = tf.get_variable("input_data", [config.batch_size, size*num_steps], dtype=tf.float32)
        self.input_data = input_data
        output_data = tf.get_variable("output_data", [config.batch_size, 3 ], dtype=tf.float32)
        self.output_data = output_data

        inputs = tf.reshape(input_data,[config.batch_size,num_steps,size])

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        outputs = []
        state = self._initial_state
        #unroll
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
        self.output = output
        # 3 - number of outputs
        softmax_w = tf.get_variable("softmax_w", [size, 3], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [3], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b

        logits = tf.reshape(logits, [config.batch_size, num_steps, 3])
        logits = logits[:,2]
        loss = tf.reduce_sum(tf.square(logits - output_data))


        self._final_state = state

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
        print(sess.run(model.loss,{model.input_data:input_train,model.output_data:output_train}))


class Config(object):
    init_scale = 0.1
    keep_prob = 1
    num_layers = 1
    hidden_size = 8
    batch_size = 2
    # num_steps - number of points used to predict
    # 3 means three points from one day
    # todo - this information must be moved to input data
    num_steps = 3

if __name__ == "__main__":
    tf.app.run()