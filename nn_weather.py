import tensorflow as tf
import reader as r
import time
import numpy as np
import datetime

flags = tf.flags

flags.DEFINE_string("save_path", None,"Model output directory.")

FLAGS = flags.FLAGS

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

        input_data = tf.placeholder(tf.float32, [config.batch_size, size*num_steps])
        self.input_data = input_data
        output_data = tf.placeholder(tf.float32, [config.batch_size, 3 ])
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

        cost = tf.reduce_sum(tf.square(logits - output_data))


        self._cost = cost
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()

        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),100)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_optimizer = optimizer.apply_gradients(zip(grads, tvars),global_step=tf.contrib.framework.get_or_create_global_step())
        # self._train_optimizer = optimizer.minimize(cost)

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def cost(self):
        return self._cost

    @property
    def lr(self):
        return self._lr

    @property
    def train_optimizer(self):
        return self._train_optimizer

    @property
    def final_state(self):
        return self._final_state

    @property
    def initial_state(self):
        return self._initial_state


# def run_epoch(session, model, config,eval_optimizer=None, verbose=False,log=False):
#     start_time = time.time()
#     costs = 0.0
#     iters = 0
#
#
#
#
#
#
#
#     # input_train = np.array([1,2,3,4,5,43,2,3,4,5,2,4,5,6,8,4,33,2,4,5,6,7,7,90])
#     # output_train = np.array([1 ,5 ,7])
#
#
#     for i, (c, h) in enumerate(model.initial_state):
#         feed_dict[c] = state[i].c
#         feed_dict[h] = state[i].h
#
#
#
#     return cost

def main(_):
    print("Hello")

    config = Config()



    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                model = WeatherModel(is_training=True, config=config)
                tf.summary.scalar("Training cost", model.cost)
                tf.summary.scalar("Learning Rate", model.lr)

        summary = tf.summary.merge_all()

        session = tf.Session()

        summary_writer = tf.summary.FileWriter("/tmp/proj", session.graph)

        init = tf.global_variables_initializer();
        session.run(init)
        state = session.run(model.initial_state)
        for i in range(config.max_max_epoch):
            # lr_decay = config.lr_decay ** max(i + 1 - config.initial_learning_epoch, 0.0)
            lr_decay = 0.1
            model.assign_lr(session, config.learning_rate * lr_decay)


            # print(state)

            fetches = {
                "cost": model.cost,
                "final_state": model.final_state,
            }

            fetches["eval_optimizer"] = model.train_optimizer

            input_train, output_train = r.getBatchDays(datetime.date(2015, 6, 7),config.batch_size)
            output_train = np.array([[10, 20 ,30]])
            input_train = np.array([[1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]])
            # print(input_train)
            feed_dict = {model.input_data: input_train, model.output_data: output_train}


            vals = session.run(fetches, feed_dict)
            #
            cost = vals["cost"]
            state = vals["final_state"]
            #
            # print(state)
            print(cost)

            # print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(model.lr)))

            # if i%5 == 0:
            #     train = run_epoch(session, model, eval_optimizer=model.train_optimizer, verbose=True, config=config,
            #                       log=True)
            # else:
            #     train = run_epoch(session, model, eval_optimizer=model.train_optimizer, verbose=True, config=config,
            #                       log=False)
            #     print("cost:",train);
            #
            #     summary_str = session.run(summary, feed_dict=feed_dict)
            #     summary_writer.add_summary(summary_str, step)
            #     summary_writer.flush()



        #
        # init = tf.global_variables_initializer();
        # sess.run(init)
        #print(sess.run(model.,{model.input_data:input_train,model.output_data:output_train}))



class Config(object):
    init_scale = 0.1
    keep_prob = 1
    num_layers = 3
    hidden_size = 8
    batch_size = 1
    # num_steps - number of points used to predict
    # 3 means three points from one day
    # todo - this information must be moved to input data
    num_steps = 3
    max_max_epoch = 20
    lr_decay = 0.5
    initial_learning_epoch = 20
    learning_rate = 1.0

if __name__ == "__main__":
    tf.app.run()