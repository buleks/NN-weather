import tensorflow as tf
import reader as r
import time
import numpy as np
import datetime
import itertools

flags = tf.flags

flags.DEFINE_string("save_path", None,"Model output directory.")
flags.DEFINE_string("state_path", None,"Model state directory.")

FLAGS = flags.FLAGS

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

class WeatherModel(object):

    def __init__(self,config,is_training):
        size = config.hidden_size
        num_steps = config.num_days*3

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)

        attn_cell = lstm_cell

        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        input_batchsize = tf.placeholder(tf.int32,[1])
        self.input_batchsize = input_batchsize

        #Return zero-filled state tensor, with data type
        self._initial_state = cell.zero_state(input_batchsize[0], tf.float32)

        input_data = tf.placeholder(tf.float32, [None  , size*num_steps])
        self.input_data = input_data
        output_data = tf.placeholder(tf.float32, [None, 3 ])
        self.output_data = output_data

        inputs = tf.reshape(input_data,[-1,num_steps,size])

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
        with tf.name_scope('weights_linear'):
            softmax_w = tf.get_variable("softmax_w", [size, 3], dtype=tf.float32)
            variable_summaries(softmax_w)
        with tf.name_scope('biases_linear'):
            softmax_b = tf.get_variable("softmax_b", [3], dtype=tf.float32)
            variable_summaries(softmax_b)

        logits = tf.matmul(output, softmax_w) + softmax_b

        logits = tf.reshape(logits, [-1, num_steps, 3])
        logits = logits[:,2]

        cost = tf.reduce_mean(tf.square(logits - output_data))

        self._logits = logits

        self._cost = cost
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()

        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),100)
        # optimizer = tf.train.GradientDescentOptimizer(self._lr)
        optimizer = tf.train.AdamOptimizer(self._lr)
        self._train_optimizer = optimizer.apply_gradients(zip(grads, tvars),global_step=tf.contrib.framework.get_or_create_global_step())
        # self._train_optimizer = optimizer.minimize(cost)

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)


        self._accuracy = tf.reduce_mean(logits - output_data)

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

    @property
    def logits(self):
        return self._logits

    @property
    def accuracy(self):
        return self._accuracy


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

        saver = tf.train.Saver()
        # session = tf.Session()

        init = tf.global_variables_initializer()


        # state = session.run(model.initial_state)
        with tf.Session() as session:
            if FLAGS.state_path:
                print("Loading model from %s." % FLAGS.state_path)
                saver.restore(session, FLAGS.state_path)
            else:
                session.run(init)

            train_writer = tf.summary.FileWriter("/tmp/proj", session.graph)


            if not FLAGS.state_path:
                for i in range(config.max_max_epoch):
                    lr_decay = config.lr_decay ** max(i + 1 - config.initial_learning_epoch, 0.0)
                    model.assign_lr(session, config.learning_rate * lr_decay)

                    batch_size = np.array([config.batch_size], dtype=np.float32)
                    state = session.run(model.initial_state,{model.input_batchsize : batch_size})

                    fetches = {
                        "cost": model.cost,
                        "final_state": model.final_state,
                    }

                    fetches["eval_optimizer"] = model.train_optimizer

                    # input_train, output_train = r.getBatchDays(datetime.date(2017, 5, 25), config.num_days, config.batch_size)
                    input_train, output_train = r.getRandomTrainBatch(config.num_days,config.batch_size)
                    # output_train = np.array([[10, 20 ,30]])
                    # input_train = np.array([[1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]])
                    # batch_size = np.array([2], dtype=np.int32)
                    # batch_size = np.array([2], dtype=np.float32)
                    feed_dict = {model.input_data: input_train, model.output_data: output_train,model.input_batchsize : batch_size}

                    if i % 100 == 0:
                        fetches["summary"]=summary

                    vals = session.run(fetches, feed_dict)
                    cost = vals["cost"]
                    state = vals["final_state"]


                    if i % 100 == 0:
                        train_writer.add_summary(vals["summary"],i)
                        print(cost)


            if FLAGS.save_path and not FLAGS.state_path:
                print("Saving model to %s." % FLAGS.save_path)
                saver.save(session, FLAGS.save_path)

            train_writer.close()


            print('Validation...')
            batch_size = np.array([1], dtype=np.float32)
            session.run(model.initial_state,{model.input_batchsize : batch_size})
            input_valid, output_valid = r.getBatchDays(datetime.date(2017, 6, 15), previousdays=config.num_days, batchsize=1)
            feed_dict = {model.input_data: input_valid, model.output_data: output_valid,model.input_batchsize : batch_size}

            [valid_cost, valid_accuracy, valid_logits] = session.run([model.cost, model.accuracy, model.logits], feed_dict)

            print('Validation completed')
            print('Cost [MSE]', valid_cost)
            print('Accuracy [ME]', valid_accuracy)
            print('Hour | Expected temp | Calculated temp')

            for x, y, z in zip(itertools.cycle([1,12,19]), output_valid.flatten(), valid_logits.flatten()):
                print(x, '\t\t|\t\t', y, '\t\t|\t\t', z)


            print('Testing...')
            print('Temperature prediction for today ')
            batch_size = np.array([1], dtype=np.float32)
            session.run(model.initial_state,{model.input_batchsize : batch_size})
            input_test, output_test = r.getBatchDays(datetime.date.today()-datetime.timedelta(days=1), previousdays=config.num_days, batchsize=1)
            feed_dict = {model.input_data: input_test, model.input_batchsize: batch_size}
            test_logits = session.run(model.logits, feed_dict)
            print('Hour | Expected temp ')
            for x, y in zip(itertools.cycle([1,12,19]), test_logits.flatten()):
                print(x, '\t\t|\t\t', y)




class Config(object):
    init_scale = 0.1
    keep_prob = 1
    num_layers = 2
    hidden_size = 8
    batch_size = 5
    # num_steps - number of days provided to network in one batch
    num_days = 5
    max_max_epoch = 2500
    lr_decay = 0.9
    initial_learning_epoch = 100
    learning_rate = 1.0

if __name__ == "__main__":
    tf.app.run()
