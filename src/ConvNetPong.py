'''Solves Pong with ConvNet and Policy Gradients in Tensorflow.'''
# written October 2016 by Sam Greydanus
# inspired by karpathy's gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as pl
import time
# hyperparameters


n_obs = 80 * 80  # dimensionality of observations
h = 200  # number of hidden layer neurons
n_actions = 3  # number of available actions
learning_rate = 1e-4
gamma = .99  # discount factor for reward
decay = 0.99  # decay rate for RMSProp gradients
save_path = 'models/pong.ckpt'

# gamespace
env = gym.make("Pong-v0")  # environment info
observation = env.reset()
prev_x = None
xs, rs, ys = [], [], []
running_reward = None
reward_sum = 0
episode_number = 0




def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, strides=[1, 1, 1, 1]):

  return tf.nn.conv2d(x, W, strides=strides, padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],/home/johnny/my480/deepreinforcementlearning/experiment_results/report_data_collection/Pong.tar.gz
                        strides=[1, 2, 2, 1], padding='VALID')



























"""
with tf.variable_scope('layer_one', reuse=False):
    xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1. / np.sqrt(n_obs), dtype=tf.float32)
    tf_model['W1'] = tf.get_variable("W1", [n_obs, h], initializer=xavier_l1)
    tf_model['b1'] = tf.get_variable("b1", [h], initializer=xavier_l1)
with tf.variable_scope('layer_two', reuse=False):
    xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1. / np.sqrt(h), dtype=tf.float32)
    tf_model['W2'] = tf.get_variable("W2", [h, n_actions], initializer=xavier_l2)
    tf_model['b2'] = tf.get_variable("b2", [n_actions], initializer=xavier_l2)
"""

# tf operations
def tf_discount_rewards(tf_r):  # tf_r ~ [game_steps,1]
    discount_f = lambda a, v: a * gamma + v;
    tf_r_reverse = tf.scan(discount_f, tf.reverse(tf_r, [True, False]))
    tf_discounted_r = tf.reverse(tf_r_reverse, [True, False])
    return tf_discounted_r


def tf_policy_forward(x):  # x ~ [1,D]
    """
    h = tf.add(tf.matmul(x, tf_model['W1']), tf_model['b1'])
    h = tf.nn.relu(h)
    logp = tf.add(tf.matmul(h, tf_model['W2']),tf_model['b2'])
    """
    extras = list()

    # First conv layer
    W_conv1 = weight_variable([20, 20, 2, 64])
    b_conv1 = bias_variable([64])
    x_image = tf.reshape(tf_x, [-1, 80, 80, 2])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, strides=[1, 20, 20, 1]) + b_conv1)
    h_pool1 = h_conv1
    #h_pool1 = max_pool_2x2(h_pool1)

    # Second conv layer
    W_conv2 = weight_variable([1, 1, 64, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
   # h_pool2 = max_pool_2x2(h_conv2)
    h_pool2 = h_conv2

    conv_out = h_pool2
    vsize = conv_out._shape[1]._value * conv_out._shape[2]._value * conv_out._shape[3]._value

    # Densely connected layer:

    W_fc1 = weight_variable([vsize, 256])
    b_fc1 = bias_variable([256])

    h_pool2_flat = tf.reshape(conv_out, [-1,vsize])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    """
    # Dropout layer
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    """
    # Readout layer
    W_fc2 = weight_variable([256, 3])
    b_fc2 = bias_variable([3])



    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2


    p = tf.nn.softmax(y_conv)

    extras.append(x_image)

    return [p, extras]


# downsampling
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()



# initialize model
tf_model = {}


# tf placeholders
tf_x = tf.placeholder(dtype=tf.float32, shape=[None, n_obs, 2], name="tf_x")
tf_y = tf.placeholder(dtype=tf.float32, shape=[None, n_actions], name="tf_y")
tf_epr = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="tf_epr")













# tf reward processing (need tf_discounted_epr for policy gradient wizardry)
tf_discounted_epr = tf_discount_rewards(tf_epr)
tf_mean, tf_variance = tf.nn.moments(tf_discounted_epr, [0], shift=None, name="reward_moments")
tf_discounted_epr -= tf_mean
tf_discounted_epr /= tf.sqrt(tf_variance + 1e-6)










# tf optimizer op
[tf_aprob, extras] = tf_policy_forward(tf_x)
#tf_aprob = tf.nn.softmax(tf_out)
loss = tf.nn.l2_loss(tf_y - tf_aprob)
optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay)
tf_grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables(), grad_loss=tf_discounted_epr)
train_op = optimizer.apply_gradients(tf_grads)

# tf graph initialization
sess = tf.InteractiveSession()
tf.initialize_all_variables().run()



# try load saved model
saver = tf.train.Saver(tf.all_variables())
load_was_success = True  # yes, I'm being optimistic
try:
    save_dir = '/'.join(save_path.split('/')[:-1])
    ckpt = tf.train.get_checkpoint_state(save_dir)
    load_path = ckpt.model_checkpoint_path
    saver.restore(sess, load_path)
except:
    print "no saved model to load. starting new session"
    load_was_success = False
else:
    print "loaded model: {}".format(load_path)
    saver = tf.train.Saver(tf.all_variables())
    episode_number = int(load_path.split('-')[-1])

#pl.ion()
#pl.show()

#f, axarr = pl.subplots(2, sharex=True)





prev_x = np.zeros(80*80)
# training loop
while True:

    env.render()


    # preprocess the observation, set input to network to be difference image
    # note: what if we implement a timestep here?

    cur_x = prepro(observation)
    #x = cur_x - prev_x if prev_x is not None else np.zeros(n_obs)
    x = np.concatenate([np.expand_dims(prev_x, 1), np.expand_dims(cur_x, 1)], axis=1)

    prev_x = cur_x
    x = np.expand_dims(x, 0)
    # stochastically sample a policy from the network
    #feed = {tf_x: np.reshape(x, (1, -1))}
    feed = {tf_x: x}
    aprob = sess.run(tf_aprob, feed)
    aprob = aprob[0, :]


    action = np.random.choice(n_actions, p=aprob)
    #action = np.argmax(aprob)

    label = np.zeros_like(aprob);
    label[action] = 1

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action + 1)
    reward_sum += reward

    # record game history
    #xs.append(np.reshape(x,(-1,1)));
    xs.append(x)
    ys.append(label);
    rs.append(reward)

    extras_eval = sess.run(extras, feed)


    images_to_plot = extras_eval[0][-1, :, :, :]
#    axarr[0].imshow(images_to_plot[:, :, 0])
 #   axarr[1].imshow(images_to_plot[:, :, 1])
    pl.pause(0.001)
  #  time.sleep(0.01)


    if done:
        # update running reward
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

        # parameter update
        feed = {tf_x: np.vstack(xs), tf_epr: np.vstack(rs), tf_y: np.vstack(ys)}
        _ = sess.run(train_op, feed)




        # print progress console
        if episode_number % 10 == 0:
            print 'ep {}: reward: {}, mean reward: {:3f}'.format(episode_number, reward_sum, running_reward)
        else:
            print '\tep {}: reward: {}'.format(episode_number, reward_sum)

        # bookkeeping
        xs, rs, ys = [], [], []  # reset game history
        episode_number += 1  # the Next Episode
        observation = env.reset()  # reset env
        reward_sum = 0
        if episode_number % 50 == 0:
            saver.save(sess, save_path, global_step=episode_number)
            print "SAVED MODEL #{}".format(episode_number)
