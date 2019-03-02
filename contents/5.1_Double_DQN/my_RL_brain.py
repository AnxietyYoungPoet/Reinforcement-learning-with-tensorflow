"""
The double DQN based on this paper: https://arxiv.org/abs/1509.06461

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class DoubleDQN:
  def __init__(
      self,
      n_actions,
      n_features,
      learning_rate=0.005,
      reward_decay=0.9,
      e_greedy=0.9,
      replace_target_iter=200,
      memory_size=3000,
      batch_size=32,
      e_greedy_increment=None,
      output_graph=False,
      double_q=True,
      sess=None,
      scope=''
  ):
    self.n_actions = n_actions
    self.n_features = n_features
    self.lr = learning_rate
    self.gamma = reward_decay
    self.epsilon_max = e_greedy
    self.replace_target_iter = replace_target_iter
    self.memory_size = memory_size
    self.batch_size = batch_size
    self.epsilon_increment = e_greedy_increment
    self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
    self.scope = scope

    self.double_q = double_q  # decide to use double q or not

    self.learn_step_counter = 0
    self.memory = np.zeros((self.memory_size, n_features*2+2))
    self._build_net()
    self.replace_target_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]

    if sess is None:
      self.sess = tf.Session()
      self.sess.run(tf.global_variables_initializer())
    else:
      self.sess = sess
    if output_graph:
      tf.summary.FileWriter("logs/", self.sess.graph)
    self.cost_his = []

  def _build_net(self):
    self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
    self.q_target = tf.placeholder(tf.float32, [None, ], name='q_target')
    self.r = tf.placeholder(tf.float32, [None, ], name='r')
    self.a = tf.placeholder(tf.int32, [None, ], name='a')

    w_initializer, b_initializer = tf.keras.initializers.he_normal(), tf.constant_initializer(0.1)
    with tf.variable_scope(self.scope + 'eval_net'):
      e1 = tf.layers.dense(self.s, 32, tf.nn.relu, kernel_initializer=w_initializer,
        bias_initializer=b_initializer, name='e1')
      self.q_eval_all = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
        bias_initializer=b_initializer, name='q_eval_all')
    
    with tf.variable_scope(self.scope + 'target_net'):
      t1 = tf.layers.dense(self.s, 32, tf.nn.relu, kernel_initializer=w_initializer,
        bias_initializer=b_initializer, name='t1')
      self.q_target_all = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
        bias_initializer=b_initializer, name='q_target_all')
    
    self.t_params = tf.get_collection(
      tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + 'target_net')
    self.e_params = tf.get_collection(
      tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + 'eval_net')

      # self.q_target = tf.stop_gradient(q_target)
    with tf.variable_scope('q_eval'):
      a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
      self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval_all, indices=a_indices)  # shape=(None, )
    with tf.variable_scope('loss'):
      self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
    with tf.variable_scope('train'):
      self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss, var_list=self.e_params)

  def store_transition(self, s, a, r, s_):
    if not hasattr(self, 'memory_counter'):
      self.memory_counter = 0
    transition = np.hstack((s, [a, r], s_))
    index = self.memory_counter % self.memory_size
    self.memory[index, :] = transition
    self.memory_counter += 1

  def choose_action(self, observation):
    observation = observation[np.newaxis, :]
    actions_value = self.sess.run(self.q_eval_all, feed_dict={self.s: observation})
    action = np.argmax(actions_value)

    if not hasattr(self, 'q'):  # record action value it gets
      self.q = []
      self.running_q = 0
    self.running_q = self.running_q*0.99 + 0.01 * np.max(actions_value)
    self.q.append(self.running_q)

    if np.random.uniform() > self.epsilon:  # choosing action
      action = np.random.randint(0, self.n_actions)
    return action

  def learn(self):
    if self.learn_step_counter % self.replace_target_iter == 0:
      self.sess.run(self.replace_target_op)
      print('\ntarget_params_replaced\n')

    if self.memory_counter > self.memory_size:
      sample_index = np.random.choice(self.memory_size, size=self.batch_size)
    else:
      sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
    batch_memory = self.memory[sample_index, :]

    q_next_eval, q_next_target = self.sess.run(
      [self.q_eval_all, self.q_target_all],
      feed_dict={self.s:batch_memory[:, -self.n_features:]}
    )

    batch_index = np.arange(self.batch_size, dtype=np.int32)
    eval_act_index = batch_memory[:, self.n_features].astype(int)
    reward = batch_memory[:, self.n_features + 1]

    if self.double_q:
      max_act4next = np.argmax(q_next_eval, axis=1)    # the action that brings the highest value is evaluated by q_eval
      selected_q_next = q_next_target[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
    else:
      selected_q_next = np.max(q_next_target, axis=1)  # the natural DQN

    q_target = reward + self.gamma * selected_q_next

    _, self.cost = self.sess.run(
      [self._train_op, self.loss],
      feed_dict={self.s: batch_memory[:, :self.n_features],
                 self.q_target: q_target,
                 self.a: eval_act_index})
    self.cost_his.append(self.cost)

    self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
    self.learn_step_counter += 1




