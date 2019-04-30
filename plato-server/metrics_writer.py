import logging
import tensorflow as tf
import threading
import torch.multiprocessing as mp

class MetricsWriter(object):
  
  queue = mp.SimpleQueue()

  def __init__(self, path):
    self.path = path
    self.sess = tf.Session()
    self.episode_length = tf.Variable(0.0, name='episode_length')
    self.episode_reward = tf.Variable(0.0, name='reward')
    self.loss = tf.Variable(0.0, name='loss')
    # self.value_loss = tf.Variable(0.0, name='value_loss')
    # self.policy_loss = tf.Variable(0.0, name='policy_loss')
    self.gradient_norm = tf.Variable(0.0, name='gradient_norm')
    self.policy_distribution = tf.Variable([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], name='policy_distribution')
    
    self.q_forward = tf.placeholder(tf.float32, name='q_forward')
    self.q_backward = tf.placeholder(tf.float32, name='q_backward')
    self.q_left = tf.placeholder(tf.float32, name='q_left')
    self.q_right = tf.placeholder(tf.float32, name='q_right')
    self.q_fire = tf.placeholder(tf.float32, name='q_fire')
    self.q_nothing = tf.placeholder(tf.float32, name='q_nothing')

    self.updates = 0
    self.episodes = 0

    length_summary = tf.summary.scalar('episode_length', self.episode_length)
    reward_summary = tf.summary.scalar('episode_reward', self.episode_reward)
    # value_loss_summary = tf.summary.scalar('value_loss', self.value_loss)
    # policy_loss_summary = tf.summary.scalar('policy_loss', self.policy_loss)
    loss_summary = tf.summary.scalar('loss', self.loss)
    gradient_summary = tf.summary.scalar('gradient_norm', self.gradient_norm)
    policy_distribution_summary = tf.summary.histogram('policy_distribution', self.policy_distribution)
    
    q_forward_summary = tf.summary.histogram('q_forward', self.q_forward)
    q_backward_summary = tf.summary.histogram('q_backward', self.q_backward)
    q_left_summary = tf.summary.histogram('q_left', self.q_left)
    q_right_summary = tf.summary.histogram('q_right', self.q_right)
    q_fire_summary = tf.summary.histogram('q_fire', self.q_fire)
    q_nothing_summary = tf.summary.histogram('q_nothing', self.q_nothing)

    self.episode_summary = tf.summary.merge([length_summary, reward_summary, q_forward_summary, q_backward_summary, q_left_summary, q_right_summary, q_fire_summary, q_nothing_summary])
    self.update_summary = tf.summary.merge([loss_summary, gradient_summary, policy_distribution_summary])
    self.writer = tf.summary.FileWriter(self.path, self.sess.graph)

    self.sess.run(tf.global_variables_initializer())

  def start_listening(self):
    t = threading.Thread(target=self._listen)
    t.daemon = True
    t.start()

  def _listen(self):
    while True:
      log = MetricsWriter.queue.get()
      summaries = None
      step = 0
      if log[0] == 0:
        summaries = self.sess.run(self.episode_summary, feed_dict={
          self.episode_length: log[1],
          self.episode_reward: log[2],
          self.q_forward: log[3],
          self.q_backward: log[4],
          self.q_left: log[5],
          self.q_right: log[6],
          self.q_fire: log[7],
          self.q_nothing: log[8]
        })

        if len(log) != 9:
          raise Exception("Invalid episode summary to MetricsWriter")

        step = self.episodes
        self.episodes += 1

        if self.episodes % 10 == 0:
          self.writer.flush()
      elif log[0] == 1:
        summaries = self.sess.run(self.update_summary, feed_dict={self.loss: log[1], self.gradient_norm: log[2], self.policy_distribution: log[3]})
        step = self.updates
        self.updates += 1
      # elif log[0] == 2:
      #   summaries = self.sess.run(self.)

        if self.updates % 1000 == 0:
          self.writer.flush()
      self.writer.add_summary(summaries, step)

  def log_episode(self, length, reward, q_forward, q_backward, q_left, q_right, q_fire, q_nothing):
    MetricsWriter.queue.put((0, length, reward, q_forward, q_backward, q_left, q_right, q_fire, q_nothing))

  def log_update(self, loss, gradient_norm, policy_distribution):
    MetricsWriter.queue.put((1, loss, gradient_norm, policy_distribution))
  
  # def log_action(self, policy_distribution):
  #   MetricsWriter.queue.put((2, policy_distribution))