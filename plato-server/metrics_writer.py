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
    self.value_loss = tf.Variable(0.0, name='value_loss')
    self.policy_loss = tf.Variable(0.0, name='policy_loss')
    self.gradient_norm = tf.Variable(0.0, name='gradient_norm')
    self.policy_distribution = tf.Variable([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], name='policy_distribution')

    self.updates = 0
    self.episodes = 0

    length_summary = tf.summary.scalar('episode_length', self.episode_length)
    reward_summary = tf.summary.scalar('episode_reward', self.episode_reward)
    value_loss_summary = tf.summary.scalar('value_loss', self.value_loss)
    policy_loss_summary = tf.summary.scalar('policy_loss', self.policy_loss)
    gradient_summary = tf.summary.scalar('gradient_norm', self.gradient_norm)
    policy_distribution_summary = tf.summary.histogram('policy_distribution', self.policy_distribution)

    self.episode_summary = tf.summary.merge([length_summary, reward_summary])
    self.update_summary = tf.summary.merge([value_loss_summary, policy_loss_summary, gradient_summary, policy_distribution_summary])
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
        summaries = self.sess.run(self.episode_summary, feed_dict={self.episode_length: log[1], self.episode_reward: log[2]})
        step = self.episodes
        self.episodes += 1
      else:
        summaries = self.sess.run(self.update_summary, feed_dict={self.value_loss: log[1], self.policy_loss: log[2], self.gradient_norm: log[3], self.policy_distribution: log[4]})
        step = self.updates
        self.updates += 1
      self.writer.add_summary(summaries, step)

  def log_episode(self, length, reward):
    MetricsWriter.queue.put((0, length, reward))

  def log_update(self, value_loss, policy_loss, gradient_norm, policy_distribution):
    MetricsWriter.queue.put((1, value_loss, policy_loss, gradient_norm, policy_distribution))