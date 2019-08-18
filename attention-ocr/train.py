# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Script to train the Attention OCR model.

A simple usage example:
python train.py
"""
import collections
import logging
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow import app
from tensorflow.python.platform import flags
from tensorflow.contrib.tfprof import model_analyzer

import data_provider
import common_flags

import metrics
import time

FLAGS = flags.FLAGS
common_flags.define()

# yapf: disable
flags.DEFINE_integer('task', 0,
                     'The Task ID. This value is used when training with '
                     'multiple workers to identify each worker.')

flags.DEFINE_integer('ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then'
                     ' the parameters are handled locally by the worker.')

flags.DEFINE_integer('save_summaries_secs', 60,
                     'The frequency with which summaries are saved, in '
                     'seconds.')

flags.DEFINE_integer('save_interval_secs', 600,
                     'Frequency in seconds of saving the model.')

flags.DEFINE_integer('max_number_of_steps', int(1e10),
                     'The maximum number of gradient steps.')

flags.DEFINE_string('checkpoint_inception', '',
                    'Checkpoint to recover inception weights from.')

flags.DEFINE_float('clip_gradient_norm', 2.0,
                   'If greater than 0 then the gradients would be clipped by '
                   'it.')

flags.DEFINE_bool('sync_replicas', False,
                  'If True will synchronize replicas during training.')

flags.DEFINE_integer('replicas_to_aggregate', 1,
                     'The number of gradients updates before updating params.')

flags.DEFINE_integer('total_num_replicas', 1,
                     'Total number of worker replicas.')

flags.DEFINE_integer('startup_delay_steps', 15,
                     'Number of training steps between replicas startup.')

flags.DEFINE_boolean('reset_train_dir', False,
                     'If true will delete all files in the train_log_dir')

flags.DEFINE_boolean('show_graph_stats', False,
                     'Output model size stats to stderr.')
# yapf: enable

TrainingHParams = collections.namedtuple('TrainingHParams', [
    'learning_rate',
    'optimizer',
    'momentum',
    'use_augment_input',
    'use_default_augment'
])


def get_training_hparams():
    return TrainingHParams(
        learning_rate=FLAGS.learning_rate,
        optimizer=FLAGS.optimizer,
        momentum=FLAGS.momentum,
        use_augment_input=FLAGS.use_augment_input,
        use_default_augment=FLAGS.use_default_augment)


def create_optimizer(hparams):
    """Creates optimized based on the specified flags."""
    if hparams.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            hparams.learning_rate, momentum=hparams.momentum)
    elif hparams.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(hparams.learning_rate)
    elif hparams.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(hparams.learning_rate)
    elif hparams.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(hparams.learning_rate)
    elif hparams.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            hparams.learning_rate, momentum=hparams.momentum)
    return optimizer


def train(loss, init_fn, CharacterAccuracy, SequenceAccuracy, hparams):
    """Wraps slim.learning.train to run a training loop.

    Args:
      loss: a loss tensor
      init_fn: A callable to be executed after all other initialization is done.
      hparams: a model hyper parameters
    """
    optimizer = create_optimizer(hparams)

    if FLAGS.sync_replicas:
        replica_id = tf.constant(FLAGS.task, tf.int32, shape=())
        optimizer = tf.LegacySyncReplicasOptimizer(
            opt=optimizer,
            replicas_to_aggregate=FLAGS.replicas_to_aggregate,
            replica_id=replica_id,
            total_num_replicas=FLAGS.total_num_replicas)
        sync_optimizer = optimizer
        startup_delay_steps = 0
    else:
        startup_delay_steps = 0
        sync_optimizer = None

    train_op = slim.learning.create_train_op(
        loss,
        optimizer,
        summarize_gradients=True,
        clip_gradient_norm=FLAGS.clip_gradient_norm)

    slim.learning.train(
        train_op=train_op,
        train_step_fn=train_step,
        train_step_kwargs={'CharacterAccuracy': CharacterAccuracy,
                           'SequenceAccuracy': SequenceAccuracy},
        logdir=FLAGS.train_log_dir,
        graph=loss.graph,
        master=FLAGS.master,
        is_chief=(FLAGS.task == 0),
        number_of_steps=FLAGS.max_number_of_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        startup_delay_steps=startup_delay_steps,
        sync_optimizer=sync_optimizer,
        init_fn=init_fn)


def train_step(sess, train_op, global_step, train_step_kwargs):
    """Function that takes a gradient step and specifies whether to stop.
    Args:
      sess: The current session.
      train_op: An `Operation` that evaluates the gradients and returns the
        total loss.
      global_step: A `Tensor` representing the global training step.
      train_step_kwargs: A dictionary of keyword arguments.
    Returns:
      The total loss and a boolean indicating whether or not to stop training.
    Raises:
      ValueError: if 'should_trace' is in `train_step_kwargs` but `logdir` is not.
    """
    start_time = time.time()

    trace_run_options = None
    run_metadata = None
    if 'should_trace' in train_step_kwargs:
        if 'logdir' not in train_step_kwargs:
            raise ValueError('logdir must be present in train_step_kwargs when '
                             'should_trace is present')
        if sess.run(train_step_kwargs['should_trace']):
            trace_run_options = config_pb2.RunOptions(
                trace_level=config_pb2.RunOptions.FULL_TRACE)
            run_metadata = config_pb2.RunMetadata()

    total_loss, np_global_step = sess.run([train_op, global_step],
                                          options=trace_run_options,
                                          run_metadata=run_metadata)
    time_elapsed = time.time() - start_time
    if 'CharacterAccuracy' in train_step_kwargs and 'SequenceAccuracy' in train_step_kwargs:
        char_acc, seq_acc = sess.run(
            [train_step_kwargs['CharacterAccuracy'], train_step_kwargs['SequenceAccuracy']])
        if np_global_step % 10 == 0:
            print('global step {}: loss = {:.4f} ({:.3f} sec/step) - Char acc: {:.3f}, Seq acc: {:.3f}'.format(
                np_global_step, total_loss, time_elapsed, char_acc[0], seq_acc[0]))

    if run_metadata is not None:
        tl = timeline.Timeline(run_metadata.step_stats)
        trace = tl.generate_chrome_trace_format()
        trace_filename = os.path.join(train_step_kwargs['logdir'],
                                      'tf_trace-%d.json' % np_global_step)
        logging.info('Writing trace to %s', trace_filename)
        file_io.write_string_to_file(trace_filename, trace)
        if 'summary_writer' in train_step_kwargs:
            train_step_kwargs['summary_writer'].add_run_metadata(run_metadata,
                                                                 'run_metadata-%d' %
                                                                 np_global_step)

    # if 'should_log' in train_step_kwargs:
    #   if sess.run(train_step_kwargs['should_log']):
    #     logging.info('global step %d: loss = %.4f (%.3f sec/step)',
    #                  np_global_step, total_loss, time_elapsed)

    # TODO(nsilberman): figure out why we can't put this into sess.run. The
    # issue right now is that the stop check depends on the global step. The
    # increment of global step often happens via the train op, which used
    # created using optimizer.apply_gradients.
    #
    # Since running `train_op` causes the global step to be incremented, one
    # would expected that using a control dependency would allow the
    # should_stop check to be run in the same session.run call:
    #
    #   with ops.control_dependencies([train_op]):
    #     should_stop_op = ...
    #
    # However, this actually seems not to work on certain platforms.
    if 'should_stop' in train_step_kwargs:
        should_stop = sess.run(train_step_kwargs['should_stop'])
    else:
        should_stop = False

    return total_loss, should_stop


def prepare_training_dir():
    if not tf.gfile.Exists(FLAGS.train_log_dir):
        logging.info('Create a new training directory %s', FLAGS.train_log_dir)
        tf.gfile.MakeDirs(FLAGS.train_log_dir)
    else:
        if FLAGS.reset_train_dir:
            logging.info('Reset the training directory %s',
                         FLAGS.train_log_dir)
            tf.gfile.DeleteRecursively(FLAGS.train_log_dir)
            tf.gfile.MakeDirs(FLAGS.train_log_dir)
        else:
            logging.info('Use already existing training directory %s',
                         FLAGS.train_log_dir)


def calculate_graph_metrics():
    param_stats = model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    return param_stats.total_parameters


def main(_):
    prepare_training_dir()

    dataset = common_flags.create_dataset(split_name=FLAGS.split_name)
    print(dataset)
    model = common_flags.create_model(dataset.num_char_classes,
                                      dataset.max_sequence_length,
                                      dataset.num_of_views, dataset.null_code)
    hparams = get_training_hparams()

    # If ps_tasks is zero, the local device is used. When using multiple
    # (non-local) replicas, the ReplicaDeviceSetter distributes the variables
    # across the different devices.
    device_setter = tf.train.replica_device_setter(
        FLAGS.ps_tasks, merge_devices=True)
    with tf.device(device_setter):
        data = data_provider.get_data(
            dataset,
            FLAGS.batch_size,
            augment=hparams.use_augment_input,
            central_crop_size=common_flags.get_crop_size(),
            use_default_augment=hparams.use_default_augment)
        endpoints = model.create_base(data.images, data.labels_one_hot)
        total_loss = model.create_loss(data, endpoints)
        CharacterAccuracy = metrics.char_accuracy(
            endpoints.predicted_chars,
            data.labels,
            streaming=True,
            rej_char=model._params.null_code)
        SequenceAccuracy = metrics.sequence_accuracy(
            endpoints.predicted_chars,
            data.labels,
            streaming=True,
            rej_char=model._params.null_code)
        model.create_summaries(
            data, endpoints, dataset.charset, is_training=True)
        init_fn = model.create_init_fn_to_restore(FLAGS.checkpoint,
                                                  FLAGS.checkpoint_inception)
        if FLAGS.show_graph_stats:
            logging.info('Total number of weights in the graph: %s',
                         calculate_graph_metrics())
        train(total_loss, init_fn, CharacterAccuracy, SequenceAccuracy, hparams)


if __name__ == '__main__':
    app.run()
