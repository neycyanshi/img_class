import tensorflow as tf
import fashion_mnist
import model
import argparse
import data_util

PREDICT = tf.estimator.ModeKeys.PREDICT
EVAL = tf.estimator.ModeKeys.EVAL
TRAIN = tf.estimator.ModeKeys.TRAIN

tf.app.flags.DEFINE_string('log_dir', './log', 'dir for model and logs')
tf.app.flags.DEFINE_float('lr', 0.002, 'learning rate')
tf.app.flags.DEFINE_integer('steps', 10000, 'training steps')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
FLAGS = tf.app.flags.FLAGS


def main(argv=None):
    """Run the training experiment"""
    # Read fashion mnist data
    mnist_train, mnist_test = fashion_mnist.load_data()
    config = tf.estimator.RunConfig(
        model_dir=FLAGS.log_dir,
        save_summary_steps=100,
        log_step_count_steps=100,
        save_checkpoints_steps=500,
    )
    # Setup the Estimator
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, config=config)
    # Start training and validation
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: get_train_inputs(FLAGS.batch_size, mnist_train),
        max_steps=FLAGS.steps)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: get_eval_inputs(FLAGS.batch_size, mnist_test),
        steps=None,
        start_delay_secs=10,  # Start evaluating after 10 sec.
        throttle_secs=30  # Evaluate only every 30 sec
    )
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def model_fn(features, labels, mode):
    """Model function used in the estimator.
    Args:
        features (Tensor): Input features to the model.
        labels (Tensor): Labels tensor for training and evaluation.
        mode (ModeKeys): Specifies if training, evaluation or prediction.
    Returns:
        (EstimatorSpec): Model to be run by Estimator.
    """
    # Define model's architecture
    logits = model.base_drop(features, mode)
    # logits = model.WideResNet(mode).forward(features)
    class_predictions = tf.argmax(logits, axis=-1)
    # Setup the estimator according to the phase (Train, eval, predict)
    loss = None
    train_op = None
    eval_metric_ops = {}
    predictions = class_predictions
    # Loss will only be tracked during training or evaluation.
    if mode in (TRAIN, EVAL):
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=tf.cast(labels, tf.int32), logits=logits)
    # Training operator only needed during training.
    if mode == TRAIN:
        optimizer = tf.train.AdamOptimizer(FLAGS.lr)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
    # Evaluation operator only needed during evaluation
    if mode == EVAL:
        eval_metric_ops = {
            'accuracy':
            tf.metrics.accuracy(
                labels=labels, predictions=class_predictions, name='accuracy')
        }
    # Class predictions and probabilities only needed during inference.
    if mode == PREDICT:
        predictions = {
            'classes': class_predictions,
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def get_train_inputs(batch_size, mnist_data):
    """Return the input function to get the training data.
    Args:
        batch_size (int): Batch size of training iterator that is returned
                          by the input function.
        mnist_data ((array, array): Mnist training data as (inputs, labels).
    Returns:
        DataSet: A tensorflow DataSet object to represent the training input
                 pipeline.
    """
    dataset = tf.data.Dataset.from_tensor_slices(mnist_data)
    dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(data_util.transform_train)
    dataset = dataset.repeat(count=None)  # infinite epochs
    return dataset


def get_eval_inputs(batch_size, mnist_data):
    """Return the input function to get the validation data.
    Args:
        batch_size (int): Batch size of validation iterator that is returned
                          by the input function.
        mnist_data ((array, array): Mnist test data as (inputs, labels).
    Returns:
        DataSet: A tensorflow DataSet object to represent the validation input
                 pipeline.
    """
    dataset = tf.data.Dataset.from_tensor_slices(mnist_data)
    dataset = dataset.map(data_util.transform_val)
    dataset = dataset.batch(batch_size)
    return dataset


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
