""" Tensorboard logger that does not use tensor operations.

Original: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
"""
import numpy as np
import tensorflow as tf

import benri.tensorflow.summaries as summary_ops


class TensorboardLogger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def log_dict(self, data, step):
        """ Log all of the data in a dictionary.

        :param data: Dictionary from tags to values.
        :param step: Global step.
        """
        for tag, value in data.items():
            if np.isscalar(value):
                self.log_scalar(tag, value, step)
            else:
                raise ValueError("Type {} not supported.".format(type(value)))

    def log_scalar(self, tag, value, step):
        """ Log a scalar variable.

        :param tag: Name of the scalar.
        :param value: Scalar value.
        :param step: Global step.    
        """
        summary = summary_ops.scalar(tag, value)
        self.writer.add_summary(summary, step)


    def log_image(self, tag, image, step):
        """Logs an images."""
        # summary = summary_ops.image(tag, image)
        # self.writer.add_summary(summary, step)
        raise NotImplementedError("Issue with TkAgg on Mac.")

    def log_images(self, tag, images, step):
        """Logs a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            img_tag = "{}/{}".format(tag, i)
            img_summary = summary_ops.image(img_tag, img)

            # Create a Summary value
            img_summaries.append(img_summary)

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        summary = summary_ops.histogram(tag, values, bins)
        self.writer.add_summary(summary, step)
        self.writer.flush()
