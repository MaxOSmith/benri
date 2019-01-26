""" Utility functions for constructing summaries. """
import io

# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def scalar(tag, value):
    """ Construct a scalar summary.

    :param tag: Name of the scalar.
    :param value: Scalar value.
    """
    summary = tf.Summary.Value(tag=tag, simple_value=value)
    summary = tf.Summary(value=[summary])
    return summary


"""
def image(tag, image):
    Construct an image summary.

    :param tag: Name of the scalar.
    :param image: Image [H, W, C].
    
    # Write the image to a string.
    s = io.StringIO()
    plt.imsave(s, image, format="png")

    image_summary = tf.Summary.Image(
        encoded_image_strng=s.getvalue(),
        height=image.shape[0],
        width=image.shape[1])

    summary = tf.Summary.Value(tag=tag, image=summary)
    summary = tf.Summary(value=[summary])
    return summary
"""


def histogram(tag, values, bins=1000):
    """ Construct a histogram summary.

    :param tag:
    :param values:
    :param bins:
    """
    values = np.array(values)

    # Create histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Thus, we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    # Create and write Summary
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    return summary
