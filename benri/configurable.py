""" Base class for objects configurable by parameter dictionaries.

Inspired by Denny Britz:
https://github.com/google/seq2seq/blob/master/seq2seq/configurable.py
"""
import abc
import copy

import six
import yaml


class abstractstaticmethod(staticmethod):  #pylint: disable=C0111,C0103
    """Decorates a method as abstract and static"""
    __slots__ = ()

    def __init__(self, function):
        """ Constructor.

        :param function: Function to be decorated.
        """
        super(abstractstaticmethod, self).__init__(function)
        function.__isabstractmethod__ = True

    __isabstractmethod__ = True


def _parse_params(params, default_params):
    """Parses parameter values to the types defined by the default parameters.
    Default parameters are used for missing values.
    """
    # Cast parameters to correct types
    if params is None:
        params = {}
    result = copy.deepcopy(default_params)
    for key, value in params.items():
        # If param is unknown, drop it to stay compatible with past versions
        if key not in default_params:
            raise ValueError("%s is not a valid model parameter" % key)
        # Param is a dictionary
        if isinstance(value, dict):
            default_dict = default_params[key]
            if not isinstance(default_dict, dict):
                raise ValueError("%s should not be a dictionary", key)
            if default_dict:
                value = _parse_params(value, default_dict)
            else:
                # If the default is an empty dict we do not typecheck it
                # and assume it's done downstream
                pass
        if value is None:
            continue
        if default_params[key] is None:
            result[key] = value
        else:
            result[key] = type(default_params[key])(value)
    return result


@six.add_metaclass(abc.ABCMeta)
class Configurable(object):
    """ Abstract base class for classes parameterized with a dict. """

    def __init__(self, params, verbose=False):
        """ Constructor.

        :param params: Parameter dictionary. Must have same keys and value type
          as the default parameter dictionary defined in `self.default_params`.
        :param verbose: Boolean, whether the configuration should be displayed.
        """
        self._params = _parse_params(params, self.default_params())
        self._verbose = verbose
        if self._verbose:
            self._print_params()

    def _print_params(self):
        """ Log parameter values. """
        classname = self.__class__.__name__
        print("Creating {}, with configuration:\n{}.".format(
            classname,
            yaml.dump({classname: self._params})))

    @property
    def params(self):
        """ Returns a dictionary of parameters.

        :return: Dictionary containing the configuration parameters.
        """
        return self._params

    @abstractstaticmethod
    def default_params():
        """ Returns a dictionary of default parameters.

        :return: Dictionary containing parameters for default configuration.
        """
        raise NotImplementedError(
            "Must specify a default parameter dictionary for a configurable "
            "class.")
