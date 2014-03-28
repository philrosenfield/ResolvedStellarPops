from __future__ import print_function
import collections
import warnings


__all__ = ['deprecated', 'flatten_dict']


def deprecated(func):
    '''This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.'''
    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func


def flatten_dict(d, parent_key='', joinchar='.'):
    """ Returns a flat dictionary from nested dictionaries

    the flatten structure will be identified by longer keys coding the tree.
    A value will be identified by the joined list of node names

    e.g.:
        {'a': {'b': 0,
               'c': {'r': 10,
                     's': 20 } },
        'd': 3 }

        becomes
        {'a.b':0,
         'a.c.r':10,
         'a.c.s':10,
         'd':3 }

    Parameters
    ----------
    d: dict
        nested dictionary

    parent_key: str, optional (default=empty)
        optional parent key used during the recursion

    joinchar: str, optional (default='.')
        joining character between levels

    Returns
    -------
    fd: dict
        flatten dictionary
    """

    items = []
    for k, v in d.items():
        new_key = parent_key + joinchar + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, joinchar).items())
        else:
            items.append((new_key, v))
    return dict(items)
