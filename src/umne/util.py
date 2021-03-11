"""
Authors: Dror Dotan <dror.dotan@gmail.com>
         Fosca Al Roumi <fosca.al.roumi@gmail.com>
"""
import math


#------------------------------------------------------------------------------------
def is_collection(value, allow_set=True):
    """
    Check whether the given value is a collection (list, tuple, etc.)

    :param value: The value to test
    :param allow_set: Indicate whether to return True when "value" is a set
    :type allow_set: bool
    """
    val_methods = dir(value)
    return "__len__" in val_methods and "__iter__" in val_methods and \
           (allow_set or "__getitem__" in val_methods) and not isinstance(value, str)


#------------------------------------------------------------------------------------
class TimeRange(object):

    def __init__(self, min_time=-math.inf, max_time=math.inf):
        self.min_time = min_time
        self.max_time = max_time

