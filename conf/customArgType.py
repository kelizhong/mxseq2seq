import sys
import logging
import os
import errno

def IntegerType(value):
    return sys.maxsize if value == 'inf' else int(value)

def LoggerLevelType(value):
    choices = {'debug': logging.DEBUG, 'info': logging.INFO, 'warn': logging.WARN, 'error': logging.ERROR}
    result = choices.get(value, logging.ERROR)
    return result

def DirectoryType(value):
    if not os.path.exists(value):
        os.makedirs(value)
    return value

def FileType(value):
    if not os.path.exists(os.path.dirname(value)):
        try:
            os.makedirs(os.path.dirname(value))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return value