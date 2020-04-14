""" This module contains custom training errors. """

class TrainingInterruptedException(Exception):
    """ Exception for when training is interrupted. """

    def __init__(self, epoch=0, batch=0):
        self.epoch = epoch
        self.message = f'training was interrupted by user at epoch {epoch}.'
        self.batch = batch
        if batch > 0:
            self.message = self.message.replace('.', f', batch {batch}.')

    def __str__(self):
        return 'TrainingInterrupted, {0} '.format(self.message)

class TrainingErrorException(Exception):
    """ Exception for when training runs into an error. """
    def __init__(self, error=None, epoch=0, batch=0):
        self.error = error
        self.epoch = epoch
        self.message = f'training was interrupted due to an error at epoch {epoch}.'
        self.batch = batch
        if batch > 0:
            self.message = self.message.replace('.', f', batch {batch}.')
        if not error is None:
            self.message = self.message + f'\n{error}'

    def __str__(self):
        return 'TrainingError, {0} '.format(self.message)