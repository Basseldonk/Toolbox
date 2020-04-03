#!/usr/bin/env python3
"""
Training
"""
import abc
import torch
from torch.optim import SGD

class DefaultTrainLoop:
    """
    A default pytorch training loop.
    
    kwargs:
        model (req)                 Model to train.
        train_loader (req)          Dataloader that provides training data.
        criterion (req)             Loss criterion(s) to use duing training.
        optimiser                   Optimiser to use during training.
                                    Default is SGD with a lr of 0.001 and momentum of 0.9
        val_loader                  Dataloader that provides validation data.
        nr_of_epochs                Number of epochs to train for.
                                    Default is 100.
    """

    def __init__(self, *args, **kwargs):
        """ Runs default training loop. """
        # Create attributes
        self.model = None
        self.train_loader = None
        self.criterion = None
        self.optimiser = None
        self.val_loader = {}
        self.nr_of_epochs = 100

        # Parse arguments
        self._set_attributes(*args, **kwargs)

    def _set_attributes(self, *args, **kwargs):
        """ Stores args and kwargs as attributes. """
        # Construct dictionary with default values
        input_dict = vars(self)

        # Fill in args
        for (key, value) in zip(input_dict.keys(), args):
            input_dict[key] = value

        # Fill in kwargs
        for (key, value) in kwargs.items():
            if key in input_dict.keys():
                input_dict[key] = value

        # Assert that critical values are filled in
        assert input_dict['model'] is not None, \
            "Model not specified."
        assert input_dict['train_loader'] is not None, \
            "Train_loader not specified."
        assert input_dict['criterion'] is not None, \
            "Criterion not specified."

        # Set defaults if not set
        if input_dict['optimiser'] is None:
            input_dict['optimiser'] = SGD(
                input_dict['model'].parameters(),
                lr=0.00005,
                momentum=0.9
                )
        
        for attr_name, value in input_dict.items():
            setattr(self, attr_name, value)

    def on_start_of_training(self):
        """ This code gets executed at the start of training. """
        pass

    def on_start_of_epoch(self, epoch):
        """ This code gets executed at the start of each epoch. """
        pass

    def on_start_of_batch(self, epoch, batch, data):
        """ This code gets executed at the start of each batch. """
        pass

    def on_end_of_batch(self, epoch, batch, data):
        """ This code gets executed at the end of each batch. """
        pass

    def on_end_of_epoch(self, epoch):
        """ This code gets executed at the end of each epoch. """
        pass

    def on_end_of_training(self):
        """ This code gets executed at the end of training. """
        pass

    def on_keyboard_interrupt(self, epoch=0, batch=0):
        """ This code gets executed when a KeyboardInterrupt exception was caught. """
        pass

    def _default_training_loop(self, start_at_epoch=1, start_at_batch=1):
        """ A default pytorch training loop. """
        # On start of training hook
        if start_at_epoch == 1 and start_at_batch == 1:
            self.on_start_of_training()
        # Loop over epochs
        for epoch in range(start_at_epoch, self.nr_of_epochs+1):
            try:
                # On start of epoch hook
                self.on_start_of_epoch(epoch)
                # Training
                for batch, data in enumerate(self.train_loader, 1):
                    if batch < start_at_batch:
                        print(f'skipped batch {batch}', end='\r')
                        continue
                    try:
                        # On start of batch hook
                        self.on_start_of_batch(epoch, batch, data)
                        # On end of batch hook
                        self.on_end_of_batch(epoch, batch, data)
                    except KeyboardInterrupt:
                        self.on_keyboard_interrupt(epoch, batch)
                # On end of epoch hook
                self.on_end_of_epoch(epoch)
            except KeyboardInterrupt:
                self.on_keyboard_interrupt(epoch)
        # On end of training hook
        self.on_end_of_training()
