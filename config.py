'''
Default config file for changing settings when training a model
Use this as a base class for specialized training
'''
class Config(object):
    # number of samples in each iteration
    batch_size       = 16

    # number of batches to process in each epoch
    train_iterations = 100

    # number of validation batches to process at the end of each epoch
    val_iterations   = 5

    # number of epochs to train for
    epochs           = 5

    # where to find the training file
    input_file       = 'fardet_genie_nonswap_genierw_fhc_v08_1000_r00014041_s60_c000_R17-11-14-prod4reco.h5'

    # number of output classes
    num_classes      = 5

    # fraction of events to put in the testing sample
    test_size        = 0.2

    # the network learning rate
    learning_rate    = 0.02

    # the network decay rate
    decay_rate       = 0.1

    # the network learning momentum
    momentum         = 0.9

    # where to store the output
    out_directory    = 'logs/'

    # name of the file to store weights in
    weights_name     = 'nova_weights'

    # Display the config
    def display(self):
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
