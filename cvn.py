import pandas as pd
import numpy as np

import h5Utils
import pmUtils
import model
import generator
from config import Config

import argparse
import os

from keras.optimizers import SGD
from keras.utils import plot_model
from keras.models import load_model

def nuCut(tables):
    df = tables['rec.mc']
    return (df.nnu > 0)

class novaConfig(Config):
    weights_name = 'cvn_weights'
    epochs = 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train with h5s!')
    parser.add_argument('command',metavar='<command>',help='train or view')
    args=parser.parse_args()

    if args.command == 'train':
        config = novaConfig()

        tables = h5Utils.importh5(config.input_file)
        df = pmUtils.trainingdf(tables,nuCut(tables))

        train, val = pmUtils.dfsplit(df,frac=config.test_size)

        train_generator = generator.data_generator(train,config.batch_size,config.num_classes)
        val_generator = generator.data_generator(val,config.batch_size,config.num_classes)

        model = model.CVN_model(config.num_classes)
        opt = SGD(lr=config.learning_rate, momentum=config.momentum, decay=config.decay_rate, nesterov=False)
        model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['acc','top_k_categorical_accuracy'])

        history = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=config.train_iterations,
                                      validation_data=val_generator,
                                      validation_steps=config.val_iterations,
                                      epochs=config.epochs,
                                      max_queue_size=1,
                                      verbose=1,
                                      workers=1)

        directory = config.out_directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        model.save(directory + config.weights_name + '.h5')

    if args.command == 'predict':
        config = novaConfig()

        tables = h5Utils.importh5(config.input_file)
        df = pmUtils.trainingdf(tables,nuCut(tables))
        df = df.sample(frac=0.2)

        directory = config.out_directory
        model = load_model(directory + config.weights_name + '.h5')

        maps = pmUtils.pmdftonp(df.cvnmap)
        cats = df.intcat.values

        for cat,map in zip(cats,maps):
            probs = model.predict([[map[0]],[map[1]]])
            print(cat,probs)

    if args.command == 'viewdata':
        config = novaConfig()

        tables = h5Utils.importh5(config.input_file)
        pmUtils.viewmap(tables,nuCut(tables))

    if args.command == 'viewmodel':
        config = novaConfig()

        model = model.CVN_model(config.num_classes)

        directory = config.out_directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        plot_model(model, to_file=directory+'model.png')
