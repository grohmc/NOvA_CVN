import os, sys
import re
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('command', metavar='<command>', help='input log file')
args = parser.parse_args()
log = args.command

def extractLoss(logPath):
    iter = []
    loss = []
    acc  = []

    eiter = []
    tloss = []
    tacc  = []
    vloss = []
    vacc  = []

    for line in open(logPath):
        # grab all training values for every iteration
        m = re.search('- loss: (\d*.\d*) - acc: (\d*.\d*)', line)
        if m:
            loss.append(m.group(1))
            acc.append(m.group(2))
            iter.append(len(loss))

        # grab everything if it's the end of an epoch
        m = re.search('- loss: (\d*.\d*) - acc: (\d*.\d*) - .* - val_loss: (\d*.\d*) - val_acc: (\d*.\d*)', line)
        if m:
            tloss.append(m.group(1))
            tacc.append(m.group(2))
            vloss.append(m.group(3))
            vacc.append(m.group(4))
            eiter.append(len(loss))

    return np.array(iter).astype(np.int),\
        np.array(loss).astype(np.float),\
        np.array(acc).astype(np.float),\
        np.array(eiter).astype(np.int),\
        np.array(tloss).astype(np.float),\
        np.array(tacc).astype(np.float),\
        np.array(vloss).astype(np.float),\
        np.array(vacc).astype(np.float)

iter, loss, acc, eiter, tloss, tacc, vloss, vacc = extractLoss(log)

# Every iteration
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(iter,loss,color='xkcd:pink',label="Train Loss")
ax2.plot(iter,acc,color='xkcd:green',label="Train Acc")

ax1.plot(eiter,vloss,color='xkcd:dark pink',label="Val Loss")
ax2.plot(eiter,vacc,color='xkcd:dark green',label="Val Acc")

ax1.axis([0,iter[-1],0,2])
ax1.set_xlabel('iterations')
ax1.set_ylabel('loss')
ax2.axis([0,iter[-1],0,1.2])
ax2.set_ylabel('accuracy')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.savefig('loss_iter.png')
plt.show()

# Every epoch
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(eiter,tloss,color='xkcd:pink',label="Train Loss")
ax2.plot(eiter,tacc,color='xkcd:green',label="Train Acc")

ax1.plot(eiter,vloss,color='xkcd:dark pink',label="Val Loss")
ax2.plot(eiter,vacc,color='xkcd:dark green',label="Val Acc")

ax1.axis([0,iter[-1],0,2])
ax1.set_xlabel('iterations')
ax1.set_ylabel('loss')
ax2.axis([0,iter[-1],0,1.2])
ax2.set_ylabel('accuracy')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.savefig('loss_epoch.png')
plt.show()
