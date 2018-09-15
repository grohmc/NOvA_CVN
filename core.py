import numpy as np
import pandas as pd

class spectrum():
    def __init__(self, var, cut, tables):
        #self._POT = 0
        self._tables = tables
        self._var = var(tables)
        self._cut = np.all(pd.concat([c(tables) for c in cut], axis=1), axis=1)

    #def fill(self): I can't think of a reason to do it this way...
        self._df = self._var[self._cut]
        self._POT = self._tables['spill'].spillpot.sum()

    def POT(self):
        return self._POT

    def df(self):
        return self._df

    def histogram(self, bins=None, range=None, POT=None):
        if not POT:
            POT = self._POT
        if bins:
            n, bins = np.histogram(self._df, bins, range)
        else:
            n, bins = np.histogram(self._df)

        return n*POT/self._POT, bins

    def entries(self):
        return self._df.shape[0]

    def integral(self,POT=None):
        if not POT:
            POT = self._POT
        return self.entries()*POT/self._POT

class cut():
    def __init__(self, cut, invert=False):
        if type(cut) is not list: cut = [cut]
        if type(invert) is not list: invert = [invert]
        assert len(cut) == len(invert)

        self._cut = cut
        self._invert = invert

    def cut(self, tables):
        return np.all(pd.concat([(~c(tables) if b else c(tables)) for c, b in zip(self._cut, self._invert)], axis=1), axis=1)

    def __and__(self, other):
        return cut(self._cut + other._cut, self._invert + other._invert)

    def __invert__(self):
        return cut(self._cut, [not b for b in self._invert])
