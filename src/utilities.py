from __future__ import print_function, division

__author__ = 'amrit'

from random import randint, uniform, choice, sample

def _randint(a=0,b=0):
    return randint(a,b)

def _randchoice(a=[]):
    return choice(a)

def _randuniform(a=0.0,b=0.0):
    return uniform(a,b)

def _randsample(a=[],b=1):
    return sample(a,b)

def unpack(l):
    tmp=[]
    for i in l:
        if list!=type(i):
            tmp.append(i)
        else:
            for x in i:
                tmp.append(x)
    return tmp