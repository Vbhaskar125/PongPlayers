from collections import deque, namedtuple
import numpy as np

experience=namedtuple('experience','screen reward')



class memory():
    def __init__(self,MemLen):
        self.Mem=deque(maxlen=MemLen)

    def add(self,screen,reward):
        self.Mem.append(experience(screen=screen,reward=reward))

    def sample(self):
        return self.Mem.pop()

    def getMemoryCapacity(self):
        return self.Mem.maxlen

