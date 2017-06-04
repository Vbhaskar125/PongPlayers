from collections import deque, namedtuple
import random


experience=namedtuple('experience','s_t a_t r s_t1 terminal')

class memory():
    def __init__(self,memlen):
        self.memlen=memlen
        self.replaymem=deque(maxlen=memlen)

    def addExperience(self,s_t,a_t,r,s_t1,terminated):
        self.replaymem.append(experience(s_t,a_t,r,s_t1,terminated))

    def sampleMemory(self,size):
        return random.sample(self.replaymem,size)

    def getMemCapacity(self):
        return self.memlen