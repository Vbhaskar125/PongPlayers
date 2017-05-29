from collections import deque, namedtuple
import numpy as np

experience=namedtuple('experience','screen reward')



class ExpMemory():
    def __init__(self,MemLen):
        self.Mem=deque(maxlen=MemLen)

    def add(self,screen,reward):
        if(reward != None & screen !=None):
            self.Mem.append(experience(screen=screen,reward=reward) )
        else:
            print('Incomplete Memory instance')

    def sample(self):
        return self.Mem.pop()
