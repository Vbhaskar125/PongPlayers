import gym
import tensorflow as tf
import numpy as np
import random
#import QEstimator
import ReplayMemory
import Preprocess

class GameAgent():
    def __init__(self,envName,MemSize):
        self.env=gym.make(envName)
        #self.Qnet=QEstimator.qnet()
        self.replayMemory=ReplayMemory.memory(MemSize)

    def startEpisode(self):
        self.env.reset()

    def PopulateReplayMemory(self):
        NumberofInstances=self.replayMemory.getMemoryCapacity()
        for instances in range(0,NumberofInstances):
            screen=reward=Terminated=0
            screenarr=[]
            rewardarr=[]
            self.startEpisode()
            for x in range(0,4):
                if Terminated==False:
                    screen,reward,Terminated,_=self.env.step(self.env.action_space.sample())
                    screenarr.append(screen)
                    rewardarr.append(reward)
                else:
                    screenarr, rewardarr = []
                    self.startEpisode()
            cumulativeReward=np.sum(rewardarr)
            ProcessedScreen=Preprocess.preprocess(screenarr[0],screenarr[1],screenarr[2],screenarr[3])
            self.replayMemory.add(screen=ProcessedScreen,reward=cumulativeReward)
        print('Done')



