import gym
import tensorflow as tf
import numpy as np
import random
import QEstimator

class Agent():
    def __init__(self,envName):
        self.env=gym.make(envName)

    def train(self):
        return 0

    def predict(self):
        return 1

    def saveNetwork(self):
        return True