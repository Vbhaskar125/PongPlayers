import gym
import numpy as np
from PIL import Image
import skimage
import random


env=gym.make("Pong-v0")

class randomAgent():
    def play(self):
        return env.action_space.sample()


#start the game and collect the images
env.reset()
ra=randomAgent()
c=False
for x in range(20000):
    if c==True:
        env.reset()
        c=False
    rds=ra.play()
    for x in range(1):
        a,b,c,d=env.step(rds)
    env.render()
