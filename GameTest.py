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
for x in range(200
               ):
    if c==True:
        env.reset()
        c=False
    rds=ra.play()
    a,b,c,d=env.step(rds)
    imgcc = Image.fromarray(a, 'RGB')
    imgcc.save('im'+str(x)+'g.JPG')
    env.render()
