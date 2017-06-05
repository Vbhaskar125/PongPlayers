import tensorflow as tf
import gym
import numpy as np
import Qnet

sess=tf.Session()
env=gym.make("Pong-v0")
pongplayer=Qnet.Qnet(memlen=50000,pathTosaveWeights="/home/pongplayerweights",gamma=0.2,env=env)
outpt,inpt=pongplayer.buildModel()
print(outpt.shape)
print(inpt.shape)
pongplayer.train(numberOfPasses=200000,outpt=outpt,inp=inpt,sess=sess,samplesize=10)
pongplayer.play(sess=sess,outpt=outpt,inp=inpt)
