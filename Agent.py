import tensorflow as tf
import gym
import numpy as np
import Qnet

sess=tf.Session()
env=gym.make("Pong-v0")
pongplayer=Qnet.Qnet(memlen=50000,pathTosaveWeights="/home/pongplayerweights",gamma=0.2,env=env)
outpt,inpt=pongplayer.buildModel()
pongplayer.train(numberOfPasses=200000,outpt=outpt,inp=inpt,sess=sess)
pongplayer.play(sess=sess,outpt=outpt,inp=inpt)
