import Qnet
import tensorflow as tf
import numpy as np
import gym
import ReplayMemory
rm=ReplayMemory.memory(4)
env=gym.make("Pong-v0")
env.reset()
asd=Qnet.Qnet(5,'/home/delta',4,env)
opt,inpt=asd.buildModel()
sess=tf.Session()
sess.run(tf.global_variables_initializer())
s,r,t=asd.getExperience(2)
ss=asd.preprocess(s[0],s[1],s[2],s[3])
ss=np.asarray(ss,dtype="float")
ss=ss.reshape([1,80,80,1])
rm.addExperience(ss,2,0,ss,t)
rm.addExperience(ss,2,0,ss,t)
rm.addExperience(ss,2,0,ss,t)
rm.addExperience(ss,2,0,ss,t)
'''s_0,r_0=asd.getExperience(2)
s_0=asd.preprocess(s_0[0],s_0[1],s_0[2],s_0[3])
#s_0=np.asarray(s_0).reshape([1,80,80,1])
sess=tf.Session()
sess.run(tf.global_variables_initializer())
print(asd.train(2,opt,inpt,sess))'''
smpl=rm.sampleMemory(3)

sini=[e[0] for e in smpl]
acini=[e[1] for e in smpl]
rini=[e[2] for e in smpl]

sini=np.reshape(sini,[3,80,80,1])

print(sini.shape)
