
import random
import gym
import numpy as np
from PIL import Image
import skimage

env = gym.make("Pong-v0")
env.reset()
c = False
for x in range(2):
    if c == True:
        env.reset()
    env.render()
    a, b, c, d = env.step(env.action_space.sample())
    #img = Image.fromarray(np.asarray(a), 'RGB')
    #gray = img.convert("L")
    #bw = gray.point(lambda x: 0 if x < 128 else 255, '1')
    #bw = np.asarray(bw)
    #crop = bw[32:198]
    #imgcc = Image.fromarray(crop)
    #imgcc.show()

   

'''
imgc=np.asarray(img[32:198])
imgc.flags.writeable = True
imgc[imgc==144]=000
#imgc[imgc==109]=0

imgcc = Image.fromarray(imgc, 'RGB')
imgcc.show()
Image.open('my8.png')'''
