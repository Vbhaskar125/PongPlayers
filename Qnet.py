import tensorflow as tf
import numpy as np
import ReplayMemory
from PIL import Image

class Qnet():
    def __init__(self,memlen,pathTosaveWeights,gamma,env):
        self.Gamma=gamma
        self.env=env
        self.actions=self.env.action_space.n
        self.replaymem=ReplayMemory.memory(memlen)
        self.path=pathTosaveWeights
        self.weights={'w1':tf.Variable(tf.random_normal([8,8,1,32],mean=0,stddev=0.1)),
                      'w2': tf.Variable(tf.random_normal([4, 4, 32, 64], mean=0, stddev=0.1)),
                      'w3': tf.Variable(tf.random_normal([2, 2, 64, 64], mean=0, stddev=0.1)),
                      'w4':tf.Variable(tf.random_normal([1600,512],mean=0.0,stddev=0.1)),
                      'w5':tf.Variable(tf.random_normal([512,self.actions],mean=0.0,stddev=0.1))
                      }
        self.biases={
            'b1':tf.Variable(tf.constant(0.1,dtype="float",shape=[32])),
            'b2' :tf.Variable(tf.constant(0.1, dtype="float", shape=[64])),
            'b3': tf.Variable(tf.constant(0.1, dtype="float", shape=[64])),
            'b4':tf.Variable(tf.constant(0.1,dtype="float",shape=[512])),
            'b5': tf.Variable(tf.constant(0.1, dtype="float", shape=[self.actions])),
        }

    def buildModel(self):
        inp=tf.placeholder(dtype="float",shape=[None,80,80,1])
        Convlayer1=tf.nn.relu(tf.nn.conv2d(inp,self.weights['w1'],strides=[1,4,4,1],padding="SAME")+self.biases['b1'])
        Convlayer1=tf.nn.max_pool(Convlayer1,ksize=[1,2,2,1],strides=[1,2,2 ,1],padding="SAME")
        Convlayer2=tf.nn.relu(tf.nn.conv2d(Convlayer1,self.weights['w2'],strides=[1,2,2,1],padding="SAME")+self.biases['b2'])
        Convlayer3 = tf.nn.relu(tf.nn.conv2d(Convlayer2, self.weights['w3'], strides=[1, 1, 1, 1], padding="SAME")+self.biases['b3'])
        fcFlat=tf.reshape(Convlayer3,[1,1600])
        FullConnected1=tf.nn.relu(tf.matmul(fcFlat,self.weights['w4'])+self.biases['b4'])
        outpt=tf.matmul(FullConnected1,self.weights['w5'])
        return outpt,inp

    def train(self,numberOfPasses,outpt,inp,sess):
        #populate the replay memory
        self.env.reset()
        #save the weights
        #getting the first instance
        a=self.env.action_space.sample()
        s_0,r_0,t=self.getExperience(a)
        s_0=self.preprocess(s_0[0],s_0[1],s_0[2],s_0[3])
        s_0 = np.asarray(s_0).reshape([1, 80, 80, 1])
        r_0=np.sum(r_0)/4

        #populating Memory
        sess.run(tf.global_variables_initializer())
        for x in range(0,self.replaymem.memlen):
            pred_act=np.argmax(sess.run(outpt,feed_dict={inp:s_0}))
            s_next,r_next,terminal=self.getExperience(pred_act)
            s_t=self.preprocess(s_next[0],s_next[1],s_next[2],s_next[3])
            s_t = np.asarray(s_t).reshape([1, 80, 80, 1])
            self.replaymem.addExperience(s_0,pred_act,r_0,s_t,terminal)
            s_0=s_t
            r_0=np.sum(r_next)/4

        a = tf.placeholder("float", [None, self.actions])
        y = tf.placeholder("float", [None])
        readout_action = tf.reduce_sum(tf.multiply(outpt, a), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y - readout_action))
        train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

        #train using the memory and updating it
        for x in range(0,numberOfPasses):
            experienceSample=self.replaymem.sampleMemory(10)
            sini = [e[0] for e in experienceSample]
            acini = [e[1] for e in experienceSample]
            rini = [e[2] for e in experienceSample]
            term=[e[4] for e in experienceSample]
            sfin=[e[3] for e in experienceSample]
            yBatch=[]
            acfin=sess.run(outpt,feed_dict={inp:sfin})
            for v in range(0,len(experienceSample)):
                termcheck=term[v]
                if termcheck:
                    yBatch.append(rini[v])
                else:
                    yBatch.append(rini[v]+self.Gamma*np.max(acfin[v]))
                    nextact=np.argmax(acfin[v])
                    s_next, r_next, terminalnext=self.getExperience(nextact)
                    s_t = self.preprocess(s_next[0], s_next[1], s_next[2], s_next[3])
                    s_t = np.asarray(s_t).reshape([1, 80, 80, 1])
                    self.replaymem.addExperience(sfin[v], nextact, r_next, s_next, terminalnext)


            sess.run(train_step,feed_dict={y:yBatch,a:acini,inp:sini})

            #save weights per 10000 runs
            if x % 10000 == 0:
                saver=tf.train.saver()
                saver.save(sess,str(self.path)+str(x),global_step=x)
                print('saved '+str(self.path)+str(x)+' instance trained weight')









    def getExperience(self,action):
        s=[]
        r=[]
        for x in range(0,4):
            s_0,r_0,T_0,_=self.env.step(action)
            s.append(s_0)
            r.append(r_0)
            if T_0==True:
                t=True
            else:
                t=False

        return s,r,t

    def preprocess(self,image1, image2, image3, image4):
        Imgarray = [image1, image2, image3, image4]
        processedImg = []
        for x in Imgarray:
            Colorimg = Image.fromarray(np.asarray(x), 'RGB')
            gray = Colorimg.convert("L")
            blackwhite = gray.point(lambda x: 0 if x < 128 else 255, '1')
            blackwhite = np.asarray(blackwhite)
            crop = blackwhite[32:198]
            processedImg.append(crop)

        one = Image.blend(Image.fromarray(processedImg[0], 'L'), Image.fromarray(processedImg[1], 'L'), 0.5)
        two = Image.blend(Image.fromarray(processedImg[2], 'L'), Image.fromarray(processedImg[3], 'L'), 0.5)
        blendedImg = Image.blend(one, two, 0.5)

        return blendedImg.resize([80,80])



    def play(self,sess,opt,inp):
        sess.run(tf.global_variables_initializer())
        self.env.reset()
        screen,_,t =self.getExperience(2)
        while t == False:
            scrImg=self.preprocess(screen[0],screen[1],screen[2],screen[3])
            actin=np.argmax(sess.run(opt,feed_dict={inp:scrImg}))
            screen,_,t=self.getExperience(action=actin)
            self.env.render()

