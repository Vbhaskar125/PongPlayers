start game
populate experience replay memory
save the graph
train the graph
save the graph
try to play


Game
GameAgent
    init
       env.reset
       allocate replay memory
       load/save network weights

    train
        populate replay mem by using randomAgent
        train on it

    predict
        predict the next action






 def preprocess(self, screen, new_game=False):
        """Converts to grayscale, resizes and stacks input screen.
        :param screen: array image in [0; 255] range with shape=[H, W, C]
        :param new_game: if True - repeats passed screen `memlen` times
                   otherwise - stacks with previous screens"
        :type screen: numpy.array
        :type new_game: bool
        :return: image in [0.0; 1.0] stacked with last `memlen-1` screens;
                shape=[1, h, w, memlen]
        :rtype: numpy.array"""
        gray = screen.astype('float32').mean(2) # no need in true grayscale, just take mean
        # convert values into [0.0; 1.0] range
        s = imresize(gray, (self.W, self.H)).astype('float32') * (1. / 255)
        s = s.reshape(1, s.shape[0], s.shape[1], 1)
        if new_game or self.stacked_s is None:
            self.stacked_s = np.repeat(s, self.memlen, axis=3)
        else:
            self.stacked_s = np.append(s, self.stacked_s[:, :, :, :self.memlen - 1], axis=3)
        return self.stacked_s