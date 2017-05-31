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
        populate replay mem by using randomAgent and Qnet
        train on it

    predict
        predict the next action






