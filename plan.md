# Wokshop plan

## PR

- [x] MLMU
- [x] FEL OI
- [x] FIT OI - public group na fb

### Ad text:

Zajímá vás umělá inteligence nebo strojové učení? Jak lze vytvořit umělou
inteligenci, která se naučí hrát počítačové hry přímo z obrazovky? Nebo jak
dokázal počítač porazit člověka v starodávné čínské hře Go, výpočetně o mnoho
těžší než šachy? Rádi by jste si vyzkoušeli něco takového naprogramovat a jěště
mít možnost vyhrát zajímavé ceny? Zúčastněte se workshopu o žhavém výzkumním
tématu - reinforcement learning, který se odehrá 8. 10. v budově FITu ČVUT.
[Přihlásit se a nalézt více informací můžete na stránkách workshopu](http://lectures.ai).

## Contents

|-----------------|-------|-----------------------------------------|-------|
| Time            | Delta | Activity                                | Who   |
|-----------------|-------|-----------------------------------------|-------|
| 8:30 - 08:40    |  10m  | Workshop introduction                   | MS,JZ |
| 8:40 - 09:30    |  50m  | Basics with TensorFlow                  | JZ    |
| 9:30 - 10:00    |  30m  | Basics from game theory                 | MS    |
|                 |       | - minimax                               |       |
|                 |       | - alfa-beta pruning                     |       |
|                 |       | - Samuel checkers                       |       |
|                 |       | - MCTS, intro to bandits                |       |
|                 |       | - UCT (MCTS)                            |       |
|                 |       | - Markov chains, MDPs                   |       |
| *10:00 - 10:30* |  30m  | *Coffee break*                          |       |
| 10:30 - 11:00   |  30m  | Theory for convnets                     | JZ    |
| 11:00 - 12:00   |  1h   | Theory for RL                           | MS    |
|                 |       | - Temporal-difference learning          |       |
|                 |       | - Q-learning                            |       |
|                 |       | - DQN                                   |       |
| *12:00 - 13:00* |  1h   | *Lunch break*                           |       |
| 13:00 - 14:30   |  1.5h | Case studies                            | MS    |
|                 |       | - TD-Gammon                             |       |
|                 |       | - Atari games                           |       |
|                 |       | - Go playing                            |       |
|                 |       | - How it's at Google                    | JC?   |
| 14:30 - 16:30   |  2h   | Tutorials                               | MS    |
|                 |       | - Atari games                           |       |
|                 |       | - Gym environment                       |       |
| *16:30 - 17:00* |  30m  | *Coffee break*                          |       |
| 17:00 - 19:30   |  2.5h | Free session coding                     | MS,JZ |
| 19:30 - 20:00   |  30m  | Finalizing the day                      | MS,JZ |
|-----------------|-------|-----------------------------------------|-------|
| midnight on     |  8d   | Running models until deadline           |       |
| Sunday 16th     |       | Submit results.                         |       |
|-----------------|-------|-----------------------------------------|-------|


### Introduction (10m)
- Who we are, what have we done in the past, why we are doing this
- Mention participants profiles

### Tensorflow (50m)

- Description of it came to being, stars on Github :)
- Mention other frameworks, what they brought
	- Caffe (Berkley) - model zoo
	- Theano (Montreal) - automatic diff
	- Torch (FB) - Lua
	- Keras (one person) - wrapper, difficult for advanced things
- Comparison of speeds between frameworks?
- TF: Computational graph
- Examples:
	- Linear regression (training example)
	- Simple MLP example (MNIST)
	- Simple convnet (MNIST), provide pre-trained examples
- Talk about limitations: RNN, ?
- How easy it is with deployment, AWS access?

### Game theory (30m)

- Explain algos with pictures:
	- minimax
	- alfa-beta pruning
	- Samuel checkers
	- MCTS, intro to bandits
	- UCT (MCTS)
	- Markov chains, MDPs
	- MDPs formulation

### Theory for convnets (30m)

- architectures
- types of neurons
- pretrained models
- strides
- whatever else Honza comes up with :)

### Theory for RL (1h)

- Temporal-difference learning
- Q-learning
- DQN

### Case studies (1.5h)

- TD-Gammon
- Atari games paper
- Go playing

### Tutorials (2h)

- Atari games
- OpenAI gym


### Finalizing the day

- What next?
	- where bachelor/diploma thesis
	- research facilities
	- jobs
- Literature
	- recommended reading
- Other sources of information
	- twitter
	- reddit
	- newsletters
- Obligatory feedback (pull request on github with their feedbacks)

--------------------------------------------------------------------------------
                                More details
--------------------------------------------------------------------------------

#### Samuel checkers:
- https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node109.html
- http://www.i-programmer.info/images/stories/ComputerCreators/Samuel/SAM.JPG

#### TD-Gammon
- http://www.bkgm.com/articles/tesauro/tdl.html


	Learning Linear Concepts First
	A third key ingredient has been found by a close examination of the early phases of the learning process. As stated previously, during the first few thousand training games, the network learns a number of elementary concepts, such as bearing off as many checkers as possible, hitting the opponent, playing safe (i.e., not leaving exposed blots that can be hit by the opponent) and building new points. It turns out that these early elementary concepts can all be expressed by an evaluation function that is linear in the raw input variables. Thus what appears to be happening in the TD learning process is that the neural network first extracts the linear component of the evaluation function, while nonlinear concepts emerge later in learning. (This is also frequently seen in backpropagation: in many applications, when training a multilayer net on a complex task, the network first extracts the linearly separable part of the problem.)


#### ATARI
Linky na repa:

- https://github.com/asrivat1/DeepLearningVideoGames
- https://github.com/gliese581gg/DQN_tensorflow
- https://github.com/Jabberwockyll/deep_rl_ale
- https://github.com/gtoubassi/dqn-atari
	- napisat o snapshoty: https://github.com/gtoubassi

- vyskusat si atari play z githubu
- vyskusat veci z gymu

#### Q-learning
dobry zdroj?
- https://www.nervanasys.com/demystifying-deep-reinforcement-learning/
- https://github.com/asrivat1/DeepLearningVideoGames


#### What next
- https://deepmind.com/
- https://openai.com/
- https://research.google.com/teams/brain/residency/
- https://www.microsoft.com/en-us/research/careers/
- https://research.facebook.com/ai/
- http://www.xrce.xerox.com/About-XRCE/Internships
- Gitter: https://gitter.im/openai/
- Twitter: https://twitter.com/michal_sustr

#### Literature
- Reinforcement Learning: An Introduction [Sutton,Barto] https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html