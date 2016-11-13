# Preparation for workshop

On your personal laptop, please make sure to:

- create an isolated environment (you can use virtualenv or conda, however you like):

  		$ pip install virtualenv
  		# put all the files in a location you like, we'll stick with ~/tf_env for simplicity
  		$ virtualenv -p /usr/bin/python3.5 ~/tf_env
  		$ source ~/tf_env/bin/activate

- install python 3.5 into virtualenv with following packages

		# Put these into text file requirements.txt and execute in your virtualenv
		$ pip install -r requirements.txt
		# (there might be some extra packages, it is a list I have on my laptop)

- [install Tensorflow in virtualenv](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#virtualenv-installation)
- install ALE - [atari learning environment](http://www.arcadelearningenvironment.org/)
	- Install necessary dependencies:

			sudo apt-get install libsdl-gfx1.2-dev libsdl-image1.2-dev libsdl1.2-dev cmake

	- Clone and build ALE:

			git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git
			cd Arcade-Learning-Environment
			cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON .
			make -j 4
			sudo make install
			sudo pip install .

- install [OpenAI Gym](https://gym.openai.com/docs)

		pip install gym
		# note: I had to update gym source files, to swap if/elif conditions
		# for ffmpeg/avconv. I issued PR to the gym but I'm not sure if they
		# will update it. If this fails for you as well, you can update the code
		# or clone my repo at https://github.com/michalsustr/gym
		# Diff:
		# https://github.com/openai/gym/compare/master...michalsustr:master#diff-54b89e317dc6e7d9dfd407344cafd1bf
		pip install gym[atari]

- optionally: [Set up TensorFlow on AWS GPU](https://github.com/gtoubassi/dqn-atari/wiki/Setting-up-TensorFlow-on-AWS-GPU)

# Test setup
- You can test your ALE setup by launching script

		$ python ale_example.py ./space_invaders.bin

- Test tensorflow (can take a while to run for the first time)

		$ python tf_example.py

- Test gym - get your API key in the [https://gym.openai.com/](gym) (by signing in with github account)
  and update the `gym_example.py` file.

		$ python gym_example.py

	You should get a reference link to your evalution board

		2016-10-02 21:18:26,920 [MainThread  ][INFO ]:
		****************************************************
		You successfully uploaded your evaluation on CartPole-v0 to
		OpenAI Gym! You can find it at:

			https://gym.openai.com/evaluations/eval_8ZzrWOlRICX3ynLBTQ8A

		****************************************************

	Please send this link to my e-mail address `michal.sustr at you know gmail.com` with title
	`[RL_workshop] gym link` so that we know how many people actually read this and prepared themselves :-)
