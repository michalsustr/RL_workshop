# Preparation for workshop

On your personal laptop, please make sure to:

- install python 3.5 into virtualenv with following packages (or you can use conda, whatever system you prefer):

		# Put these into text file requirements.txt and execute in your virtualenv
		$ pip install -r requirements.txt
		# (there might be some extra packages, this is a list I have on my laptop)

		alabaster (0.7.8)
		arch (3.2)
		Babel (2.3.3)
		blosc (1.4.4)
		cycler (0.10.0)
		dask (0.11.0)
		decorator (4.0.10)
		descartes (1.0.2)
		docutils (0.12)
		filterpy (0.1.3)
		humanize (0.5.1)
		imagesize (0.7.1)
		ipykernel (4.3.1)
		ipympl (0.0.2)
		ipython (5.0.0)
		ipython-genutils (0.1.0)
		ipywidgets (5.2.2)
		jsonschema (2.5.1)
		jupyter (1.0.0)
		jupyter-client (4.3.0)
		jupyter-console (5.0.0)
		jupyter-contrib-core (0.3.0)
		jupyter-core (4.1.0)
		jupyter-nbextensions-configurator (0.2.1)
		Markdown (2.6.6)
		MarkupSafe (0.23)
		matplotlib (2.0.0b3)
		mistune (0.7.2)
		mpld3 (0.2)
		mpmath (0.19)
		nbconvert (4.2.0)
		nbformat (4.0.1)
		networkx (1.11)
		notebook (4.2.1)
		numpy (1.11.1)
		pandas (0.18.1)
		patsy (0.4.1)
		pexpect (4.0.1)
		pickleshare (0.7.2)
		Pillow (3.3.1)
		pip (8.1.2)
		plotly (1.12.9)
		prompt-toolkit (1.0.3)
		protobuf (3.0.0b2.post2)
		ptyprocess (0.5.1)
		pyparsing (2.1.4)
		python-dateutil (2.5.3)
		pytz (2016.4)
		PyYAML (3.11)
		pyzmq (15.2.0)
		qtconsole (4.2.1)
		recommonmark (0.4.0)
		requests (2.10.0)
		scikit-image (0.12.3)
		scikit-learn (0.17.1)
		scipy (0.17.1)
		seaborn (0.7.1)
		setuptools (23.0.0)
		simplegeneric (0.8.1)
		snowballstemmer (1.2.1)
		Sphinx (1.4.1)
		sphinx-rtd-theme (0.1.9)
		statsmodels (0.6.1)
		sympy (1.0)
		tensorflow (0.9.0)
		terminado (0.6)
		toolz (0.8.0)
		tornado (4.3)
		tqdm (4.7.6)
		wheel (0.29.0)

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
