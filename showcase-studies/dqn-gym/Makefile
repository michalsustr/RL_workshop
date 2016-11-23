train:
	# use default params
	python3 ./run_dqn.py breakout dqn brick_hunter

quick_train:
	python ./run_dqn.py breakout dqn quick_train --memory_capacity 100000 --training_length=1000 --random_exploration_length=400 --test_games=3 --test_frequency=250
quick_watch:
	python ./run_dqn.py breakout dqn quick_train --watch --test_games=3