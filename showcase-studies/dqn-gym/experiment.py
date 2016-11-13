import logging
from visuals import Visuals

def evaluate_agent(args, agent, test_emulator, test_stats):
	step = 0
	games = 0
	reward = 0.0
	reset = test_emulator.reset()
	agent.test_state = list(next(zip(*reset)))
	screen = test_emulator.preprocess()
	visuals = None
	if args.watch:
		visuals = Visuals(test_emulator.get_possible_actions())

	while (step < args.test_steps) and (games < args.test_episodes):
		while not test_emulator.isGameOver() and step < args.test_steps_hardcap:
			action, q_values = agent.test_step(screen)
			results = test_emulator.run_step(action)
			screen = results[0]
			reward += results[4]

			# record stats
			if not (test_stats is None):
				test_stats.add_reward(results[4])
				if not (q_values is None):
					test_stats.add_q_values(q_values)
				# endif
			#endif

			# update visuals
			if args.watch and (not (q_values is None)):
				visuals.update(q_values)

			step +=1
		# endwhile
		games += 1
		if not (test_stats is None):
			test_stats.add_game()
		reset = test_emulator.reset()
		agent.test_state = list(next(zip(*reset)))

	return reward / games



def run_experiment(args, agent, test_emulator, test_stats):


	agent.run_random_exploration()

	for epoch in range(1, args.epochs + 1):

		if epoch == 1:
			agent.run_epoch(args.epoch_length - agent.random_exploration_length, epoch)
		else:
			agent.run_epoch(args.epoch_length, epoch)

		results = evaluate_agent(args, agent, test_emulator, test_stats)
		logging.info("Score for epoch {0}: {1}".format(epoch, results))
		steps = 0
		if args.parallel:
			steps = agent.random_exploration_length + (agent.train_steps * args.training_frequency)
		else:
			steps = agent.total_steps

		test_stats.record(steps)
		if results >= args.saving_threshold:
			agent.save_model(epoch)
		