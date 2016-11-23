import logging
from tqdm import tqdm

from visuals import Visuals


def evaluate_agent(args, agent, test_emulator, test_stats):
    logging.info("Evaluating agent performace in test emulator")
    step = 0
    total_reward = 0.0
    reset = test_emulator.reset()
    agent.test_state = list(next(zip(*reset)))
    screen = test_emulator.preprocess()
    visuals = None
    if args.watch:
        visuals = Visuals(test_emulator.get_possible_actions())

    # either play as many steps as possible or as many games
    for _ in tqdm(range(args.test_games), unit="game"):
        while not test_emulator.isGameOver() and step < args.test_steps:
            action, q_values = agent.test_step(screen)
            screen, action, reward, terminal, raw_reward = test_emulator.run_step(action)
            total_reward += raw_reward

            # record stats
            if not (test_stats is None):
                test_stats.add_reward(raw_reward)
                if not (q_values is None):
                    test_stats.add_q_values(q_values)
                # endif
            # endif

            # update visuals
            if args.watch and (not (q_values is None)):
                visuals.update(q_values)

            step += 1
        # endwhile
        if not (test_stats is None):
            test_stats.add_game()
        reset = test_emulator.reset()
        agent.test_state = list(next(zip(*reset)))

    return total_reward / args.test_games
