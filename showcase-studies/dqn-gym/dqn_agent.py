import experiment
from visuals import Visuals


import random
import logging
import numpy as np
from tqdm import tqdm


class DQNAgent():
    def __init__(self, args, q_network,
                 train_emulator, test_emulator,
                 experience_memory, num_actions,
                 train_stats, test_stats):

        self.network = q_network
        self.train_emulator = train_emulator
        self.test_emulator = test_emulator

        self.memory = experience_memory
        self.train_stats = train_stats
        self.test_stats = test_stats

        self.num_actions = num_actions
        self.history_length = args.history_length

        self.training_frequency = args.training_frequency
        self.random_exploration_length = args.random_exploration_length
        self.training_length = args.training_length
        self.initial_exploration_rate = args.initial_exploration_rate
        self.final_exploration_rate = args.final_exploration_rate
        self.final_exploration_frame = args.final_exploration_frame
        self.test_exploration_rate = args.test_exploration_rate
        self.recording_frequency = args.recording_frequency
        self.test_frequency = args.test_frequency

        self.exploration_rate = self.initial_exploration_rate
        self.total_steps = 0

        self.args = args

        self.test_state = []

        logging.info("DQN Agent Initialized")

    def choose_action(self):
        if random.random() >= self.exploration_rate:
            state = self.memory.get_current_state()
            q_values = self.network.inference(state)
            self.train_stats.add_q_values(q_values)
            return np.argmax(q_values)
        else:
            return random.randrange(self.num_actions)

    def checkGameOver(self):
        if self.train_emulator.isGameOver():
            initial_state = self.train_emulator.reset()
            for experience in initial_state:
                self.memory.add(experience[0], experience[1], experience[2],
                                experience[3])
            self.train_stats.add_game()

    def run_random_exploration(self):
        for step in tqdm(range(self.random_exploration_length)):
            state, action, reward, terminal, raw_reward = self.train_emulator.run_step(
                random.randrange(self.num_actions))
            self.train_stats.add_reward(raw_reward)
            self.memory.add(state, action, reward, terminal)
            self.checkGameOver()
            self.total_steps += 1
            if (self.total_steps % self.recording_frequency == 0):
                self.train_stats.record(self.total_steps)

    def run_training(self):
        # show pbars only if not evaluating agent
        pbar = tqdm()
        for step in range(self.training_length):
            # test agent
            if step % self.test_frequency == 0:
                pbar.close()
                experiment.evaluate_agent(self.args, self, self.test_emulator, self.test_stats)
                self.save_model(step)
                logging.info("Training... (%d/%d) " % (step, self.training_length))
                pbar = tqdm(total=min(self.test_frequency, self.training_length), unit="step")
            pbar.update(1)

            # play step
            state, action, reward, terminal, raw_reward = self.train_emulator.run_step(
                self.choose_action())
            self.train_stats.add_reward(raw_reward)
            self.memory.add(state, action, reward, terminal)
            self.checkGameOver()

            # training
            if self.total_steps % self.training_frequency == 0:
                states, actions, rewards, next_states, terminals = self.memory.get_batch()
                loss = self.network.train(states, actions, rewards, next_states,
                                          terminals)
                self.train_stats.add_loss(loss)

            self.total_steps += 1

            if self.total_steps < self.final_exploration_frame:
                self.exploration_rate -= (
                                         self.exploration_rate - self.final_exploration_rate) / (
                                         self.final_exploration_frame - self.total_steps)

            if self.total_steps % self.recording_frequency == 0:
                self.train_stats.record(self.total_steps)
                self.network.record_params(self.total_steps)

        pbar.close()

    def test_step(self, observation):
        if len(self.test_state) < self.history_length:
            self.test_state.append(observation)

        # choose action
        q_values = None
        action = None
        if random.random() >= self.test_exploration_rate:
            state = np.expand_dims(np.transpose(self.test_state, [1, 2, 0]),
                                   axis=0)
            q_values = self.network.inference(state)
            action = np.argmax(q_values)
        else:
            action = random.randrange(self.num_actions)

        self.test_state.pop(0)
        return [action, q_values]

    def save_model(self, step):
        self.network.save_model(step)

    def run_experiment(self):
        logging.info("Running random exploration")
        self.run_random_exploration()

        self.train_emulator.reset()
        self.run_training()
