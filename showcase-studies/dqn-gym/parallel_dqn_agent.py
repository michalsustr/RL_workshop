import logging
import random
import threading

import numpy as np


class ParallelDQNAgent():
    def __init__(self, args, q_network, emulator, experience_memory,
                 num_actions, train_stats):

        self.network = q_network
        self.emulator = emulator
        self.memory = experience_memory
        self.train_stats = train_stats

        self.num_actions = num_actions
        self.history_length = args.history_length
        self.training_frequency = args.training_frequency
        self.random_exploration_length = args.random_exploration_length
        self.initial_exploration_rate = args.initial_exploration_rate
        self.final_exploration_rate = args.final_exploration_rate
        self.final_exploration_frame = args.final_exploration_frame
        self.test_exploration_rate = args.test_exploration_rate
        self.recording_frequency = args.recording_frequency

        self.exploration_rate = self.initial_exploration_rate
        self.total_steps = 0
        self.train_steps = 0
        self.current_act_steps = 0
        self.current_train_steps = 0

        self.test_state = []
        self.epoch_over = False

    def choose_action(self):

        if random.random() >= self.exploration_rate:
            state = self.memory.get_current_state()
            q_values = self.network.inference(state)
            self.train_stats.add_q_values(q_values)
            return np.argmax(q_values)
        else:
            return random.randrange(self.num_actions)

    def checkGameOver(self):
        if self.emulator.isGameOver():
            initial_state = self.emulator.reset()
            for experience in initial_state:
                self.memory.add(experience[0], experience[1], experience[2],
                                experience[3])
            self.train_stats.add_game()

    def run_random_exploration(self):

        for step in range(self.random_exploration_length):

            state, action, reward, terminal, raw_reward = self.emulator.run_step(
                random.randrange(self.num_actions))
            self.train_stats.add_reward(raw_reward)
            self.memory.add(state, action, reward, terminal)
            self.checkGameOver()
            self.total_steps += 1
            self.current_act_steps += 1
            if (self.total_steps % self.recording_frequency == 0):
                self.train_stats.record(self.total_steps)

    def train(self, steps):

        for step in range(steps):
            states, actions, rewards, next_states, terminals = self.memory.get_batch()
            loss = self.network.train(states, actions, rewards, next_states,
                                      terminals)
            self.train_stats.add_loss(loss)
            self.train_steps += 1
            self.current_train_steps += 1

            if self.train_steps < (
                self.final_exploration_frame / self.training_frequency):
                self.exploration_rate -= (
                                         self.exploration_rate - self.final_exploration_rate) / (
                                         (
                                         self.final_exploration_frame / self.training_frequency) - self.train_steps)

            if ((
                    self.train_steps * self.training_frequency) % self.recording_frequency == 0) and not (
                step == steps - 1):
                self.train_stats.record(self.random_exploration_length + (
                self.train_steps * self.training_frequency))
                self.network.record_params(self.random_exploration_length + (
                self.train_steps * self.training_frequency))

        self.epoch_over = True

    def run_epoch(self, steps, epoch):

        self.epoch_over = False
        threading.Thread(target=self.train,
                         args=(int(steps / self.training_frequency),)).start()

        while not self.epoch_over:
            state, action, reward, terminal, raw_reward = self.emulator.run_step(
                self.choose_action())
            self.memory.add(state, action, reward, terminal)
            self.train_stats.add_reward(raw_reward)
            self.checkGameOver()

            self.total_steps += 1
            self.current_act_steps += 1

        logging.info("act_steps: {0}".format(self.current_act_steps))
        logging.info("learn_steps: {0}".format(self.current_train_steps))
        self.train_stats.record(self.random_exploration_length + (
        self.train_steps * self.training_frequency))
        self.network.record_params(self.random_exploration_length + (
        self.train_steps * self.training_frequency))
        self.network.save_model(epoch)
        self.current_act_steps = 0
        self.current_train_steps = 0

    def test_step(self, observation):

        if len(self.test_state) < self.history_length:
            self.test_state.append(observation)

        # choose action
        q_values = None
        action = None
        if random.random() >= self.test_exploration_rate:
            state = np.expand_dims(np.transpose(self.test_state, [1, 2, 0]),
                                   axis=0)
            q_values = self.network.gpu_inference(state)
            action = np.argmax(q_values)
        else:
            action = random.randrange(self.num_actions)

        self.test_state.pop(0)
        return [action, q_values]

    def save_model(self, epoch):
        self.network.save_model(epoch)
