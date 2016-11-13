import numpy as np
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
import seaborn as sns

class Visuals:

	def __init__(self, actions):

		all_action_names = ['no-op', 'fire', 'up', 'right', 'left', 'down', 'up_right', 'up_left', 'down-right', 'down-left', 
			'up-fire', 'right-fire', 'left-fire', 'down-fire', 'up-right-fire', 'up-left-fire', 'down-right-fire', 'down-left-fire']

		action_names = [all_action_names[i] for i in actions]
		self.num_actions = len(actions)
		self.max_q = 1
		self.min_q = 0
		# self.max_avg_q = 1

		xlocations = np.linspace(0.5, self.num_actions - 0.5, num=self.num_actions)
		xlocations = np.append(xlocations, self.num_actions + 0.05)
		if self.num_actions > 7:
			self.fig = plt.figure(figsize=(self.num_actions * 1.1, 6.0))
		else:
			self.fig = plt.figure()
		self.bars = plt.bar(np.arange(self.num_actions), np.zeros(self.num_actions), 0.9)
		plt.xticks(xlocations, action_names + [''])
		plt.ylabel('Expected Future Reward')
		plt.xlabel('Action')
		plt.title("State-Action Values")
		color_palette = sns.color_palette(n_colors=self.num_actions)
		for bar, color in zip(self.bars, color_palette):
			bar.set_color(color)
		self.fig.show()


	def update(self, q_values):

		for bar, q_value in zip(self.bars, q_values):
			bar.set_height(q_value)
		step_max = np.amax(q_values)
		step_min = np.amin(q_values)
		if step_max > self.max_q:
			self.max_q = step_max
			plt.gca().set_ylim([self.min_q, self.max_q])
		if step_min < self.min_q:
			self.min_q = step_min
			plt.gca().set_ylim([self.min_q, self.max_q])

		self.fig.canvas.draw()