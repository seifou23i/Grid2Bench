import os
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from rlbenchplot.AgentsAnalytics import AgentsAnalytics
from configmanager.configmanager import ConfigManager

conf_path = os.path.abspath("../conf.ini")
conf = ConfigManager(benchmark_name='Tests', path=conf_path)

data_path = os.path.abspath(os.path.join('..', conf.get_option('data_path')))
agents_names = conf.get_option_tolist('agents_names')
episodes_names = conf.get_option_tolist('episodes_names')


class TestAgentsAnalytics(unittest.TestCase):

  def setUp(self):
    self.data_path = data_path
    self.agents_names = agents_names
    self.episodes_names = episodes_names
    self.agents = AgentsAnalytics(data_path=self.data_path, agents_names=self.agents_names,
      episodes_names=self.episodes_names)

  def test_plot_actions_freq_by_station(self):
    AgentsAnalytics.plot_actions_freq_by_station_pie_chart(agents_results=self.agents.agents_data)

  def test_plot_actions_freq_by_type(self):
    AgentsAnalytics.plot_actions_freq_by_type(agents_results=self.agents.agents_data)

  def test_plot_actions_freq_by_station_pie_chart(self):
    AgentsAnalytics.plot_actions_freq_by_station_pie_chart(agents_results=self.agents.agents_data)

  def test_plot_lines_impact(self):
    AgentsAnalytics.plot_lines_impact(agents_results=self.agents.agents_data, fig_type='overloaded')

  def test_plot_lines_impact(self):
    AgentsAnalytics.plot_lines_impact(agents_results=self.agents.agents_data, fig_type='disconnected')

  def test_plot_computation_times(self):
    AgentsAnalytics.plot_computation_times(agents_results=self.agents.agents_data)

  def test_plot_distance_from_initial_topology(self):
    AgentsAnalytics.plot_distance_from_initial_topology(agents_results=self.agents.agents_data)

  def test_plot_actions_sequence_length(self):
    AgentsAnalytics.plot_actions_sequence_length(agents_data=self.agents.agents_data, sequence_range=range(0, 20))

  def test_plot_cumulative_reward(self):
    AgentsAnalytics.plot_cumulative_reward(agents_data=self.agents.agents_data)

  def test_cumulative_reward(self):
    df1 = AgentsAnalytics.cumulative_reward(agent_data=self.agents.agents_data[0], episodes_names=self.episodes_names)
    df2 = pd.DataFrame(
      {'Episode': ['0', '1'], 'Played timesteps': [10, 10], 'Cumulative reward': [109.480518, 110.966211]})
    assert_frame_equal(df1, df2, check_dtype=False)
