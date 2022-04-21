import datetime
import os
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from grid2bench.EpisodesDataTransformer import EpisodesDataTransformer
from configmanager.configmanager import ConfigManager

conf_path = os.path.abspath("../conf.ini")
conf = ConfigManager(benchmark_name='Tests', path=conf_path)

data_path = os.path.abspath(os.path.join('..', conf.get_option('data_path')))
agent_name = conf.get_option_tolist('agents_names')[1]

agent_path = os.path.join(data_path, agent_name)
episodes_names = conf.get_option_tolist('episodes_names')


class TestEpisodesDataTransformer(unittest.TestCase):

  def setUp(self):
    self.agent_path = agent_path
    self.agent_name = agent_name
    self.episodes_names = episodes_names

    self.agent = EpisodesDataTransformer(agent_path=self.agent_path,
                                         episodes_names=self.episodes_names)

  def test_concat_episodes_actions_freq_by_type(self):
    df1 = self.agent.actions_freq_by_type_several_episodes()
    df2 = pd.DataFrame({'Frequency': [1, 24, 1, 0, 0]},
                       index=['NB line switched', 'NB topological change',
                              'NB redispatching', 'NB storage changes',
                              'NB curtailment'])
    assert_frame_equal(df1, df2, check_dtype=False)

  def test_concat_episodes_actions_freq_by_station(self):
    df1 = self.agent.actions_freq_by_station_several_episodes()
    df2 = pd.DataFrame({'Substation': ['sub_13', 'sub_12', 'sub_1', 'sub_11'],
                        'Frequency': [2, 1, 2, 1]})
    assert_frame_equal(df1, df2, check_dtype=False)

  def test_concat_computation_times(self):
    df1 = self.agent.computation_times_several_episodes(
      episodes_names=self.episodes_names[0])
    df2 = pd.DataFrame({'Timestamp': [
      datetime.datetime.strptime('2019-01-06 00:00:00', '%Y-%m-%d %H:%M:%S'),
      datetime.datetime.strptime('2019-01-06 00:05:00', '%Y-%m-%d %H:%M:%S'),
      datetime.datetime.strptime('2019-01-06 00:10:00', '%Y-%m-%d %H:%M:%S')],
                        'Execution time': [2.1457 * pow(10, -5),
                                           2.1934 * pow(10, -5),
                                           4.4345 * pow(10, -5)]})
    assert_frame_equal(df1.head(3), df2, check_dtype=False)

  def test_concat_distance_from_initial_topology(self):
    df1 = self.agent.distance_from_initial_topology(
      episodes_names=self.episodes_names[0])
    df2 = pd.DataFrame({'Timestamp': [
      datetime.datetime.strptime('2019-01-06 00:00:00', '%Y-%m-%d %H:%M:%S'),
      datetime.datetime.strptime('2019-01-06 00:05:00', '%Y-%m-%d %H:%M:%S'),
      datetime.datetime.strptime('2019-01-06 00:10:00', '%Y-%m-%d %H:%M:%S')],
                        'Distance': [0, 1, 1]})
    assert_frame_equal(df1.head(3), df2, check_dtype=False)

  def test_display_sequence_actions_filter(self):
    self.agent.display_sequence_actions_filter()

  def test_concat_disconnected_lines_freq(self):
    df1 = self.agent.disconnected_lines_freq_several_episodes()
    df2 = pd.DataFrame(
      {'Line': ['11_12_5', '5_12_14', '3_6_15'], 'Disconnected': [2, 1, 1]},
      index=[5, 14, 15])
    assert_frame_equal(df1, df2, check_dtype=False)

  def test_concat_overloaded_lines_freq(self):
    df1 = self.agent.overloaded_lines_freq_several_episodes()
    df2 = pd.DataFrame(
      {'Line': ['11_12_5', '1_3_8', '1_4_9', '5_12_14', '3_8_16', '4_5_17'],
       'Overloaded': [2, 1, 2, 2, 1, 1]}, index=[5, 8, 9, 14, 16, 17])
    assert_frame_equal(df1, df2, check_dtype=False)
