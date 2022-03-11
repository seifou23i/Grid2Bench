import unittest
from rlbenchplot.EpisodeDataExtractor import EpisodeDataExtractor
import os
import datetime

# to put in confi.ini file later
agent_path = os.path.abspath("data/agents/rte_case14_redisp_random_agent")
episode_name = "0"


class TestEpisodeDataExtractor(unittest.TestCase):

    def setUp(self):
        self.agent_path = agent_path
        self.agent_name = "rte_case14_redisp_random_agent"
        self.episode_name = "0"

        self.episode_data = EpisodeDataExtractor(self.agent_path, self.episode_name)


    def test_get_observation_by_timestamp(self):
        observation = self.episode_data.get_observation_by_timestamp(
            datetime.datetime(year=2019, month=1, day=6, hour=0, minute=20))

        self.assertEqual(observation, self.episode_data.observations[-2])

    def test_get_action_by_timestamp(self):
        action = self.episode_data.get_action_by_timestamp(
            datetime.datetime(year=2019, month=1, day=6, hour=0, minute=15))
        self.assertEqual(action, self.episode_data.actions[3])

    def test_get_computation_time_by_timestamp(self):
        computation_time = self.episode_data.get_computation_time_by_timestamp(
            datetime.datetime(year=2019, month=1, day=6, hour=0, minute=20))
        self.assertEqual(computation_time, self.episode_data.computation_times[4])

    def test_get_timestep_by_datetime(self):
        timestep = self.episode_data.get_timestep_by_datetime(
            datetime.datetime(year=2019, month=1, day=6, hour=0, minute=0))
        self.assertEqual(timestep, 0)

    def test_compute_actions_freq_by_timestamp(self):
        list_actions_freq = self.episode_data.compute_actions_freq_by_timestamp()["NB action"]
        self.assertListEqual(list_actions_freq.to_list(), [1, 4, 1, 1])

    def test_compute_actions_freq_by_type(self):
        nb_switch_line = self.episode_data.compute_actions_freq_by_type()["NB line switched"]
        nb_topological_changes = self.episode_data.compute_actions_freq_by_type()["NB topological change"]
        nb_redispatch_changes = self.episode_data.compute_actions_freq_by_type()["NB redispatching"]
        nb_storage_changes = self.episode_data.compute_actions_freq_by_type()["NB storage changes"]
        nb_curtailment_changes = self.episode_data.compute_actions_freq_by_type()["NB curtailment"]

        self.assertListEqual(nb_switch_line.to_list(), [0, 0, 0, 1])
        self.assertListEqual(nb_topological_changes.to_list(), [0, 4, 1, 0])
        self.assertListEqual(nb_redispatch_changes.to_list(), [1, 0, 0, 0])
        self.assertListEqual(nb_storage_changes.to_list(), [0, 0, 0, 0])
        self.assertListEqual(nb_curtailment_changes.to_list(), [0, 0, 0, 0])

    def test_compute_actions_freq_by_station(self):
        list_impacted_substations = self.episode_data.compute_actions_freq_by_station()
        impacted_sub_stations = [list_impacted_substations[i]["subs_impacted"] for i in
                                 range(len(list_impacted_substations))]
        self.assertListEqual(impacted_sub_stations, [{'sub_13'}, {'sub_12'}])

    def test_compute_overloaded_lines_by_timestamp(self):
        list_overloaded_lines = self.episode_data.compute_overloaded_lines_by_timestamp()
        overloaded_lines = [list_overloaded_lines[i]["Overloaded lines"] for i in range(len(list_overloaded_lines))]
        self.assertListEqual(overloaded_lines, [{5, 14}, {5, 14}, {8, 16}])

    def test_compute_disconnected_lines_by_timestamp(self):
        list_disconnected_lines = self.episode_data.compute_disconnected_lines_by_timestamp()
        disconnected_lines = [list_disconnected_lines[i]["Disconnected lines"] for i in
                              range(len(list_disconnected_lines))]
        self.assertListEqual(disconnected_lines, [{5, 14, 15}])

    def test_compute_action_sequences_length(self):
        actions_sequence_length = self.episode_data.compute_action_sequences_length()["Sequence length"]

        self.assertListEqual(actions_sequence_length.to_list(), [1, 2, 3, 4])

