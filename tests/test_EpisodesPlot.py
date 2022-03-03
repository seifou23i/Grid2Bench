import unittest
import os
from rlbenchplot.EpisodesPlot import EpisodesPlot

# to put in confi.ini file later
agent_path = os.path.abspath("data/agents/rte_case14_redisp_random_agent")
episodes_names = ['0', '1']


class TestEpisodesPlot(unittest.TestCase):

    def setUp(self):
        self.agent_path = agent_path
        self.agent_name = "rte_case14_redisp_random_agent"
        self.episodes_names = "0"

        self.episodes_plot = EpisodesPlot(self.agent_path, self.episodes_names)

    def test_plot_actions_freq_by_type(self):
        self.episodes_plot.plot_actions_freq_by_type()

    def test_plot_overloaded_disconnected_lines_freq(self):
        self.episodes_plot.plot_overloaded_disconnected_lines_freq()

    def test_plot_actions_sequence_length_by_type(self):
        self.episodes_plot.plot_computation_times()

    def test_plot_distance_from_initial_topology(self):
        self.episodes_plot.plot_distance_from_initial_topology()

    def test_plot_actions_freq_by_station(self):
        self.episodes_plot.plot_actions_freq_by_station()
