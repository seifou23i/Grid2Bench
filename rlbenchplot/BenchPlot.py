from grid2op.Episode import EpisodeData
import os


class BenchPlot:
    """

    """

    def __init__(self, agent_path, episode_name):
        """

        """
        self.agent_path = os.path.abspath(agent_path)
        self.episode_name = os.path.join(self.agent_path, episode_name)
        self.episode_data = EpisodeData.from_disk(agent_path=self.agent_path, name=self.episode_name)

    def get_observations(self):
        """

        :return:
        """
        return [obs for obs in self.episode_data.observations]


    def get_actions(self):
        """

        :return:
        """
        return [action for action in self.episode_data.actions]


