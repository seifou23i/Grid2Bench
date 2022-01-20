import os
import numpy as np
from grid2op.Episode import EpisodeData


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

    def get_actions_freq_by_timestamp(self):
        """

        :return:
        """
        actions = self.get_actions()
        action_freq = []

        for action in actions:
            action_impact = action.impact_on_objects()

            if not action_impact["has_impact"]:
                action_freq.append(0)
            else:
                action_freq.append(
                    # number of injections
                    action_impact["injection"]["count"]
                    +
                    # number of lines forced
                    action_impact["force_line"]["reconnections"]["count"]
                    +
                    action_impact["force_line"]["disconnections"]["count"]
                    +
                    # number of lines switched
                    action_impact["switch_line"]["count"]
                    +
                    # number of topological changes
                    len(action_impact["topology"]["bus_switch"])
                    +
                    len(action_impact["topology"]["assigned_bus"])
                    +
                    len(action_impact["topology"]["disconnect_bus"])
                    +
                    # number of redispatch changes
                    len(action_impact["redispatch"]["generators"])
                    +
                    # number of storage changes
                    len(action_impact["storage"]["capacities"])
                    +
                    # number of curtailment changes
                    len(action_impact["curtailment"]["limit"])
                )

        return action_freq

    def get_actions_freq_by_type(self):
        """

        :return:
        """
        actions = self.get_actions()

        nb_injection = []
        nb_force_line = []
        nb_switch_line = []
        nb_topological_changes = []
        nb_redispatch_changes = []
        nb_storage_changes = []
        nb_curtailment_changes = []

        for action in actions:
            action_impact = action.impact_on_objects()

            nb_injection.append(action_impact["injection"]["count"])
            nb_force_line.append(
                action_impact["force_line"]["reconnections"]["count"]
                +
                action_impact["force_line"]["disconnections"]["count"]
            )
            nb_switch_line.append(action_impact["switch_line"]["count"])
            nb_topological_changes.append(
                len(action_impact["topology"]["bus_switch"])
                +
                len(action_impact["topology"]["assigned_bus"])
                +
                len(action_impact["topology"]["disconnect_bus"])
            )
            nb_redispatch_changes.append(len(action_impact["redispatch"]["generators"]))
            nb_storage_changes.append(len(action_impact["storage"]["capacities"]))
            nb_curtailment_changes.append(len(action_impact["curtailment"]["limit"]))
            # actions_freq = [0 if action.impact_on_objects()["has_impact"] else 1 for action in actions]

        return {
            'nb_injection': nb_injection,
            'nb_lines_forced': nb_force_line,
            'nb_lines_switched': nb_switch_line,
            'nb_topological_changes': nb_topological_changes,
            'nb_redispatching_changes': nb_redispatch_changes,
            'nb_storage_changes': nb_storage_changes,
            "nb_curtailment_changes": nb_curtailment_changes,
        }

    def get_actions_freq_by_station(self):
        """

        :return:
        """
        actions = self.get_actions()
        subs_impacted_by_timestamp = []

        for action, i in zip(actions, range(len(actions))):
            subs_impacted = action.get_topological_impact()[1]
            list_subs_impacted = action.name_sub[np.where(subs_impacted == True)]
            if len(list_subs_impacted) != 0:
                subs_impacted_by_timestamp.append({
                    "timestamp": i,
                    "subs_impacted": set(list_subs_impacted)

                }
                )

        return subs_impacted_by_timestamp

    def get_overflow_lines_by_timestamp(self):
        """

        :return:
        """
        observations = self.get_observations()
        overloaded_lines = []

        for observation, i in zip(observations, range(len(observations))):
            lines = np.where(observation.timestep_overflow != 0)[0]
            if len(lines) != 0:
                overloaded_lines.append(
                    {
                        "timestamp": i,
                        "lines_overflowed": set(lines)
                    }
                )

        return overloaded_lines

    def get_disconnected_lines_by_timestamp(self):
        """

        :return:
        """
        observations = self.get_observations()
        disconnected_lines = []

        for observation, i in zip(observations, range(len(observations))):
            lines_status = np.where(observation.line_status != True)[0]
            if len(lines_status) != 0:
                disconnected_lines.append(
                    {
                        "timestamp": i,
                        "lines_disconnected": set(lines_status)
                    }
                )

        return disconnected_lines
