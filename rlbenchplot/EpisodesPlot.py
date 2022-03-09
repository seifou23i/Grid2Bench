import os

from IPython.core.display_functions import display

from rlbenchplot.EpisodeDataExtractor import EpisodeDataExtractor
import pandas as pd
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from collections import Counter
import itertools
from datetime import timedelta
from matplotlib import pyplot as plt


import qgrid
import ipywidgets as widgets

class EpisodesPlot:
    """

    """
    def __init__(self, agent_path, episodes_names=[]):
        """

        """
        self.agent_path = agent_path
        self.episodes_names = episodes_names
        self.agent_name = os.path.basename(agent_path)

        self.episodes_data = self.load_all_episodes()
        self.n_lines =  self.episodes_data[0]._n_lines()
        self.n_episode = len(self.episodes_names)

    def load_all_episodes(self):
        """

        :return:
        """
        if not self.episodes_names :
            self.episodes_names = [ name for name in os.listdir(self.agent_path) if os.path.isdir(os.path.join(self.agent_path, name)) ]


        return [EpisodeDataExtractor(self.agent_path, episode_name) for episode_name in tqdm(self.episodes_names)]


    def plot_actions_freq_by_type(
            self,
            episodes_names=[],
            title="Frequency of actions by type",
            **fig_kwargs):
        """

        :return:
        """
        if not episodes_names: episodes_names = self.episodes_names

        df = pd.DataFrame()
        for episode_data in self.episodes_data:
            if episode_data.episode_name in episodes_names:
                df = pd.concat([df, episode_data.compute_actions_freq_by_type()], axis=0)

        df_sum = pd.DataFrame({'Frequency': df.sum(axis=0)})

        fig = px.pie(df_sum, values='Frequency', names=df_sum.index,
                     title=title)
        fig.update_traces(textposition='inside', textinfo='percent+label')

        fig.update_layout(**fig_kwargs)

        return fig

    def plot_actions_freq_by_station_pie_chart(
            self,
            episodes_names=[],
            title="Frequency of actions by station",
            **fig_kwargs):
        if not episodes_names: episodes_names = self.episodes_names

        actions_freq_by_station = []
        for episode_data in self.episodes_data:
            if episode_data.episode_name in episodes_names:
                actions_freq_by_station.extend(episode_data.compute_actions_freq_by_station())

        impacted_stations_flatten = []

        for item in actions_freq_by_station:
            impacted_stations_flatten.extend(list(item["subs_impacted"]))

        x = list(Counter(impacted_stations_flatten).keys())
        y = list(Counter(impacted_stations_flatten).values())

        fig = px.pie(names=x, values=y, title=title)
        fig.update_traces(textfont_size=12, textposition='inside', textinfo='percent+label')
        fig.update_layout(**fig_kwargs)

        return fig

    def plot_overloaded_disconnected_lines_freq(
            self,
            episodes_names=[],
            title="Frequency of overloaded and disconnected lines",
            **fig_kwargs):

        if not episodes_names: episodes_names = self.episodes_names

        overloaded_lines = []
        disconnected_lines = []

        for episode_data in self.episodes_data:
            if episode_data.episode_name in episodes_names:
                overloaded_lines_for_episode_i = episode_data.compute_overloaded_lines_by_timestamp()
                overloaded_lines_flatten = [list(item) for dict in overloaded_lines_for_episode_i for key, item in
                                            dict.items() if key == "Overloaded lines"]
                overloaded_lines = overloaded_lines + overloaded_lines_flatten

                disconnected_lines_for_episode_i = episode_data.compute_disconnected_lines_by_timestamp()
                disconnected_lines_flatten = [list(item) for dict in disconnected_lines_for_episode_i for key, item in
                                              dict.items() if key == "Disconnected lines"]
                disconnected_lines = disconnected_lines + disconnected_lines_flatten

        overloaded_lines_flatten = list(itertools.chain(*overloaded_lines))
        disconnected_lines_flatten = list(itertools.chain(*disconnected_lines))

        data = [[overloaded_lines_flatten.count(x), disconnected_lines_flatten.count(x)] for x in
                range(self.episodes_data[0]._n_lines())]
        df = pd.DataFrame(data, columns=["Overloaded", "Disconnected"])
        df = df.loc[~(df == 0).all(axis=1)]

        fig = px.bar(df, y=["Overloaded", "Disconnected"], x=self.episodes_data[0]._name_of_lines(df.index), log_y=True,
                     text_auto='.2s', labels={
                "x": "Line name",
                "value": "Frequency",
            },
                     title=title)
        fig.update_traces(textfont_size=12, textangle=0, cliponaxis=False)

        fig.update_layout(**fig_kwargs)

        return fig

    def plot_actions_sequence_length_by_type(
            self,
            episodes_names= [],
            title="Sequence length of actions by type",
            **fig_kwargs):

        if not episodes_names: episodes_names = self.episodes_names

        df = pd.DataFrame()
        for episode_data in self.episodes_data:
            if episode_data.episode_name in episodes_names:
                df = pd.concat([df, episode_data.compute_action_sequences_length()], axis=0)

        max = df.loc[:, df.columns != "Timestamp"].to_numpy().max()

        dict_list = []

        for j in range(1, df.shape[1]):
            i = df.shape[0] - 1
            while i >= 0:
                if df.iloc[i, j] != 0:
                    end = df.iloc[i, 0]
                    start = end - timedelta(minutes=5) * df.iloc[i, j]
                    dict_list.append(dict(Type=df.columns[j].removeprefix("NB "), Start=str(start), Finish=str(end),
                                          Actions=df.iloc[i, j],
                                          Actions_percent=(df.iloc[i, j] / max) * 100))
                    i = i - df.iloc[i, j]
                else:
                    i = i - 1
        fig = px.timeline(
            dict_list,
            x_start="Start",
            x_end="Finish",
            y="Type",
            color="Actions",
            labels={
                "Type": "Action Type",
                "Actions": "Action length"},
            title=title,
            color_continuous_scale=["green", "red"],
        )
        fig.update_layout(xaxis={"rangeslider": {"visible": True}})
        fig.update_layout(**fig_kwargs)


        return fig

    def plot_computation_times(
            self,
            episodes_names=[],
            title="Action Execution Time",
            **fig_kwargs):

        all_executions_time = pd.DataFrame(columns=["Timestamp", "Execution time"])

        if not episodes_names: episodes_names = self.episodes_names

        for episode_data in self.episodes_data:
            if episode_data.episode_name in episodes_names:
                df = pd.DataFrame(
                    np.column_stack((episode_data.timestamps, episode_data.computation_times[:episode_data.n_action])),
                    columns=all_executions_time.columns)
                all_executions_time = pd.concat([all_executions_time, df], ignore_index=True)

        all_executions_time =all_executions_time.sort_values(by='Timestamp', ascending=True)
        fig = px.line(all_executions_time, x=all_executions_time["Timestamp"], y="Execution time", title=title)
        fig.update_layout(xaxis={"rangeslider": {"visible": True}})

        fig.update_layout(**fig_kwargs)

        return fig

    def plot_distance_from_initial_topology(
            self,
            episodes_names=[],
            title="Distances from initial topology",
            **fig_kwargs
    ):


        if not episodes_names: episodes_names = self.episodes_names

        x = []
        y = []

        for episode_data in self.episodes_data:
            if episode_data.episode_name in episodes_names :
                for j in range(episode_data.n_action):
                    act = episode_data.actions[j]
                    obs = episode_data.observations[j]
                    # True == connected, False == disconnect
                    # So that len(line_statuses) - line_statuses.sum() is the distance for lines
                    line_statuses = episode_data.observations[j].line_status
                    # True == sub has something on bus 2, False == everything on bus 1
                    # So that subs_on_bus2.sum() is the distance for subs
                    subs_on_bus_2 = np.repeat(False, episode_data.observations[j].n_sub)
                    # objs_on_bus_2 will store the id of objects connected to bus 2
                    objs_on_bus_2 = {id: [] for id in range(episode_data.observations[j].n_sub)}
                    distance, _, _, _ = episode_data.get_distance_from_obs(act, line_statuses, subs_on_bus_2, objs_on_bus_2,
                                                                           obs)

                    x.append(episode_data.timestamps[j])
                    y.append(distance)

        df_distance = pd.DataFrame(data = np.array([x, y]).transpose(), columns = ['Timestamp', 'Distance'])
        df_distance = df_distance.sort_values(by='Timestamp', ascending=True)


        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_distance["Timestamp"].tolist(), y=df_distance["Distance"].tolist(),
                                 mode='lines+markers',line_shape='hvh'))

        fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16), xaxis_title="Timestamp",
                          yaxis_title="Distance", title=title)
        fig.update_layout(xaxis={"rangeslider": {"visible": True}})

        fig.update_layout(**fig_kwargs)

        return fig

    def plot_actions_freq_by_station(
            self,
            episodes_names=[],
            title="Frequency of actions by station",
            **fig_kwargs):
        if not episodes_names: episodes_names = self.episodes_names

        actions_freq_by_station = []
        for episode_data in self.episodes_data:
            if episode_data.episode_name in episodes_names:
                actions_freq_by_station.extend(episode_data.compute_actions_freq_by_station())

        impacted_stations_flatten = []

        for item in actions_freq_by_station:
            impacted_stations_flatten.extend(list(item["subs_impacted"]))

        x = list(Counter(impacted_stations_flatten).keys())
        y = list(Counter(impacted_stations_flatten).values())
        fig = px.bar(y=y, x=x, text_auto='.2s',
                     title=title,
                     labels={
                         "x": "Station",
                         "y": "Frequency"})
        fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)

        fig.update_layout(**fig_kwargs)
        return fig


    def plot_cumulative_reward(
            self,
            episodes_names=[],
            title="Cumulative reward",
            **fig_kwargs):
        """

        :param episodes_names:
        :param title:
        :param fig_kwargs:
        :return:
        """
        if not episodes_names: episodes_names = self.episodes_names

        chron_ids = list()
        cum_rewards = list()
        nb_time_steps = list()

        for episode_data in self.episodes_data:
            if episode_data.episode_name in episodes_names:
                chron_ids.append(episode_data.episode_name)
                cum_rewards.append(episode_data.cum_reward / 100)
                nb_time_steps.append(episode_data.nb_timestep_played)

        # visualizing the results
        x = np.arange(len(cum_rewards))
        fig = plt.figure(figsize=(14, 8))
        ax = plt.subplot(111)
        ax2 = ax.twinx()
        l1 = ax.plot(x, cum_rewards, marker='x', linewidth=2, markersize=12, label="Rewards", color="b")
        ax.set_xticks(x)
        ax.set_xticklabels(chron_ids, rotation=45, fontsize=15, ha="right")
        ax.set_ylabel("$ \\frac{Cumulative reward}{100}$", fontsize=20, color="b")
        ax.set_xlabel("Chronics", fontsize=14)
        l2 = ax2.plot(x, nb_time_steps, 'g--', marker='o', markersize=10, label="Time steps")
        ax2.set_ylabel("Accomplished time steps", color="g", fontsize=14)
        # added these three lines
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0, fontsize=16)
        ax.grid()
        ax2.grid()

    def plot_acted_actions(
            self,
            episodes_names=[],
            title="Acted vs not Acted actions",
            **fig_kwargs):
        """

        :param episodes_names:
        :param title:
        :param fig_kwargs:
        :return:
        """

        if not episodes_names: episodes_names = self.episodes_names

        not_acted_actions = 0
        acted_actions = 0
        df = pd.DataFrame()
        for episode_data in self.episodes_data:
            if episode_data.episode_name in episodes_names:
                acted_actions = acted_actions + len(episode_data.acted_actions())
                not_acted_actions = not_acted_actions + (episode_data.n_action-len(episode_data.acted_actions()))

        df_sum = pd.DataFrame({'Frequency': df.sum(axis=0)})

        fig = px.pie(values=[not_acted_actions, acted_actions, ], names=["Not Acted Actions", "Acted Actions"],
                     title=title)
        fig.update_traces(textposition='inside', textinfo='percent+label')

        fig.update_layout(**fig_kwargs)

        return fig







    def display_detailed_action_type(self, episodes_names=[]):

        data_display = display(display_id="data_display")
        output_display = display(display_id="output_display")
        grid = qgrid.QGridWidget(df=pd.DataFrame())

        w = widgets.Dropdown(
            options=['Select','Tolopology','Force_line', 'Redispatching', 'Injection', 'Curtailment', 'Storage'],
            value='Select',
            description='Table',
        )

        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':

                result = pd.DataFrame()
                c=0
                for episode_data in self.episodes_data:
                    if (not len(episodes_names)) or episode_data.episode_name in episodes_names:
                        functions = {
                            'Tolopology':episode_data.create_topology_df,
                            'Force_line': episode_data.create_force_line_df,
                            'Redispatching': episode_data.create_dispatch_df,
                            'Injection': episode_data.create_injection_df,
                            'Curtailment': episode_data.create_curtailment_df,
                            'Storage' : episode_data.create_storage_df
                        }
                        r= functions[change['new']]()
                        r[1]['episode_name'] = episode_data.episode_name
                        result = pd.concat([result, r[1]])
                        c+=r[0]
                output_display.update("total Number of "+change['new']+" changes:" + str(c))
                grid.df=result

        w.observe(on_change)
#         ouptup_display = display(display_id="ouptup_display")

        display(w)
        output_display.display('')
        data_display.display(grid)
        return



