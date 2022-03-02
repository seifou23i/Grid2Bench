import os
from rlbenchplot.EpisodeDataExtractor import EpisodeDataExtractor
import pandas as pd
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from collections import Counter
import itertools
from datetime import timedelta

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
        for episode_data in tqdm(self.episodes_data):
            if episode_data.episode_name in episodes_names:
                df = pd.concat([df, episode_data.compute_actions_freq_by_type()], axis=0)

        df_sum = pd.DataFrame({'Frequency': df.sum(axis=0)})

        fig = px.pie(df_sum, values='Frequency', names=df_sum.index,
                     title=title)
        fig.update_traces(textposition='inside', textinfo='percent+label')

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

        for episode_data in tqdm(self.episodes_data):
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

        fig = px.bar(df, y=["Overloaded", "Disconnected"], x=self.episodes_data[0]._name_of_lines(df.index),
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

        # Initialize the dict for design purposes
        dict_list = [
            dict(Type=df.columns[j].removeprefix("NB "), Start=str(df.iloc[0, 0]), Finish=str(df.iloc[0, 0]), Actions=0,
                 Actions_percent=0) for j in range(1, len(df.columns)) if df.columns[j] != "Timestamp"]

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

        fig = px.line(all_executions_time, x=np.arange(all_executions_time.shape[0]), y="Execution time", title=title)
        fig.update_layout(xaxis={"rangeslider": {"visible": True}})

        fig.update_layout(**fig_kwargs)

        return fig

    def plot_distance_from_initial_topology(
            self,
            episodes_names=[],
            title="Distances from initial topologys",
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

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(len(x)), y=y, name="linear",
                                 line_shape='hvh'))

        fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16), xaxis_title="Timestamp",
                          yaxis_title="Distance", title=title)
        fig.update_layout(xaxis={"rangeslider": {"visible": True}})

        fig.update_layout(**fig_kwargs)

        return fig

    def plot_actions_freq_by_station(
            self,
            episodes_names=[],
            title="Frequency of overloaded and disconnected lines",
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



