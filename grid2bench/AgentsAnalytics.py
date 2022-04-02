# Copyright (C) 2022,
# IRT-SystemX (https://www.irt-systemx.fr), RTE (https://www.rte-france.com)
# See authors in pyporject.toml
# SPDX-License-Identifier: MPL-2.0
"""This module allows to benchmark several agents in several scenarios
through predefined KPIs (see KPI section).

The module uses the `plotly package <https://plotly.com/>`_ to visualize
these KPIs in the form of graphs.

The modules load data in the forms :class:`EpisodesDataTransformer` and
:class:`EpisodesDataExtractor` . You can access these objects like this:

  .. code-block:: python

    from grid2bench.AgentsAnalytics import AgentsAnalytics

    input_data_path = os.path.abspath('../data/input')
    agents_names = ['Expert_Agent', 'IEE_PPO_Agent']
    episodes_names = ['dec16_1', 'dec16_2']

    # loading data
    agents = AgentsAnalytics(
      data_path=input_data_path,
      agents_names=agents_names,
      episodes_names=episodes_names
    )

    # get indexes
    expert_agent_idx = agents.agents_names.index('Expert_Agent')
    dec16_2_idx = agents.episodes_names.index('dec16_2')

    # access expert agent all episodes results --> EpisodesDataTransformer class
    episodes = agents.agents_data[expert_agent_idx]
    print('Loaded episodes for {} are : {}'.format(
      episodes.agent_name,
      episodes.episodes_names))

    # access expert agent dec16_2 episode results --> EpisodeDataExtractor class
    dec16_2 = agents.agents_data[expert_agent_idx].episodes_data[dec16_2_idx]
    print('Episode name: {}'.format(dec16_2.episode_name))

"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Dict

from grid2bench.EpisodesDataTransformer import EpisodesDataTransformer


class AgentsAnalytics:
  """This class is used to benchmark several agents in several scenarios.

  It is also used to store agents' episode results via the agents_data attribute

  Attributes:

    - data_path: data folder path, parent directory path for agent log files
    - agents_names: list of agents names with the same name as the agent's
      data folders
    - episodes_names: list of episode names with the same name as the episode
      data folders
    - agents_data: list of :class:`EpisodesDataTransformer`

  """

  def __init__(self, data_path: str, agents_names: Optional[List] = None,
               episodes_names: Optional[List] = None):
    """Init and loading data for the class.

    :param data_path: parent directory path for agent log files
    :param agents_names: a list of each agent repository name,
                         if empty (not recommended) it will load all
                         repositories in the data path
    :param episodes_names: list of episode names with the same name as the
                           episode data folder
    """
    self.data_path = data_path
    self.agents_names = agents_names
    self.episodes_names = episodes_names

    self.agents_data = self.load_agents_results()

  def load_agents_results(self) -> List[EpisodesDataTransformer]:
    """Load agents' episodes data from the episodes_names list.

    Each element of the list contains agent's episodes data as
    :class:`EpisodesDataTransformer`

    :return: list of agents' episodes
    :rtype: list of :class:`EpisodesDataTransformer`
    """
    if not self.agents_names:
      self.agents_names = [name for name in os.listdir(self.data_path) if
                           os.path.isdir(os.path.join(self.data_path, name))]

    if not self.episodes_names:
      self.episodes_names = [name for name in os.listdir(
        os.path.join(self.data_path, self.agents_names[0])) if os.path.isdir(
        os.path.join(self.data_path, self.agents_names[0], name))]

    return [EpisodesDataTransformer(
      agent_path=os.path.join(self.data_path, agent_name),
      episodes_names=self.episodes_names) for agent_name in self.agents_names]

  @staticmethod
  def plot_actions_freq_by_station(
      agents_results: Optional[List] = None,
      episodes_names: Optional[List] = None,
      title: Optional[str] = 'Frequency of actions by station',
      **fig_kwargs):
    """Plot frequency of actions by station as a barchart for several agents
    and selected episodes_names

    :param agents_results: list of agent objects of class 'Agents_Evaluation ' or class 'Episode_Plot'
    :param episodes_names: filter some episodes, if empty it will show all loaded episodes
    :param title: plot title, if empty it will return default value
    :param fig_kwargs: keyword for plotly arguments, example: height= 700
    :return:
    """
    agent_names = []

    # for the first agent
    agent_names.append(agents_results[0].agent_name)
    df = agents_results[0].actions_freq_by_station_several_episodes(
      episodes_names)
    df = df.rename(columns={'Frequency': agents_results[0].agent_name})

    for agent in agents_results[1:]:
      agent_names.append(agent.agent_name)

      df2 = agent.actions_freq_by_station_several_episodes(episodes_names)
      df2 = df2.rename(columns={'Frequency': agent.agent_name})

      df = df.join(df2.set_index('Substation'), on='Substation')

    newnames = {}
    y_list = []
    for i in range(len(agent_names)):
      newnames['wide_variable_{}'.format(i)] = agent_names[i]
      y_list.append(df[agent_names[i]].to_list())

    fig = px.bar(x=df['Substation'].to_list(), y=y_list, text_auto='.2s',
                 labels={'x': 'Station', 'value': 'Frequency'}, barmode='group',
                 title=title)

    fig.for_each_trace(
      lambda t: t.update(name=newnames[t.name], legendgroup=newnames[t.name],
                         hovertemplate=t.hovertemplate.replace(t.name, newnames[
                           t.name])))

    fig.update_traces(textfont_size=12, textangle=0, textposition='outside',
                      cliponaxis=False)

    fig.update_layout(**fig_kwargs)

    return fig

  @staticmethod
  def plot_actions_freq_by_type(agents_results=[], episodes_names=[],
                                title='Frequency of actions by type', row=1,
                                col=2, **fig_kwargs):
    """

    :param agents_results: list of agent objects of class 'Agents_Evaluation ' or class 'Episode_Plot'
    :param episodes_names: filter some episodes, if empty it will show all loaded episodes
    :param title: plot title, if empty it will return default value
    :param fig_kwargs: keyword for plotly arguments, example: height= 700
    :param row: number of rows in plotly subplot
    :param col: number of cols in plotly subplot, need to be customized based on number of agents
    :return:
    """
    agent_names = []
    for agent in agents_results:
      agent_names.append(agent.agent_name)

    list_i = []
    for i in range(row):
      list_j = []
      for j in range(col):
        list_j.append({'type': 'domain'})
      list_i.append(list_j)

    fig = make_subplots(row, col, specs=list_i, subplot_titles=agent_names)

    for i in range(row):
      for j in range(col):
        data = agents_results[i + j].actions_freq_by_type_several_episodes(
          episodes_names)
        fig.add_trace(go.Pie(labels=data.index, values=data['Frequency'],
                             name=agent_names[i + j]), i + 1, j + 1)

    fig.update_traces(textposition='inside')
    fig.update_layout(title_text=title, uniformtext_minsize=12,
                      uniformtext_mode='hide', **fig_kwargs)
    return fig

  @staticmethod
  def plot_actions_freq_by_station_pie_chart(agents_results=[],
                                             episodes_names=[],
                                             title='Frequency of actions by station',
                                             row=1, col=2, **fig_kwargs):
    """
    :param agents_results: list of agent objects of class 'Agents_Evaluation ' or class 'Episode_Plot'
    :param episodes_names: filter some episodes, if empty it will show all loaded episodes
    :param title: plot title, if empty it will return default value
    :param row: number of rows in plotly subplot
    :param col: number of cols in plotly subplot, need to be customized based on number of agents
    :param fig_kwargs: keyword for plotly arguments, example: height= 700
    :return:
    """
    agent_names = []
    for agent in agents_results:
      agent_names.append(agent.agent_name)

    list_i = []
    for i in range(row):
      list_j = []
      for j in range(col):
        list_j.append({'type': 'domain'})
      list_i.append(list_j)

    fig = make_subplots(row, col, specs=list_i, subplot_titles=agent_names)

    for i in range(row):
      for j in range(col):
        data = agents_results[i + j].actions_freq_by_station_several_episodes(
          episodes_names)
        fig.add_trace(
          go.Pie(labels=data['Substation'], values=data['Frequency'],
                 name=agent_names[i + j]), i + 1, j + 1)

    fig.update_traces(textposition='inside')
    fig.update_layout(title_text=title, uniformtext_minsize=12,
                      uniformtext_mode='hide', **fig_kwargs)
    return fig

  @staticmethod
  def plot_lines_impact(agents_results=[], episodes_names=[],
                        title='Overloaded Lines by station',
                        fig_type='overloaded', **fig_kwargs):
    """Visualize the impact of actions on power system lines

    :param agents_results: list of agent objects of class 'Agents_Evaluation ' or class 'Episode_Plot', defaults to []
    :type agents_results: list, optional
    :param episodes_names: filter some episodes, if empty it will show all loaded episodes, defaults to []
    :type episodes_names: list, optional
    :param title: plot title, if empty it will return default value, defaults to 'Overloaded Lines by station'
    :type title: str, optional
    :param fig_type: disconnected, if True plots disconnected lines, else draws overflowed lines, defaults to 'overloaded'
    :type fig_type: str, optional
    :param fig_kwargs: keyword for plotly arguments, example: height= 700, defaults to {}
    :type fig_kwargs: dict, optional
    :return: fig object
    :rtype: fig, px.bar
    """
    agent_names = []

    # for the first agent
    agent_names.append(agents_results[0].agent_name)

    if fig_type == 'overloaded':
      df = agents_results[0].overloaded_lines_freq_several_episodes(
        episodes_names)
      df = df.rename(columns={'Overloaded': agents_results[0].agent_name})
    else:
      df = agents_results[0].disconnected_lines_freq_several_episodes(
        episodes_names)
      df = df.rename(columns={'Disconnected': agents_results[0].agent_name})
      title = 'Disconnected Lines by station'

    for agent in agents_results[1:]:
      agent_names.append(agent.agent_name)
      if fig_type == 'overloaded':
        df2 = agent.overloaded_lines_freq_several_episodes(episodes_names)
        df2 = df2.rename(columns={'Overloaded': agent.agent_name})
      else:
        df2 = agent.disconnected_lines_freq_several_episodes(episodes_names)
        df2 = df2.rename(columns={'Disconnected': agent.agent_name})

      df = df.join(df2.set_index('Line'), on='Line')

    newnames = {}
    y_list = []
    for i in range(len(agent_names)):
      newnames['wide_variable_{}'.format(i)] = agent_names[i]
      y_list.append(df[agent_names[i]].to_list())

    fig = px.bar(df, x='Line', y=y_list, text_auto='.2s',
                 labels={'x': 'Line', 'value': 'Frequency'}, barmode='group',
                 title=title, log_y=True, )

    fig.for_each_trace(
      lambda t: t.update(name=newnames[t.name], legendgroup=newnames[t.name],
                         hovertemplate=t.hovertemplate.replace(t.name, newnames[
                           t.name])))

    fig.update_traces(textfont_size=12, textangle=0, textposition='outside',
                      cliponaxis=False)

    fig.update_layout(**fig_kwargs)

    return fig

  @staticmethod
  def plot_computation_times(agents_results=[], episodes_names=[],
                             title='Action Execution Time', **fig_kwargs):
    """

    :param agents_results: list of agent objects of class 'Agents_Evaluation ' or class 'Episode_Plot'
    :param episodes_names: filter some episodes, if empty it will show all loaded episodes
    :param title: plot title, if empty it will return default value
    :param fig_kwargs: keyword for plotly arguments, example: height= 700
    :return:
    """
    agent_names = []

    # for the first agent
    agent_names.append(agents_results[0].agent_name)
    df = agents_results[0].computation_times_several_episodes(episodes_names)
    df = df.rename(columns={'Execution time': agents_results[0].agent_name})

    for agent in agents_results[1:]:
      agent_names.append(agent.agent_name)
      df2 = agent.computation_times_several_episodes(episodes_names)
      df2 = df2.rename(columns={'Execution time': agent.agent_name})
      df = df.join(df2.set_index('Timestamp'), on='Timestamp')

    # Create traces
    fig = go.Figure()

    for agent_name in agent_names:
      fig.add_trace(
        go.Scatter(x=df['Timestamp'].tolist(), y=df[agent_name].tolist(),
                   mode='lines+markers', name=agent_name))

    fig.update_layout(xaxis={'rangeslider': {'visible': True}}, title=title,
                      xaxis_title='Timestamp', yaxis_title='Execution Time (s)')
    fig.update_layout(**fig_kwargs)

    return fig

  @staticmethod
  def plot_distance_from_initial_topology(agents_results=[], episodes_names=[],
                                          title='Distance from initial topology',
                                          **fig_kwargs):
    """
    :param agents_results: list of agent objects of class 'Agents_Evaluation ' or class 'Episode_Plot'
    :param episodes_names: filter some episodes, if empty it will show all loaded episodes
    :param title: plot title, if empty it will return default value
    :param fig_kwargs: keyword for plotly arguments, example: height= 700
    :return:
    """
    # TODO : creating a function to reuse this

    # for the first agent
    agent_names = [agents_results[0].agent_name]
    df = agents_results[0].distance_from_initial_topology(episodes_names)
    df = df.rename(columns={'Distance': agents_results[0].agent_name})

    for agent in agents_results[1:]:
      agent_names.append(agent.agent_name)
      df2 = agent.distance_from_initial_topology(episodes_names)
      df2 = df2.rename(columns={'Distance': agent.agent_name})

      df = df.join(df2.set_index('Timestamp'), on='Timestamp')

    # Create traces
    fig = go.Figure()

    for agent_name in agent_names:
      fig.add_trace(
        go.Scatter(x=df['Timestamp'].tolist(), y=df[agent_name].tolist(),
                   mode='lines+markers', line_shape='hvh', name=agent_name))

    fig.update_layout(xaxis={'rangeslider': {'visible': True}}, title=title,
                      xaxis_title='Timestamp', yaxis_title='Distance')
    fig.update_layout(**fig_kwargs)

    return fig

  @staticmethod
  def plot_actions_sequence_length(agents_data, episodes_names=[],
                                   title='Sequence length of actions',
                                   min_length: int = 0, max_length: int = 400,
                                   **fig_kwargs):

    plot_data = []
    for agent in agents_data:
      plot_data.extend(
        agent.action_sequences_to_dict(episodes_names, min_length, max_length))

    fig = px.timeline(plot_data, x_start='Start', x_end='Finish', y='Type',
                      color='Actions',
                      labels={'Type': 'Agent ', 'Actions': 'Action Sequence'},
                      title=title, color_continuous_scale=['green', 'red'], )
    fig.update_layout(xaxis={'rangeslider': {'visible': True}})
    fig.update_layout(**fig_kwargs)

    return fig

  @staticmethod
  # cumulative reward extraction
  def cumulative_reward(agent_data, episodes_names):

    episode_names = list()
    cum_rewards = list()
    nb_time_steps = list()

    for episode in agent_data.episodes_data:
      if episode.episode_name in episodes_names:
        episode_names.append(episode.episode_name)
        cum_rewards.append(episode.cum_reward / 100)
        nb_time_steps.append(episode.nb_timestep_played)

    df = pd.DataFrame(
      data=np.array([episode_names, nb_time_steps, cum_rewards]).transpose(),
      columns=['Episode', 'Played timesteps', 'Cumulative reward'], )

    return df.astype(
      {'Episode': 'str', 'Played timesteps': int, 'Cumulative reward': float})

  @staticmethod
  def plot_cumulative_reward(agents_data=[], episodes_names=[],
                             fig_type='CumReward',
                             title='Cumulative reward per episode',
                             **fig_kwargs):
    if not episodes_names: episodes_names = agents_data[0].episodes_names

    new_names = {}
    y_list = []

    for i, agent in enumerate(agents_data):
      if fig_type == 'CumReward':
        df = AgentsAnalytics.cumulative_reward(agent, episodes_names)
        y = df['Cumulative reward'].tolist()
      else:
        df = AgentsAnalytics.cumulative_reward(agent, episodes_names)
        y = df['Played timesteps'].tolist()

      agent_episodes = df['Episode'].tolist()
      new_names['wide_variable_{}'.format(i)] = agent.agent_name
      y_list.append(y)

    if fig_type == 'CumReward':
      x_title = '$\\frac{Cumulative reward}{100}$'
    else:
      title = x_title = 'Accomplished time steps'

    fig = px.bar(x=agent_episodes, y=y_list, text_auto='.2s',
                 labels={'x': 'Scenario', 'value': x_title}, barmode='group',
                 title=title)

    fig.for_each_trace(
      lambda t: t.update(name=new_names[t.name], legendgroup=new_names[t.name],
                         hovertemplate=t.hovertemplate.replace(t.name,
                                                               new_names[
                                                                 t.name])))

    # Set y-axes titles
    fig.update_layout(**fig_kwargs)

    return fig
