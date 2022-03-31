# Copyright (C) 2022,
# IRT-SystemX (https://www.irt-systemx.fr), RTE (https://www.rte-france.com)
# See authors in pyporject.toml
# SPDX-License-Identifier: MPL-2.0
"""

"""
import itertools
import os
from collections import Counter
from datetime import datetime

import ipywidgets as widgets
import numpy as np
import pandas as pd
import qgrid
from IPython.display import display
from tqdm import tqdm

from grid2bench.EpisodeDataExtractor import EpisodeDataExtractor


class EpisodesDataTransformer:
  """

  """

  def __init__(self, agent_path: str, episodes_names: list):
    """

    """
    self.agent_path = agent_path
    self.episodes_names = episodes_names
    self.agent_name = os.path.basename(agent_path)

    self.episodes_data = self.load_all_episodes()
    self.n_lines = self.episodes_data[0]._n_lines()
    self.n_episode = len(self.episodes_names)

  def load_all_episodes(self):
    """

    :return:
    """
    if not self.episodes_names:
      self.episodes_names = [name for name in os.listdir(self.agent_path) if
                             os.path.isdir(os.path.join(self.agent_path, name))]

    return [EpisodeDataExtractor(self.agent_path, episode_name) for episode_name
            in tqdm(self.episodes_names)]

  def get_action_by_timestamp(self, timestamp_str: str):
    """
    :param timestamp_str:
    :return:
    """
    for episode in self.episodes_data:
      timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
      if timestamp in episode.timestamps:
        return episode.get_action_by_timestamp(timestamp)

  def get_observation_by_timestamp(self, timestamp_str: str):
    """
    :param timestamp_str:
    :return:
    """
    for episode in self.episodes_data:
      timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
      if timestamp in episode.timestamps:
        return episode.get_observation_by_timestamp(timestamp)

  def concat_episodes_actions_freq_by_type(self, episodes_names: list = None, ):
    """

    :return:
    """
    if not episodes_names: episodes_names = self.episodes_names

    df = pd.DataFrame()
    for episode_data in self.episodes_data:
      if episode_data.episode_name in episodes_names:
        df = pd.concat([df, episode_data.actions_freq_by_type()], axis=0)

    df_sum = pd.DataFrame({'Frequency': df.sum(axis=0)})

    return df_sum

  def concat_episodes_actions_freq_by_station(self,
                                              episodes_names: list = None, ):
    if not episodes_names: episodes_names = self.episodes_names

    actions_freq_by_station = []
    for episode_data in self.episodes_data:
      if episode_data.episode_name in episodes_names:
        actions_freq_by_station.extend(
          episode_data.impacted_subs_by_timestamp())

    impacted_stations_flatten = []

    for item in actions_freq_by_station:
      impacted_stations_flatten.extend(list(item['subs_impacted']))

    x = list(Counter(impacted_stations_flatten).keys())
    y = list(Counter(impacted_stations_flatten).values())
    df = pd.DataFrame(data=np.array([x, y]).transpose(),
                      columns=['Substation', 'Frequency'])
    df = df.astype({'Frequency': int}, errors='raise')

    return df

  def concat_computation_times(self, episodes_names=[]):

    all_executions_time = pd.DataFrame(columns=['Timestamp', 'Execution time'])

    if not episodes_names: episodes_names = self.episodes_names

    for episode_data in self.episodes_data:
      if episode_data.episode_name in episodes_names:
        df = pd.DataFrame(
          np.column_stack((episode_data.timestamps,
                           episode_data.computation_times[
                           :episode_data.n_action])),
          columns=all_executions_time.columns)
        all_executions_time = pd.concat([all_executions_time, df],
                                        ignore_index=True)

    all_executions_time = all_executions_time.sort_values(by='Timestamp',
                                                          ascending=True)

    return all_executions_time

  def concat_distance_from_initial_topology(self, episodes_names=[],

                                            ):

    if not episodes_names: episodes_names = self.episodes_names

    x = []
    y = []

    for episode_data in self.episodes_data:
      if episode_data.episode_name in episodes_names:
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
          objs_on_bus_2 = {id: [] for id in
                           range(episode_data.observations[j].n_sub)}
          distance, _, _, _ = episode_data.get_distance_from_obs(act,
                                                                 line_statuses,
                                                                 subs_on_bus_2,
                                                                 objs_on_bus_2,
                                                                 obs)

          x.append(episode_data.timestamps[j])
          y.append(distance)

    df_distance = pd.DataFrame(data=np.array([x, y]).transpose(),
                               columns=['Timestamp', 'Distance'])
    df_distance = df_distance.sort_values(by='Timestamp', ascending=True)
    df_distance = df_distance.astype({'Distance': int}, errors='raise')

    return df_distance

  def get_sequence_actions(self, episodes_names=[]):
    i = 0
    incremental_id = 0

    if not episodes_names: episodes_names = self.episodes_names

    df_sequence_length = pd.DataFrame()
    for episode_data in self.episodes_data:
      if episode_data.episode_name in episodes_names:
        df_sequence_length = pd.concat(
          [df_sequence_length, episode_data.compute_action_sequences_length()],
          axis=0,
          ignore_index=True)

    df_sequence_length.insert(0, 'Sequence ID',
                              -1 * np.ones(df_sequence_length.shape[0], int))
    column = 'Sequence length'

    while i + 1 < df_sequence_length.shape[0]:
      if df_sequence_length.loc[i + 1, column] == 1:
        df_sequence_length.iloc[i, 0] = incremental_id
        i = i + 1
      else:
        j = 2
        while (i + j) < df_sequence_length.shape[0] and df_sequence_length.loc[
          i + j, column] != 1: j = j + 1
        df_sequence_length.loc[i:i + j, 'Sequence ID'] = incremental_id
        i = i + j
      incremental_id = incremental_id + 1
    df_sequence_length.loc[
      df_sequence_length['Sequence ID'] == -1, 'Sequence ID'] = incremental_id
    grp = df_sequence_length.groupby('Sequence ID')
    return grp

  def display_sequence_actions(self, episodes_names=[], sequence_range=None):

    grp = self.get_sequence_actions(episodes_names=episodes_names)

    list_dict = []
    columns = ['Timestamp', 'Sequence start', 'Sequence end', 'Sequence length',
               'NB actions', 'NB unitary actions',
               'Impacted Subs', 'Impacted lines']

    if not sequence_range:
      sequence_range = range(0, 400)

    for _, group in grp:

      if group.shape[0] in sequence_range:
        start_timestamp = group.iloc[0, 1]
        end_timestamp = group.iloc[group.shape[0] - 1, 1]
        sequence_length = group.shape[0]

        # count total number of impacted subs
        nb_impacted_subs = len(
          [item for sublist in group['Impacted subs'] for item in sublist])
        # count total number of impacted lines
        nb_impacted_lines = len(
          [item for sublist in group['Impacted lines'] for item in sublist])
        # number of total actions
        nb_actions = nb_impacted_subs + nb_impacted_lines

        nb_unitary_actions = group['NB action'].sum()

        impacted_subs = group['Impacted subs'].tolist()
        impacted_lines = group['Impacted lines'].tolist()

        dict_row = {'Timestamp': group['Timestamp'].tolist(),
                    'Sequence start': start_timestamp,
                    'Sequence end': end_timestamp,
                    'Sequence length': sequence_length,
                    'NB actions': nb_actions,
                    'NB unitary actions': nb_unitary_actions,
                    'Impacted Subs': impacted_subs,
                    'Impacted lines': impacted_lines}
        list_dict.append(dict_row)

    return pd.DataFrame(list_dict, columns=columns)

  def display_sequence_actions_filter(self, episodes_names=[],
                                      sequence_range=None):

    grp = self.get_sequence_actions(episodes_names)
    if not sequence_range: sequence_range = range(0, 400)
    for _, group in grp:
      if group.shape[0] in sequence_range:
        df = group.drop(['Sequence ID', 'Sequence length'], axis=1)
        df = df.rename(columns={'NB action': 'NB unitary actions'})
        display(df)

  def action_sequences_to_dict(self, episodes_names=[], sequence_range=None):

    df = self.display_sequence_actions(episodes_names, sequence_range)
    max_lenght = df['Sequence length'].max()
    dict_list = []

    for i in range(df.shape[0]):
      dict_list.append(
        dict(Type=self.agent_name, Start=str(df.loc[i, 'Sequence start']),
             Finish=str(df.loc[i, 'Sequence end']),
             Actions=df.loc[i, 'Sequence length'], Actions_percent=(df.loc[
                                                                      i, 'NB unitary actions'] / max_lenght) * 100, ))
    return dict_list

  def concat_overloaded_lines_freq(self, episodes_names=[], ):

    if not episodes_names: episodes_names = self.episodes_names

    overloaded_lines = []

    for episode_data in self.episodes_data:
      if episode_data.episode_name in episodes_names:
        overloaded_lines_for_episode_i = episode_data.overloaded_lines_by_timestamp()
        overloaded_lines_flatten = [list(item) for dict in
                                    overloaded_lines_for_episode_i for key, item
                                    in dict.items()
                                    if key == 'Overloaded lines']
        overloaded_lines = overloaded_lines + overloaded_lines_flatten

    overloaded_lines_flatten = list(itertools.chain(*overloaded_lines))

    data = [overloaded_lines_flatten.count(x) for x in
            range(self.episodes_data[0]._n_lines())]

    df = pd.DataFrame(np.array(
      [self.episodes_data[0]._name_of_lines(range(len(data))),
       data]).transpose(),
                      columns=['Line', 'Overloaded'])
    df = df.astype({'Line': str, 'Overloaded': int}, errors='raise')
    df = df.loc[df['Overloaded'] != 0]

    return df

  def concat_disconnected_lines_freq(self, episodes_names=[], ):

    if not episodes_names: episodes_names = self.episodes_names

    disconnected_lines = []

    for episode_data in self.episodes_data:
      if episode_data.episode_name in episodes_names:
        disconnected_lines_for_episode_i = episode_data.disconnected_lines_by_timestamp()
        disconnected_lines_flatten = [list(item) for dict in
                                      disconnected_lines_for_episode_i for
                                      key, item in
                                      dict.items() if
                                      key == 'Disconnected lines']
        disconnected_lines = disconnected_lines + disconnected_lines_flatten

    disconnected_lines_flatten = list(itertools.chain(*disconnected_lines))

    data = [disconnected_lines_flatten.count(x) for x in
            range(self.episodes_data[0]._n_lines())]

    df = pd.DataFrame(np.array(
      [self.episodes_data[0]._name_of_lines(range(len(data))),
       data]).transpose(),
                      columns=['Line', 'Disconnected'])
    df = df.astype({'Line': str, 'Disconnected': int}, errors='raise')
    df = df.loc[df['Disconnected'] != 0]

    return df

  # TODO optmize this --> Fereshteh

  def display_detailed_action_type(self, episodes_names=[]):

    data_display = display(display_id='data_display')
    output_display = display(display_id='output_display')
    grid = qgrid.QGridWidget(df=pd.DataFrame())

    w = widgets.Dropdown(
      options=['Select', 'Tolopology', 'Force_line', 'Redispatching',
               'Injection', 'Curtailment', 'Storage'],
      value='Select', description='Table', )

    def on_change(change):
      if change['type'] == 'change' and change['name'] == 'value':

        result = pd.DataFrame()
        c = 0
        for episode_data in self.episodes_data:
          if (not len(
              episodes_names)) or episode_data.episode_name in episodes_names:
            functions = {'Tolopology': episode_data.create_topology_df,
                         'Force_line': episode_data.create_force_line_df,
                         'Redispatching': episode_data.create_dispatch_df,
                         'Injection': episode_data.create_injection_df,
                         'Curtailment': episode_data.create_curtailment_df,
                         'Storage': episode_data.create_storage_df}
            r = functions[change['new']]()
            r[1]['episode_name'] = episode_data.episode_name
            result = pd.concat([result, r[1]])
            c += r[0]
        output_display.update(
          'total Number of ' + change['new'] + ' changes:' + str(c))
        grid.df = result

    w.observe(on_change)
    #         ouptup_display = display(display_id='ouptup_display')

    display(w)
    output_display.display('')
    data_display.display(grid)
    return
