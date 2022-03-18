"""
Please do not use, functions are migrated to Episode Data Extractor,
use EpisodeDataExtractor and EpisdesPlot instead
"""
import os
import numpy as np
import pandas as pd
from grid2op.Episode import EpisodeData
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from datetime import datetime
from collections import Counter
import datetime
import qgrid

from IPython.display import display
import ipywidgets as widgets


class BenchPlot:
    """

    """
    def __init__(self, agent_path, episode_name):
        """

        :param agent_path:
        :param episode_name:
        """
        self.agent_path = os.path.abspath(agent_path)
        self.episode_name = episode_name

        self.episode_data = self.load_episode_data()
        self.actions = self.get_actions()
        self.n_action = len (self.actions)
        self.observations = self.get_observations()
        self.computation_times = self.get_computation_times()



    def load_all_episodes_data(self, episodes_names=[]):
        """

        :return:
        """
        if not episodes_names : episodes_names = [ name for name in os.listdir(self.agent_path) if os.path.isdir(os.path.join(self.agent_path, name)) ]

        return [EpisodeData.from_disk(agent_path=self.agent_path, name=episode_name) for episode_name in episodes_names]

    def load_episode_data(self):
        """

        :return:
        """

        return EpisodeData.from_disk(agent_path=self.agent_path, name=self.episode_name)

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

    def get_computation_times(self):
        """

        :return:
        """

        #return [computation_time for computation_time in self.episode_data.times if not np.isnan(computation_time)]
        return self.episode_data.times


    def get_current_timestamp(self, step):
        """

        :return:
        """
        return self.observations[step].get_time_stamp()


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
                    # number of forced lines
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

        nb_force_line = []
        nb_switch_line = []
        nb_topological_changes = []
        nb_redispatch_changes = []
        nb_storage_changes = []
        nb_curtailment_changes = []

        for action in actions:
            action_impact = action.impact_on_objects()

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
            'nb_line_forced': nb_force_line,
            'nb_line_switched': nb_switch_line,
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

    def get_overloaded_lines_by_timestamp(self):
        """

        :return:
        """
        observations = self.get_observations()
        overloaded_lines = []

        for observation, i in zip(observations, range(len(observations))):
            lines_id = np.where(observation.timestep_overflow != 0)[0]
            if len(lines_id) != 0:
                overloaded_lines.append(
                    {
                        "timestamp": i,
                        "lines_overloaded": set(lines_id)
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
            lines_id = np.where(observation.line_status != True)[0]
            if len(lines_id) != 0:
                disconnected_lines.append(
                    {
                        "timestamp": i,
                        "lines_disconnected": set(lines_id)
                    }
                )

        return disconnected_lines

    def plot_actions_freq_by_type(self):
        """

        :return:
        """
        dict_actions_freq = self.get_actions_freq_by_type()

        df_actions_freq_by_type = pd.DataFrame(columns=["action type", "frequency"])


        for key, values in dict_actions_freq.items():
            for char in ["nb", "_"]: key = key.replace(char, " ")
            df_actions_freq_by_type = df_actions_freq_by_type.append({'action type': key, 'frequency': sum(values)},
                                                                     ignore_index=True)

        fig = px.pie(df_actions_freq_by_type, values='frequency', names='action type',
                     title='Frequency of actions by type', )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig


    def plot_actions_sequence_length_by_type(self):
        """

        :return:
        """
        dict_list=[]
        actions_freq_by_type, max = self.compute_action_sequences_lengh()

        for key, value in actions_freq_by_type.items():
            i = len(value) - 1
            while i >= 0:
                if value[i] != 0:
                    end = self.get_current_timestamp(i + 1)
                    start = end - timedelta(minutes=5) * value[i]
                    dict_list.append(dict(Task=key, Start=str(start), Finish=str(end), Actions=value[i],
                                   Actions_percent=(value[i] / max) * 100))
                    i = i - value[i]
                else:
                    i = i - 1

        fig = px.timeline(dict_list, x_start="Start", x_end="Finish", y="Task", color="Actions",
                          color_continuous_scale=["green", "red"])
        return fig

    def compute_action_sequences_lengh(self): #helper function
        """

        :return:
        """
        action_sequences_lengh = self.get_actions_freq_by_type()
        max = 0
        for key, value in action_sequences_lengh.items():

            for i in range(len(value)):
                if value[i] != 0:
                    if value[i - 1] == 0 and (i - 1) >= 0:
                        action_sequences_lengh[key][i] = 1
                    elif value[i - 1] != 0:
                        action_sequences_lengh[key][i] = action_sequences_lengh[key][i - 1] + 1
                    if max < action_sequences_lengh[key][i]: max = action_sequences_lengh[key][i]

        return action_sequences_lengh, max


    def plot_actions_freq_by_station(self):
        """

        :return:
        """

        actions_freq_by_station = self.get_actions_freq_by_station()
        impacted_stations_flatten = []

        for item in actions_freq_by_station:
            impacted_stations_flatten.extend(list(item["subs_impacted"]))

        x = list(Counter(impacted_stations_flatten).keys())
        y = list(Counter(impacted_stations_flatten).values())

        fig = px.bar(y=y, x=x, text_auto='.2s',
                     title="")
        fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)

        return fig


    def plot_overloaded_disconnected_lines_freq(self):
        """

        :return:
        """
        #overloaded lines frequency
        overloaded_lines_by_timestep = self.get_overloaded_lines_by_timestamp()
        overloaded_lines_flatten = []

        for item in overloaded_lines_by_timestep:
            overloaded_lines_flatten.extend(list(item["lines_overloaded"]))

        #disconnected lines frequency
        disconnected_lines_by_timestep = self.get_disconnected_lines_by_timestamp()
        disconnected_lines_flatten = []

        for item in disconnected_lines_by_timestep:
            disconnected_lines_flatten.extend(list(item["lines_disconnected"]))

        data = [[overloaded_lines_flatten.count(x), disconnected_lines_flatten.count(x)] for x in
                range(self._n_lines())]
        df = pd.DataFrame(data)
        df = df.loc[~(df == 0).all(axis=1)]

        fig = px.bar(df, y=[0, 1], x=self._name_of_lines(df.index), text_auto='.2s',
                     title="")
        fig.update_traces(textfont_size=12, textangle=0, cliponaxis=False)

        return fig

    def _n_lines(self):
        """

        :return:
        """
        return self.observations[0].n_line

    def _name_of_lines(self, lines_id):
        """

        :return:
        """
        return self.actions[0].name_line[lines_id]



    def plot_distance_from_intial_topology(self):
        """

        :return:
        """

        x = []
        y = []

        for i in range(self.n_action):
            act = self.actions[i]
            obs = self.observations[i]
            # True == connected, False == disconnect
            # So that len(line_statuses) - line_statuses.sum() is the distance for lines
            line_statuses = self.observations[i].line_status
            # True == sub has something on bus 2, False == everything on bus 1
            # So that subs_on_bus2.sum() is the distance for subs
            subs_on_bus_2 = np.repeat(False, self.observations[i].n_sub)
            # objs_on_bus_2 will store the id of objects connected to bus 2
            objs_on_bus_2 = {id: [] for id in range(self.observations[i].n_sub)}
            distance, _, _, _ = self.get_distance_from_obs(act, line_statuses, subs_on_bus_2, objs_on_bus_2, obs)

            x.append(self.get_current_timestamp(i))
            y.append(distance)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, name="linear",
                                 line_shape='hvh'))

        fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16))
        return fig

    def plot_computation_times(self):
        """

        :return:
        """

        x = []
        y = []
        for i in range(self.n_action):
            x.append(self.get_current_timestamp(i))
            y.append(self.get_computation_times()[i])
        fig = px.line(x=x, y=y)

        return fig

















    # reused from grid2vis
    def get_distance_from_obs(
            self, act, line_statuses, subs_on_bus_2, objs_on_bus_2, obs
    ):

        impact_on_objs = act.impact_on_objects()

        # lines reconnetions/disconnections
        line_statuses[
            impact_on_objs["force_line"]["disconnections"]["powerlines"]
        ] = False
        line_statuses[
            impact_on_objs["force_line"]["reconnections"]["powerlines"]
        ] = True
        line_statuses[impact_on_objs["switch_line"]["powerlines"]] = np.invert(
            line_statuses[impact_on_objs["switch_line"]["powerlines"]]
        )

        topo_vect_dict = {
            "load": obs.load_pos_topo_vect,
            "generator": obs.gen_pos_topo_vect,
            "line (extremity)": obs.line_ex_pos_topo_vect,
            "line (origin)": obs.line_or_pos_topo_vect,
        }

        # Bus manipulation
        if impact_on_objs["topology"]["changed"]:
            for modif_type in ["bus_switch", "assigned_bus"]:

                for elem in impact_on_objs["topology"][modif_type]:
                    objs_on_bus_2 = self.update_objs_on_bus(
                        objs_on_bus_2, elem, topo_vect_dict, kind=modif_type
                    )

            for elem in impact_on_objs["topology"]["disconnect_bus"]:
                # Disconnected bus counts as one for the distance
                subs_on_bus_2[elem["substation"]] = True

        subs_on_bus_2 = [
            True if objs_on_2 else False for _, objs_on_2 in objs_on_bus_2.items()
        ]

        distance = len(line_statuses) - line_statuses.sum() + sum(subs_on_bus_2)
        return distance, line_statuses, subs_on_bus_2, objs_on_bus_2

    def update_objs_on_bus(self, objs_on_bus_2, elem, topo_vect_dict, kind):
        for object_type, pos_topo_vect in topo_vect_dict.items():
            if elem["object_type"] == object_type and elem["bus"]:
                if kind == "bus_switch":
                    objs_on_bus_2 = self.update_objs_on_bus_switch(
                        objs_on_bus_2, elem, pos_topo_vect
                    )
                else:
                    objs_on_bus_2 = self.update_objs_on_bus_assign(
                        objs_on_bus_2, elem, pos_topo_vect
                    )
                break
        return objs_on_bus_2

    @staticmethod
    def update_objs_on_bus_switch(objs_on_bus_2, elem, pos_topo_vect):
        if pos_topo_vect[elem["object_id"]] in objs_on_bus_2[elem["substation"]]:
            # elem was on bus 2, remove it from objs_on_bus_2
            objs_on_bus_2[elem["substation"]] = [
                x
                for x in objs_on_bus_2[elem["substation"]]
                if x != pos_topo_vect[elem["object_id"]]
            ]
        else:
            objs_on_bus_2[elem["substation"]].append(pos_topo_vect[elem["object_id"]])
        return objs_on_bus_2

    @staticmethod
    def update_objs_on_bus_assign(objs_on_bus_2, elem, pos_topo_vect):
        if (
                pos_topo_vect[elem["object_id"]] in objs_on_bus_2[elem["substation"]]
                and elem["bus"] == 1
        ):
            # elem was on bus 2, remove it from objs_on_bus_2
            objs_on_bus_2[elem["substation"]] = [
                x
                for x in objs_on_bus_2[elem["substation"]]
                if x != pos_topo_vect[elem["object_id"]]
            ]
        elif (
                pos_topo_vect[elem["object_id"]] not in objs_on_bus_2[elem["substation"]]
                and elem["bus"] == 2
        ):
            objs_on_bus_2[elem["substation"]].append(pos_topo_vect[elem["object_id"]])
        return objs_on_bus_2














    def get_episode_timestamps(self):
        """
        This function returns a list of timestamps which associates to echa time step of the episode.
        NB: The len(observations) = len(actions)+1
        """
        observations = self.get_observations()
        timestamp_list = []

        for i in range(0,len(observations)):
            obs=observations[i]
            timestamp_list.append(datetime.datetime(obs.year, obs.month, obs.day, hour= obs.hour_of_day, minute= obs.minute_of_hour ))

        return timestamp_list


    def plot_action_freq_with_slider(self):
        """
        NB: The len(time_l) = len(data)+1
        """
        time_l= self.get_episode_timestamps()
        d=self.get_actions_freq_by_type()

        action_l= [i for i in d]
        data=pd.DataFrame.from_dict(d)

        fig = px.bar(data, x= time_l[:-1] , y=action_l).update_layout(
            xaxis={
                "range": [data.index, data.index.max()],
                "rangeslider": {"visible": True},
            })

        fig.update_layout(title_text="Action Frequency per time stamp")

        return fig


    def create_topology_df(self):
        c1 =0
        c2 = 0
        c3 = 0

        topo_df = pd.DataFrame(columns= [ 't_step', 'time_stamp', 'type',  'object_type', 'object_id', 'susbtation'])
        for i in range(0, len(self.episode_data.actions)):

            # to extract time stamp, we need the observation object:
            obs= self.episode_data.observations[i]
            t = datetime.datetime(obs.year, obs.month, obs.day, hour= obs.hour_of_day, minute= obs.minute_of_hour )

            d= self.episode_data.actions[i].impact_on_objects()
            if d['has_impact']:
                c1+= len(d['topology']['disconnect_bus'])
                c2+= len(d['topology']['bus_switch'])
                c3+= len(d['topology']['assigned_bus'])
                for j in d:
                    if ( type(d[j])is dict) and (d[j]['changed']):
                        if j== 'topology':
                            if d['topology']['bus_switch']:
                                for n in range(0,len(d['topology']['bus_switch'])):
                                    o_type= d['topology']['bus_switch'][n]['object_type']
                                    o_id = d['topology']['bus_switch'][n]['object_id']
                                    subs = d['topology']['bus_switch'][n]['substation']
                                    topo_df.loc[len(topo_df)]= [i,t, 'swithc_bus', o_type, o_id, subs]

                            if d['topology']['assigned_bus']:
                                for n in range(0,len(d['topology']['assigned_bus'])):
                                    #print(d['topology']['assigned_bus'][n])
                                    print('assigned_bus')
                                    o_type= d['topology']['assigned_bus'][n]['object_type']
                                    o_id = d['topology']['assigned_bus'][n]['object_id']
                                    subs = d['topology']['assigned_bus'][n]['substation']
                                    topo_df.loc[len(topo_df)]= [i,t, 'assigned_bus', o_type, o_id, subs]

                            if d['topology']['disconnect_bus']:
                                for n in range(0,len(d['topology']['disconnect_bus'])):
                                    o_type= d['topology']['disconnect_bus'][n]['object_type']
                                    o_id = d['topology']['disconnect_bus'][n]['object_id']
                                    subs = d['topology']['disconnect_bus'][n]['substation']
                                    topo_df.loc[len(topo_df)]= [i,t,'disconnect_bus', o_type, o_id, subs]
#       print("total Number of topology changes:" ,c1+c2+c3)
#         df = qgrid.show_grid(topo_df)
    
        return [c1+c2+c3, topo_df]


    def create_injection_df(self):
        c =0
        inj_df = pd.DataFrame(columns= [ 't_step', 'time_stamp', 'count',  'impacted'])
        for i in range(0, len(self.episode_data.actions)):

            obs=self.episode_data.observations[i]
            t = datetime.datetime(obs.year, obs.month, obs.day, hour= obs.hour_of_day, minute= obs.minute_of_hour )

            d= self.episode_data.actions[i].impact_on_objects()
            if d['has_impact']:

                for j in d:
                    if ( type(d[j])is dict) and (d[j]['changed']):
                        if j== 'injection':
                            c+= d['injection']['count']
                            co= d['injection']['count']
                            impacted= d['injection']['impacted']
                            inj_df.loc[len(inj_df)]= [i,t, co, impacted]

#         print('total number of injection:', c)
#         df = qgrid.show_grid(inj_df)
        return(c, inj_df)


    def create_dispatch_df(self):
        c = 0
        dispatch_df = pd.DataFrame(columns= ['t_step','time_stamp','generator_id', 'generator_name', 'amount'])

        for i in range(0, len(self.episode_data.actions)):

            obs=self.episode_data.observations[i]
            t = datetime.datetime(obs.year, obs.month, obs.day, hour= obs.hour_of_day, minute= obs.minute_of_hour )

            d= self.episode_data.actions[i].impact_on_objects()
            if d['has_impact']:
                for j in d:
                    if ( type(d[j])is dict) and (d[j]['changed']):
                        #print(d[j])
                        if j == 'redispatch':
                            c+=1
                            gen= (d[j]['generators'][0])
                            gen_id= gen['gen_id']
                            gen_name = gen['gen_name']
                            amount = gen['amount']
                            #print([i, gen_id, gen_name, amount ])
                            dispatch_df.loc[len(dispatch_df)]= [i,t, gen_id, gen_name, amount]
#         print('total numnber of redispatches:', c)
#         df = qgrid.show_grid(dispatch_df)
        return(c, dispatch_df)
    
    def create_force_line_df(self):
        c1 =0
        c2 = 0
        c3 = 0
        line_df = pd.DataFrame(columns= [ 't_step', 'time_stamp', 'type',  'powerline'])
        for i in range(0, len(self.episode_data.actions)):

            # to extract time stamp, we need the observation object:
            obs=self.episode_data.observations[i]
            t = datetime.datetime(obs.year, obs.month, obs.day, hour= obs.hour_of_day, minute= obs.minute_of_hour )

            d= self.episode_data.actions[i].impact_on_objects()
            if d['has_impact']:
                for j in d:
                    if ( type(d[j])is dict) and (d[j]['changed']):
                         if j == 'force_line':
                            c1 += d[j]['reconnections']['count']
                            c2 += d[j]['disconnections']['count']
                            if c1>0:
                                line_df.loc[len(line_df)]= [i,t,'reconnection', d[j]['reconnections']['powerlines']]
                            if c2>0:
                                line_df.loc[len(line_df)]= [i,t,'disconnection', d[j]['reconnections']['powerlines']]

        #print("total Number of force_line changes:" ,c1+c2)  
#         df = qgrid.show_grid(line_df)
        return(c1+c2, line_df)

    
    def display_detailed_action_type(self):

        data_display = display(display_id="data_display")
        output_display = display(display_id="output_display")
        grid = qgrid.QGridWidget(df=pd.DataFrame())

        w = widgets.Dropdown(
            options=['Select','Tolopology changes','Force_line', 'Redispatching', 'Injection', 'Switch line'],
            value='Select',
            description='Table',
        )
        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':

                if change['new'] =='Tolopology changes':
                    result = self.create_topology_df()
                    output_display.update("total Number of topology changes:" + str(result[0]))
                    grid.df=result[1]

                if change['new'] =='Redispatching':
                    result = self.create_dispatch_df()
                    output_display.update("total Number of Redispatching changes:" + str(result[0]))
                    grid.df=result[1]

                if change['new'] == "Injection":
                    result = self.create_injection_df()
                    output_display.update("total Number of Injections changes:" + str(result[0]))
                    grid.df=result[1]
                    
                if change['new'] == "Force_line":
                    result = self.create_force_line_df()
                    output_display.update("total Number of Force_line changes:" + str(result[0]))
                    grid.df=result[1]

        w.observe(on_change)
#         ouptup_display = display(display_id="ouptup_display")

        display(w)
        output_display.display('')
        data_display.display(grid)
        return