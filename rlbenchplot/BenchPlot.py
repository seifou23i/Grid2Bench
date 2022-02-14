import os
import numpy as np
import pandas as pd
from grid2op.Episode import EpisodeData
import plotly.express as px
import datetime

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
        self.observations = self.get_observations()
        self.computation_times = self.get_computation_times()

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

        return [computation_time for computation_time in self.episode_data.times if not np.isnan(computation_time)]


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
#         print("total Number of topology changes:" ,c1+c2+c3)                            
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
        return(c, dispatch_df)

    
    def present_detailed_action_type(self):
        
        data_display = display(display_id="data_display")
        output_display = display(display_id="output_display")

        w = widgets.Dropdown(
            options=['Select','Tolopology changes', 'Redispatching', 'Injection', 'Switch line'],
            value='Select',
            description='Table', 
        )
        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':

                if change['new'] =='Tolopology changes': 
                    result = self.create_topology_df()
                    output_display.update("total Number of topology changes:" + str(result[0]))
                    data_display.update(result[1])

                if change['new'] =='Redispatching': 
                    result = self.create_dispatch_df()
                    output_display.update("total Number of Redispatching changes:" + str(result[0]))
                    data_display.update(result[1])

                if change['new'] == "Injection":
                    result = self.create_injection_df()
                    output_display.update("total Number of Injections changes:" + str(result[0]))
                    data_display.update(result[1])

        w.observe(on_change)
#         ouptup_display = display(display_id="ouptup_display")

        display(w)
        output_display.display('')
        data_display.display('')
        return