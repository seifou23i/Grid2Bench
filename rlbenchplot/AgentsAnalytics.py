import os
from rlbenchplot.EpisodesPlot import EpisodesPlot
import pandas as pd
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime
import ipywidgets as widgets
from ipywidgets import interact

class AgentsAnalytics:
    """

    """

    def __init__(self, data_path, agents_names=[],episodes_names=[]):
        """

        :param data_path: parent directory path for agent log files
        :param agents_names: a list of each agent repository name, if empty (not recommended) it will load all repositories in the data path
        :param episodes_names: a list of episode names, must be the same on all agents
        """
        self.data_path = data_path
        self.agents_names = agents_names
        self.episodes_names = episodes_names

        self.agents_data = self.load_agents_resutls()

    def load_agents_resutls(self):
        """

        :return:
        """
        if not self.agents_names :
            self.agents_names = [name for name in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, name)) ]

        return [EpisodesPlot(agent_path=os.path.join(self.data_path, agent_name), episodes_names=self.episodes_names) for agent_name in self.agents_names]



    @staticmethod
    def plot_actions_freq_by_station(
            agents_results=[],
            episodes_names=[],
            title="Frequency of actions by station",
            **fig_kwargs):
        """

        :param agents_results: list of agent objects of class "Agents_Evaluation " or class "Episode_Plot"
        :param episodes_names: filter some episodes, if empty it will show all loaded episodes
        :param title: plot title, if empty it will return default value
        :param fig_kwargs: keyword for plotly arguments, example: height= 700
        :return:
        """
        agent_names = []

        # for the first agent
        agent_names.append(agents_results[0].agent_name)
        data = agents_results[0].plot_actions_freq_by_station(episodes_names).data[0]
        x, y = data.x, data.y
        df = pd.DataFrame(np.array([x, y]).transpose(), columns=['Substation', agents_results[0].agent_name])

        for agent in agents_results[1:]:
            agent_names.append(agent.agent_name)
            data = agent.plot_actions_freq_by_station(episodes_names).data[0]
            x, y = data.x, data.y
            df2 = pd.DataFrame(np.array([x, y]).transpose(), columns=['Substation', agent.agent_name])
            df = df.join(df2.set_index('Substation'), on='Substation')

        newnames = {}
        y_list = []
        for i in range(len(agent_names)):
            newnames["wide_variable_{}".format(i)] = agent_names[i]
            y_list.append(df[agent_names[i]].to_list())

        fig = px.bar(x=df["Substation"].to_list(),
                     y=y_list,
                     text_auto='.2s',
                     labels={
                         "x": "Station",
                         "value": "Frequency"},
                     barmode="group",
                     title=title)

        fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                              legendgroup=newnames[t.name],
                                              hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name])
                                              )
                           )

        fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)

        fig.update_layout(**fig_kwargs)

        return fig

    @staticmethod
    def plot_actions_freq_by_type(
            agents_results=[],
            episodes_names=[],
            title="Frequency of actions by type",
            row=1,
            col=2,
            **fig_kwargs):
        """

        :param agents_results: list of agent objects of class "Agents_Evaluation " or class "Episode_Plot"
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

        fig = make_subplots(row, col, specs=list_i,
                            subplot_titles=agent_names)

        for i in range(row):
            for j in range(col):
                data = agents_results[i + j].plot_actions_freq_by_type(episodes_names).data[0]
                fig.add_trace(go.Pie(labels=data.labels.tolist(), values=data.values.tolist(),
                                     name=agent_names[i + j]), i + 1, j + 1)

        fig.update_traces(textposition='inside')
        fig.update_layout(title_text=title, uniformtext_minsize=12, uniformtext_mode='hide', **fig_kwargs)
        return fig

    @staticmethod
    def plot_actions_freq_by_station_pie_chart(
            agents_results=[],
            episodes_names=[],
            title="Frequency of actions by station",
            row=1,
            col=2,
            **fig_kwargs):
        """
        :param agents_results: list of agent objects of class "Agents_Evaluation " or class "Episode_Plot"
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

        fig = make_subplots(row, col, specs=list_i,
                            subplot_titles=agent_names)

        for i in range(row):
            for j in range(col):
                data = agents_results[i + j].plot_actions_freq_by_station_pie_chart(episodes_names).data[0]
                fig.add_trace(go.Pie(labels=data.labels.tolist(), values=data.values.tolist(),
                                     name=agent_names[i + j]), i + 1, j + 1)

        fig.update_traces(textposition='inside')
        fig.update_layout(title_text=title, uniformtext_minsize=12, uniformtext_mode='hide', **fig_kwargs)
        return fig

    @staticmethod
    def plot_lines_impact(
            agents_results=[],
            episodes_names=[],
            title="Overloaded Lines by station",
            disconnected=False,
            **fig_kwargs):
        """
        :param agents_results: list of agent objects of class "Agents_Evaluation " or class "Episode_Plot"
        :param episodes_names: filter some episodes, if empty it will show all loaded episodes
        :param title: plot title, if empty it will return default value
        :param disconnected, if True plots disconnected lines, else draws overflowed lines
        :param fig_kwargs: keyword for plotly arguments, example: height= 700

        """
        k = 0
        agent_names = []
        if disconnected:
            k = 1
            if not title : title = "Disconnected Lines by station"

        # for the first agent
        agent_names.append(agents_results[0].agent_name)
        data = agents_results[0].plot_overloaded_disconnected_lines_freq(episodes_names).data[k]
        x, y = data.x, data.y
        df = pd.DataFrame(np.array([x, y]).transpose(), columns=['Line', agents_results[0].agent_name])

        for agent in agents_results[1:]:
            agent_names.append(agent.agent_name)
            data = agent.plot_overloaded_disconnected_lines_freq(episodes_names).data[k]
            x, y = data.x, data.y
            df2 = pd.DataFrame(np.array([x, y]).transpose(), columns=['Line', agent.agent_name])
            df = df.join(df2.set_index('Line'), on='Line')

        newnames = {}
        y_list = []
        for i in range(len(agent_names)):
            newnames["wide_variable_{}".format(i)] = agent_names[i]
            y_list.append(df[agent_names[i]].to_list())

        fig = px.bar(x=df["Line"].to_list(),
                     y=y_list,
                     text_auto='.2s',
                     labels={
                         "x": "Line",
                         "value": "Frequency"},
                     barmode="group",
                     title=title,
                     log_y=True, )

        fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                              legendgroup=newnames[t.name],
                                              hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name])
                                              )
                           )

        fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)

        fig.update_layout(**fig_kwargs)

        return fig

    @staticmethod
    def plot_computation_times(
            agents_results=[],
            episodes_names=[],
            title="Action Execution Time",
            **fig_kwargs):
        """

        :param agents_results: list of agent objects of class "Agents_Evaluation " or class "Episode_Plot"
        :param episodes_names: filter some episodes, if empty it will show all loaded episodes
        :param title: plot title, if empty it will return default value
        :param fig_kwargs: keyword for plotly arguments, example: height= 700
        :return:
        """
        agent_names = []

        # for the first agent
        agent_names.append(agents_results[0].agent_name)
        data = agents_results[0].plot_computation_times(episodes_names).data[0]
        x, y = data.x, data.y
        df = pd.DataFrame(np.array([x, y]).transpose(), columns=['Timestamp', agents_results[0].agent_name])

        for agent in agents_results[1:]:
            agent_names.append(agent.agent_name)
            data = agent.plot_computation_times(episodes_names).data[0]
            x, y = data.x, data.y
            df2 = pd.DataFrame(np.array([x, y]).transpose(), columns=['Timestamp', agent.agent_name])
            df = df.join(df2.set_index('Timestamp'), on='Timestamp')

        # Create traces
        fig = go.Figure()

        for agent_name in agent_names:
            fig.add_trace(go.Scatter(x=df["Timestamp"].tolist(), y=df[agent_name].tolist(),
                                     mode='lines+markers',
                                     name=agent_name))

        fig.update_layout(xaxis={"rangeslider": {"visible": True}}, title=title, xaxis_title="Timestamp",
                          yaxis_title="Execution Time (s)")
        fig.update_layout(**fig_kwargs)

        return fig


    @staticmethod
    def plot_distance_from_initial_topology(
            agents_results=[],
            episodes_names=[],
            title="Distance from initial topology",
            **fig_kwargs):
        """
        :param agents_results: list of agent objects of class "Agents_Evaluation " or class "Episode_Plot"
        :param episodes_names: filter some episodes, if empty it will show all loaded episodes
        :param title: plot title, if empty it will return default value
        :param fig_kwargs: keyword for plotly arguments, example: height= 700
        :return:
        """
        # TODO : creating a function to reuse this

        # for the first agent
        agent_names = [agents_results[0].agent_name]
        data = agents_results[0].plot_distance_from_initial_topology(episodes_names).data[0]
        x, y = data.x, data.y
        df = pd.DataFrame(np.array([x, y]).transpose(), columns=['Timestamp', agents_results[0].agent_name])

        for agent in agents_results[1:]:
            agent_names.append(agent.agent_name)
            data = agent.plot_distance_from_initial_topology(episodes_names).data[0]
            x, y = data.x, data.y
            df2 = pd.DataFrame(np.array([x, y]).transpose(), columns=['Timestamp', agent.agent_name])
            df = df.join(df2.set_index('Timestamp'), on='Timestamp')

        # Create traces
        fig = go.Figure()

        for agent_name in agent_names:
            fig.add_trace(go.Scatter(x=df["Timestamp"].tolist(), y=df[agent_name].tolist(),
                                     mode='lines+markers', line_shape='hvh',
                                     name=agent_name))

        fig.update_layout(xaxis={"rangeslider": {"visible": True}}, title=title, xaxis_title="Timestamp",
                          yaxis_title="Distance")
        fig.update_layout(**fig_kwargs)

        return fig

    @staticmethod
    def plot_actions_sequence_length(
            agents_data=[],
            episodes_names=[],
            title="Sequence length of actions",
            sequence_range=None,
            **fig_kwargs):

        plot_data = []
        for agent in agents_data:
            plot_data.extend(agent.action_sequences_to_dict(episodes_names, sequence_range))

        fig = px.timeline(
            plot_data,
            x_start="Start",
            x_end="Finish",
            y="Type",
            color="Actions",
            labels={
                "Type": "Agent ",
                "Actions": "Action Sequence"},
            title=title,
            color_continuous_scale=["green", "red"],
        )
        fig.update_layout(xaxis={"rangeslider": {"visible": True}})
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
            columns=["Episode", "Played timesteps", "Cumulative reward"],
        )

        return df.astype({'Episode': 'str', "Played timesteps": int, "Cumulative reward": float})


    @staticmethod
    def plot_cumulative_reward(
            agents_data=[],
            episodes_names=[],
            title="Cumulative reward per episode(CR) vs Accomplished time steps (Ats)",
            **fig_kwargs):

        if not episodes_names : episodes_names= agents_data[0].episodes_names

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        colors = px.colors.qualitative.G10

        # Add cumulative reward traces
        for i, agent in zip(range(len(agents_data)), agents_data):
            df = AgentsAnalytics.cumulative_reward(agent, episodes_names)
            agent_episodes = df["Episode"].tolist()
            agent_cum_reward = df["Cumulative reward"].tolist()

            # Add traces
            fig.add_trace(
                go.Scatter(x=agent_episodes, y=agent_cum_reward, name="CR : {}".format(agent.agent_name),
                           line=dict(width=2, color=colors[i], )),
                secondary_y=False,
            )

        # Add played time steps traces
        for i, agent in zip(range(len(agents_data)), agents_data):
            df = AgentsAnalytics.cumulative_reward(agent, episodes_names)
            agent_episodes = df["Episode"].tolist()
            agent_played_timesteps = df["Played timesteps"].tolist()

            # Add traces
            fig.add_trace(
                go.Scatter(x=agent_episodes, y=agent_played_timesteps, name="Ats : {}".format(agent.agent_name),
                           line=dict(width=1, dash='dot', color=colors[i], )),
                secondary_y=True,
            )

        # Add figure title
        fig.update_layout(
            title_text=title
        )

        # Set x-axis title
        fig.update_xaxes(title_text="Scenario")

        # Set y-axes titles
        fig.update_yaxes(title_text="$ \\frac{Cumulative reward}{100}$", secondary_y=False)
        fig.update_yaxes(title_text="Accomplished time steps", secondary_y=True)
        fig.update_layout(**fig_kwargs)

        return  fig
