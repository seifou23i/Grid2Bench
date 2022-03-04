import os
from rlbenchplot.EpisodesPlot import EpisodesPlot
from tqdm import tqdm
import pandas as pd
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class AgentsAnalysis :
    """

    """

    def __init__(self, data_path, agents_names=[],episodes_names=[]):
        """

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
    def plot_actions_freq_by_station_multi_agent(
            agents_results=[],
            episodes_names=[],
            title="Frequency of actions by station",
            **fig_kwargs):
        """

        :param agents:
        :param episodes_names:
        :param title:
        :param fig_kwargs:
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
    def plot_actions_freq_by_type_multi_agent(
            agents_results=[],
            episodes_names=[],
            title="Frequency of actions by type",
            row=1,
            col=2,
            **fig_kwargs):
        """

        :param agents_results:
        :param episodes_names:
        :param title:
        :param fig_kwargs:
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
    def plot_actions_freq_by_station_pie_chart_multi_agent(
            agents_results=[],
            episodes_names=[],
            title="Frequency of actions by type",
            row=1,
            col=2,
            **fig_kwargs):
        """

        :param agents_results:
        :param episodes_names:
        :param title:
        :param fig_kwargs:
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
    def plot_lines_impact_multi_agent(
            agents_results=[],
            episodes_names=[],
            title="Overloaded Lines by station",
            overload=True,
            **fig_kwargs):
        """

        :param agents:
        :param episodes_names:
        :param title:
        :param fig_kwargs:
        :return:
        """
        k = 0
        agent_names = []
        if not overload:
            k = 1
            title = "Disconnected Lines by station"

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
    def plot_computation_times_multi_agent(
            agents_results=[],
            episodes_names=[],
            title="Action Execution Time",
            **fig_kwargs):
        """

        :param agents:
        :param episodes_names:
        :param title:
        :param fig_kwargs:
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
