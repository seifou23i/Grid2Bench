Examples
========
This section presents a code example to show how the package could be used.

Installation/Usage:
*******************

.. code-block:: python

    """
    This examples shows how to load agent data
    """
    import os
    from grid2bench.AgentsAnalytics import AgentsAnalytics

    # define a path to experimented agents results
    input_data_path = os.path.abspath("../data/input")

    # Importing agents results
    agents = AgentsAnalytics(
        data_path=input_data_path,
        agents_names= ["PPO_Agent", "MazeRL_Agent"],
    )

    # visualize the the cumulative rewards
    agents.plot_cumulative_reward(agents_results)

    # visualize the accomplished time stamps
    agents.plot_cumulative_reward(agents_results, alive_time=True)

    # visualize the actions frequency by type
    AgentsAnalytics.plot_actions_freq_by_type(
        agents.agents_data,
        col=2,
        title = "Frequency of actions based on action types"
    )

    # visualize actions frequency by station
    AgentsAnalytics.plot_actions_freq_by_station_pie_chart(
        agents.agents_data,
        col=2,
        title = "Frequency of actions by station"
    )

    # Impact of actions on overloaded lines
    AgentsAnalytics.plot_lines_impact(
        agents.agents_data,
        title = "Overloaded lines",
        yaxis_type = "linear"
    )

    # Impact of actions on disconnected lines
    AgentsAnalytics.plot_lines_impact(
        agents.agents_data,
        title = "Overloaded lines",
        fig_type = "disconnected"
    )

    # Impact of actions on reference topology
    #TODO: should be updated when issue #16 is treated
    AgentsAnalytics.plot_distance_from_initial_topology(
        agents.agents_data,
        title = "Frequency of actions by station",
    )

    # Impact of actions on subsations
    AgentsAnalytics.plot_actions_freq_by_station(
        agents.agents_data,
        title = "Frequency of actions by station",
        yaxis_type = "log"
    )

.. todo::
    1. Treate #TODO tags in example snippets (plot_distance_from_initial_topology)
