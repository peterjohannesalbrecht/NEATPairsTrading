"""Module that holds the VisualizationService()-class.

This class holds several methods, that are partly custom,
partly extracted and customized from the neat-python package.
"""
from typing import Any, Optional

import graphviz
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class VisualizationService:
    """Class for visualizing results.

    This class holds methods for plotting several insights
    into the results of a training/testing run.
    """

    def __init__(
        self, portfolio_info: pd.DataFrame, data: pd.DataFrame, env: Any
    ) -> None:
        self.portfolio_info = portfolio_info
        self.data = data
        self.env = env
        sns.set_style('dark')

    def plot_exposure(self,
                      compare: str = 'z_score',
                      wealth: bool = False,
                      date_range: Optional[list] = None) -> None:
        """Visualize exposure alongside z_score.

        Legend is not yet working correctly
        """
        # Prepare data
        info = self.portfolio_info.copy().reset_index()
        viz = info.merge(self.data.copy(), on='Date')[
            ['Date', compare, 'exposure', 'portfolio_value']
        ].set_index('Date')
        viz.index = pd.to_datetime(viz.index)
        if date_range is not None:
            start_date = date_range[0]
            end_date = date_range[1]
            viz = viz[(viz.index>= start_date) & (viz.index <= end_date)]


        # Create a figure and axis for the plot
        fig, ax = plt.subplots(figsize=(10, 6))


        # Plot the time series with color and linestyle based on exposure
        for i in range(len(viz)):
            if viz['exposure'][i] == 0:
                color = 'lightgrey'
                linestyle = ':'
            elif viz['exposure'][i] == -1:
                color = 'red'
                linestyle = '-'
            else:
                color = 'green'
                linestyle = '-'
            ax.plot(
                viz.index[i : i + 2],
                viz[compare][i : i + 2],
                color=color,
                linestyle=linestyle,
                linewidth = 0.8
            )

        # Iterate over the DataFrame and mark entries and exits when exposure changes
        for i in range(1, len(viz)):
            if viz['exposure'][i] != viz['exposure'][i - 1]:
                if viz['exposure'][i] == 1:
                    ax.scatter(
                        viz.index[i],
                        viz[compare][i],
                        color='green',
                        marker='^',
                        label='Entry',
                        s = 10
                    )
                elif viz['exposure'][i] == -1:
                    ax.scatter(
                        viz.index[i],
                        viz[compare][i],
                        color='red',
                        marker='v',
                        label='Exit',
                        s = 10
                    )

        if wealth:
            # Create a secondary axis for the portfolio value line
            ax2 = ax.twinx()

            # Plot the portfolio value as a line on the secondary y-axis
            ax2.plot(
                viz.index,
                viz['portfolio_value'],
                color='#5EB2F2',
                linestyle=(0, (1, 1)),
                label='Portfolio Value',
            )

            # Set labels for the secondary y-axis
            ax2.set_ylabel('Portfolio Value')
            ax2.fill_between(viz.index, viz['portfolio_value'], y2=0,
                             color='#5EB2F2', alpha=0.1)

            # Set the labels for the twin axis
            ax2.set_ylabel('Portfolio value')

        # Set labels and title
        ax.set_xlabel('Date')
        y_label = 'Z-Score' if compare == 'z_score' else 'Spread'
        ax.set_ylabel(y_label)
        title = 'Positions in Long-Short Portfolio alongside ' + y_label
        if wealth:
            title += ' and Portfolio value'
        ax.set_title(title)

        # Create legend with desired entries
        green_line = mlines.Line2D([], [], color='green', label='Long position open')
        red_line = mlines.Line2D([], [], color='red', label='Short position open')
        grey_line = mlines.Line2D(
            [], [], color='lightgrey', label='No open position', linestyle = ':'
        )
        blue_line = mlines.Line2D(
            [], [], color='#5EB2F2', label='Portfolio value', linestyle = ':'
        )
        green_marker = mlines.Line2D(
            [], [], color='green', label='Long position entered',
            linestyle='None', marker='^'
        )
        red_marker = mlines.Line2D(
            [], [], color='red', label='Short position entered',
            linestyle='None',  marker='v'
        )

        handles = [green_line, red_line, green_marker, red_marker, grey_line]
        if wealth:
            handles.append(blue_line)
        plt.legend(handles=handles)

        # Format x-axis ticks
        date_locator = mdates.AutoDateLocator()
        date_formatter = mdates.AutoDateFormatter(date_locator)

        # Apply the locator and formatter to the x-axis
        plt.gca().xaxis.set_major_locator(date_locator)
        plt.gca().xaxis.set_major_formatter(date_formatter)

        # Rotate x-axis labels for better visibility
        plt.xticks(rotation=45)

        # Show the plot
        plt.show()

    def plot_portfolio_insights(self) -> None:
        """Plot several portfolio infos.

        Information includes:
            - Portfolio exposure over time
            - Price of assets over time
            - Portfolio weights over time
        """
        # Create a figure with four subplots
        sns.set_style('dark')
        sns.set_palette(sns.color_palette('Blues'))
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))

        # Subplot 2: exposure
        axs[0, 1].plot(self.portfolio_info['exposure'])
        axs[0, 1].set_title('Exposure')

        # Subplot 3: price_asset_1 and price_asset_2
        axs[1, 0].plot(self.portfolio_info['price_asset_1'])
        axs[1, 0].plot(self.portfolio_info['price_asset_2'])
        axs[1, 0].set_title('Price of Asset 1 and Asset 2')

        # Subplot 4: weight_1 and weight_2 from the 'data' DataFrame
        axs[1, 1].plot(self.data['weight_1'])
        axs[1, 1].plot(self.data['weight_2'])
        axs[1, 1].set_title('Weight of Asset 1 and Asset 2')

        # Rotate x-axis labels in subplots
        for ax in axs.flat:
            ax.tick_params(axis='x', rotation=45)

        # Show the plot
        plt.show()

    def plot_number_assets(self) -> None:
        """Plot the number of assets held over time."""
        # Prepare data
        info = self.portfolio_info.copy().reset_index()
        viz = info.merge(self.data.copy(), on='Date')[
            ['Date', 'asset_1', 'asset_2']
        ].set_index('Date')
        fig, ax1 = plt.subplots()

        # Plot the first column on the left y-axis
        color1 = 'tab:red'
        ax1.set_xlabel('X-axis')
        ax1.set_ylabel('Asset 1', color=color1)
        ax1.plot(viz['asset_1'], color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)

        # Create a second y-axis
        ax2 = ax1.twinx()

        # Plot the second column on the right y-axis
        color2 = 'tab:blue'
        ax2.set_ylabel('Asset 2', color=color2)
        ax2.plot(viz['asset_2'], color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        # Add a title and legend
        plt.title('Plot with Two Different Y Scales')
        lines = [ax1.get_lines()[0], ax2.get_lines()[0]]
        plt.legend(lines, ['Column 1', 'Column 2'])

        # Display the plot
        plt.show()

    def draw_net(
        self,
        config: Any,
        genome: Any,
        view: bool = True,
        filename: str = 'src/Output/visuals/net.svg',
        node_names=None,
        show_disabled: bool = True,
        prune_unused: bool = False,
        node_colors=None,
        fmt: str = 'svg',
    ) -> Any:
        """Receive a genome and draw a neural network with arbitrary topology."""
        # If requested, use a copy of the genome which omits all components
        # that won't affect the output.
        if prune_unused:
            genome = genome.get_pruned_copy(config.genome_config)

        if node_names is None:
            # Get input node names from environemnt
            input_node_names = self.env.input_nodes
            # Get default node names from neat-package
            input_node_names_ = config.genome_config.input_keys
            # Map default to environment names
            node_names = dict(zip(input_node_names_, input_node_names))
            # Do the same for the output nodes
            output_node_names = ['Sell', 'Hold', 'Buy']
            output_node_names_ = config.genome_config.output_keys
            output_dict = dict(zip(output_node_names_, output_node_names))
            # Generate one dict holding it all
            node_names.update(output_dict)

        assert type(node_names) is dict

        if node_colors is None:
            node_colors = {}

        assert type(node_colors) is dict

        node_attrs = {
            'shape': 'circle',
            'fontsize': '9',
            'height': '0.2',
            'width': '0.2',
        }

        dot = graphviz.Digraph(format=fmt, node_attr=node_attrs, graph_attr={'ratio': '0.8'})

        inputs = set()
        for k in config.genome_config.input_keys:
            inputs.add(k)
            name = node_names.get(k, str(k))
            input_attrs = {
                'style': 'filled',
                'shape': 'box',
                'fillcolor': node_colors.get(k, 'lightgray'),
            }
            dot.node(name, _attributes=input_attrs)

        outputs = set()
        for k in config.genome_config.output_keys:
            outputs.add(k)
            name = node_names.get(k, str(k))
            node_attrs = {
                'style': 'filled',
                'fillcolor': node_colors.get(k, 'lightblue'),
            }

            dot.node(name, _attributes=node_attrs)

        used_nodes = set(genome.nodes.keys())
        for n in used_nodes:
            if n in inputs or n in outputs:
                continue

            attrs = {'style': 'filled', 'fillcolor': node_colors.get(n, 'white')}
            dot.node(str(n), _attributes=attrs)

        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                input, output = cg.key
                a = node_names.get(input, str(input))
                b = node_names.get(output, str(output))
                style = 'solid' if cg.enabled else 'dotted'
                color = 'green' if cg.weight > 0 else 'red'
                width = str(0.1 + abs(cg.weight / 5.0))
                dot.edge(
                    a,
                    b,
                    _attributes={'style': style, 'color': color, 'penwidth': width},
                )

        dot.render(filename, view=view)

        return dot

    @staticmethod
    def plot_species(statistics: Any,
                     view: bool = True,
                     filename: str ='src/Output/visuals/speciation.svg') -> None:
        """Visualize speciation throughout evolution."""
        species_sizes = statistics.get_species_sizes()
        num_generations = len(species_sizes)
        curves = np.array(species_sizes).T

        sns.set_style('dark')
        sns.set_palette(sns.color_palette('Paired', 100))


        fig, ax = plt.subplots()
        ax.stackplot(range(num_generations), *curves)

        plt.title('Speciation')
        plt.ylabel('Size per Species')
        plt.xlabel('Generations')
        plt.savefig(filename)

        if view:
            plt.show()
        sns.set()


    @staticmethod
    def plot_stats(statistics: Any,
                   ylog: bool = False,
                   view: bool = True,
                   filename: str = 'src/Output/visuals/avg_fitness.svg'
    ) -> None:
        """Plot the population's average and best fitness."""
        sns.set_style('dark')
        sns.set_palette(sns.color_palette('Blues'))

        generation = range(len(statistics.most_fit_genomes))
        best_fitness = [c.fitness for c in statistics.most_fit_genomes]
        avg_fitness = np.array(statistics.get_fitness_mean())
        stdev_fitness = np.array(statistics.get_fitness_stdev())

        plt.plot(generation, avg_fitness, 'b-', label='average')
        plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label='-1 sd')
        plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label='+1 sd')
        plt.plot(generation, best_fitness, 'r-', label='best')

        plt.title("Population's average and best fitness")
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.grid()
        plt.legend(loc='best')
        if ylog:
            plt.gca().set_yscale('symlog')

        plt.savefig(filename)
        if view:
            plt.show()

    @staticmethod
    def plot_wealth(
        neat: pd.DataFrame, linear: pd.DataFrame, non_linear: pd.DataFrame
        ) -> None:
        """Plot wealth for different provided portfolios.

        Method plots wealth for different PairsTradingPortfolios.
        Can be used for benchmarking purposes.
        """
        wealth = neat.merge(linear, left_index=True, right_index=True)[
            ['portfolio_value_x', 'portfolio_value_y']
        ]
        wealth = wealth.merge(non_linear, left_index=True, right_index=True)[
            ['portfolio_value_x', 'portfolio_value_y', 'portfolio_value']
        ]
        wealth = wealth.rename(
            columns={
                'portfolio_value_x': 'NEAT',
                'portfolio_value_y': 'Static thresholds',
                'portfolio_value': 'DRL',
            }
        )
        wealth.index = pd.to_datetime(wealth.index)
        sns.set_style('dark')
        sns.set_palette(sns.color_palette('Paired'))
        plt.plot(wealth.index, wealth['NEAT'], label='NEAT')
        plt.plot(wealth.index, wealth['Static thresholds'], label='STATIC')
        plt.plot(wealth.index, wealth['DRL'], label='DRL')
        # Format x-axis ticks
        date_locator = mdates.YearLocator()  # YearLocator to display only years
        date_formatter = mdates.DateFormatter('%Y')  # Format to display only the year

        # Apply the locator and formatter to the x-axis
        plt.gca().xaxis.set_major_locator(date_locator)
        plt.gca().xaxis.set_major_formatter(date_formatter)

        # Rotate x-axis labels for better visibility
        plt.xlabel('Date')
        plt.ylabel('Portfolio value')
        plt.title(
         'Portfolio value over the trading period from NEAT and the benchmarks'
        )
        plt.legend()
        plt.show()
        sns.set()
