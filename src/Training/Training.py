"""Module for the Training class.

This module holds the Training()-class that is responsible
for training the NEAT-algorithm
"""
import multiprocessing

import neat

from src.NeatService.CustomParallelEvaluator import CustomParallelEvaluator
from src.NeatService.CustomPopulation import CustomPopulation
from src.NeatService.NeatService import eval_genome_feed_forward, eval_genome_recurrent
from src.PipelineSettings.PipelineSettings import PipelineSettings


class Training:
    """Handles the training process of neat."""

    def __init__(self, pipeline_settings: PipelineSettings) -> None:
        self.pipeline_settings = pipeline_settings
        self.config = None
        self.eval_func = None

    def setup_training(self) -> None:
        """Set up the training process."""
        # Setup config using neat specific config file
        input_dir = self.pipeline_settings['input_dir']
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            f'{input_dir}neat_config.txt',
        )

        # Overwrite defaults by specifications in pipeline_settings
        self.config.genome_config.feed_forward = self.pipeline_settings['feed_forward']
        self.config.pop_size = self.pipeline_settings['pop_size']

        # specify eval_function
        self.eval_func = (
            eval_genome_feed_forward
            if self.pipeline_settings['feed_forward']
            else eval_genome_recurrent
        )

    def run_training(self) -> tuple:
        """Run the NEAT-Algorithm."""
        self.setup_training()
        # Create the population and run the NEAT algorithm
        neat.Population = CustomPopulation(self.config)
        pop = neat.Population
        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)

        # Create parallel evaluator
        neat.CustomParallelEvaluator = CustomParallelEvaluator(
            multiprocessing.cpu_count() * 2, self.eval_func
        )
        pe = neat.CustomParallelEvaluator
        net = pop.run(
            pe.evaluate,
            n=self.pipeline_settings['num_generations'],
            pipeline_settings=self.pipeline_settings,
        )

        return net, stats
