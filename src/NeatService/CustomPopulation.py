"""Module that holds the CustomPopulation()-class.

This module holds the CustomPopulation()-class which
is used to enable Parallelization specifically for the project's
use Case. It is just a customized version of the neat-python
packages Population class.
"""
import sqlite3
from typing import Any

import pandas as pd
from neat.population import CompleteExtinctionException, Population
from neat.six_util import iteritems, itervalues

from src.PipelineSettings.PipelineSettings import PipelineSettings


class CustomPopulation(Population):
    """Class that provides customized Population class.

    This class enables parallel genome evaluation for the project's
    custom neat implementation
    It inherits from Population() and is customized
    to work for the project's use case.
    """

    def run(
        self, fitness_function: Any, n: int, pipeline_settings: PipelineSettings
    ) -> Any:
        """Run NEAT's genetic algorithm for at most n generations.  If n
        is None, run until solution is found or extinction occurs.

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.

        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.
        """
        if self.config.no_fitness_termination and (n is None):
            msg = 'Cannot have no generational limit with no fitness termination'
            raise RuntimeError(msg)

        k = 0
        while n is None or k < n:
            k += 1

            self.reporters.start_generation(self.generation)
            pair = pipeline_settings['pair']
            query = f'SELECT * FROM training_pairs WHERE pair = "{pair}" ORDER BY Date'
            with sqlite3.connect('src/Data/pairs_trading.db') as conn:
                data = pd.read_sql(query, conn).drop(columns='pair')

            # Perform data split to avoid overfitting
            data = (
                data.iloc[0:600, :]
                if k < 30
                else data.iloc[200:800, :].reset_index(drop=True)
            )

            # Evaluate all genomes using the user-provided function.
            fitness_function(list(iteritems(self.population)), self.config, data)

            # Gather and report statistics.
            best = None
            for g in itervalues(self.population):
                if best is None or g.fitness > best.fitness:
                    best = g
            self.reporters.post_evaluate(
                self.config, self.population, self.species, best
            )

            # Track the best genome ever seen.
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(
                    g.fitness for g in itervalues(self.population)
                )
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best)
                    break

            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(
                self.config, self.species, self.config.pop_size, self.generation
            )

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(
                        self.config.genome_type,
                        self.config.genome_config,
                        self.config.pop_size,
                    )
                else:
                    raise CompleteExtinctionException

            # Divide the new population into species.
            self.species.speciate(self.config, self.population, self.generation)

            self.reporters.end_generation(self.config, self.population, self.species)

            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(
                self.config, self.generation, self.best_genome
            )

        return self.best_genome
