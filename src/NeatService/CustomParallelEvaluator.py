"""Module that holds the CustomParallelEvaluater()-class.

This module holds the CustomParallelEvaluater()-class which
is used to enable Parallelization specifically for the thesis's
use Case.
"""
from typing import Any

import pandas as pd
from neat.parallel import ParallelEvaluator


class CustomParallelEvaluator(ParallelEvaluator):
    """Class for handling parallel genome evaluation for neat.

    This class enables parallel genome evaluation for neat.
    It inherits from ParallelEvaluator() and is customized
    to work for the thesis's use case. It is used to speed
    up the training process.
    """

    def evaluate(self, genomes: Any, config: Any, data: pd.DataFrame) -> None:
        jobs = []
        for _ignored_genome_id, genome in genomes:
            jobs.append(
                self.pool.apply_async(self.eval_function, (genome, config, data))
            )

        # assign the fitness back to each genome
        for job, (_ignored_genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)
