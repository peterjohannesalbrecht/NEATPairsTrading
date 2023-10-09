"""Service Module for NEAT.

This module provides Service functions for the NEAT-algorithm.
"""
from typing import Any

import neat
import numpy as np
import pandas as pd

from src.PairsTrading.NeatEnvironment import NeatEnvironment


def eval_genome_feed_forward(genome: Any, config: Any, data: pd.DataFrame) -> float:
    """Evaluate the provided genomes's fitness.

    Will be run in parallel on indidivudal processes.
    Computes the fitness of one genome on the provided
    training data. To be used if recurrent nodes are disabled.
    """
    env_neat = NeatEnvironment(data, 1000000)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = 0.0
    obs = env_neat.reset()
    done = False
    while not done:
        obs = obs.flatten()
        action_probabilities = net.activate(obs)
        action = np.argmax(action_probabilities) - 1
        obs, reward, done, __ = env_neat.step(action)
        fitness += reward
    return fitness


def eval_genome_recurrent(genome: Any, config: Any, data: pd.DataFrame) -> float:
    """Evaluate the provided genomes's fitness.

    Will be run in parallel on indidivudal processes.
    Computes the fitness of one genome on the provided
    training data. To be used if recurrent nodes are enabled.
    """
    env_neat = NeatEnvironment(data, 1000000)
    net = neat.nn.RecurrentNetwork.create(genome, config)
    fitness = 0.0
    obs = env_neat.reset()
    done = False
    while not done:
        obs = obs.flatten()
        action_probabilities = net.activate(obs)
        action = np.argmax(action_probabilities) - 1
        obs, reward, done, __ = env_neat.step(action)
        fitness += reward
    return fitness
