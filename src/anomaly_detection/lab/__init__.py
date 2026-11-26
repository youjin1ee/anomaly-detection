"""Tools for a fault injection lab: a compressed-day load generator, a fault injector that
writes ground truth labels, a metric dump and a replay of the dump through the harness.

Everything here uses only the standard library and numpy, except `contender`, which needs
torch and runs in the serving engine's environment.
"""
