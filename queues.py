from __future__ import division
from math import *
from collections import defaultdict
import numpy as np
import random
import time
import MDP

class CRW:
    """
        Single-server queue with a single customer class, modeled as a discrete
        -time controlled random walk. At most one event (i.e., an arrival or 
        service completion) can occur during each time slot.

        Attributes:
        buffer_size = maximum number of customers (including any in service) that 
        can be in the system
        service_probs = list of possible service probabilities
        arrival_prob = probability that there's an arrival
        cost_func = function whose input is (x, q) and whose output is the cost 
                    incurred when the current state is x and the service 
                    probability q is used
    """
    def __init__(self, N, A, p, g):
        self.buffer_size = N
        self.service_probs = A
        self.arrival_prob = p
        self.cost_func = g

    def generate_mdp_dict(self):
        mdp_dict = {}
        p = self.arrival_prob
        for a in self.service_probs:
            mdp_dict[(0, a)] = (-self.cost_func(0, a),
                                    defaultdict(lambda : 0, {0 : 1-p, 1 : p}))
            for x in range(1, self.buffer_size):
                mdp_dict[(x, a)] = (-self.cost_func(x, a),
                                        defaultdict(lambda : 0, {x-1 : a, x : 1-p-a, x+1 : p}))
            mdp_dict[(self.buffer_size, a)] = (-self.cost_func(self.buffer_size, a), defaultdict(lambda : 0, {self.buffer_size-1 : a, self.buffer_size : 1-a}))
        return mdp_dict

    def compute_optimal_policy(self, discount_factor):
        mdp_dict = self.generate_mdp_dict()
        mdp = MDP.MDP(mdp_dict)
        pi = MDP.policy_iteration(mdp, discount_factor)
        return pi

    def simulate(self, state, service_prob):
        """Simulate the queueing model for one step.

           state = current state
           service_prob = probability of a service completion 
        """
        u = np.random.uniform()
        if state == 0:
            if u < self.arrival_prob:
                return (self.cost_func(state, service_prob), state+1)
            else:
                return (self.cost_func(state, service_prob), state)
        elif state > 0 and state < self.buffer_size-1:
            if u < self.arrival_prob:
                return (self.cost_func(state, service_prob), state+1)
            elif u < self.arrival_prob + service_prob:
                return (self.cost_func(state, service_prob), state-1)
            else:
                return (self.cost_func(state, service_prob), state)
        elif state == self.buffer_size:
            if u < service_prob:
                return (self.cost_func(state, service_prob), state-1)
            else:
                return (self.cost_func(state, service_prob), state)

    def simulate_test(self, x, q, n):
        """Check that the simulator is providing transitions at the
           right frequencies.
        """
        n_down = 0
        n_stay = 0
        n_up = 0
        for i in range(n):
            (r, y) = self.simulate(x, q)
            if y == x-1:
                n_down += 1
            elif y == x:
                n_stay += 1
            elif y == x+1:
                n_up += 1
        return (n_down/n, n_stay/n, n_up/n)

    def run_random_policy(self, init_state, num_steps):
        """Run the policy that selects actions uniformly at random, starting 
           from a given initial state.

           init_state = initial state
           num_steps = number of steps to run the policy
        """
        sample_transitions = {}
        for q in self.service_probs:
            sample_transitions[q] = []
        current_state = init_state
        for i in range(num_steps):
            service_prob = random.choice(self.service_probs)
            (cost, next_state) = self.simulate(current_state, service_prob)
            """Assume that the RL algorithm will maximize rewards."""
            sample_transitions[service_prob].append((current_state,
                                                         -cost, next_state))
            current_state = next_state
        return sample_transitions
