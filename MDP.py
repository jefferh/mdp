from __future__ import division
from math import *
from collections import defaultdict
import numpy as np
import random
import time

class MDP:
    def __init__(self, mdp_dict):
        """
        Input: a dictionary representation of the MDP; the keys are state-action 
               pairs, and the values are pairs whose first element is the 
               corresponding one-step reward and whose second element is a 
               dictionary where each key is a state and each value is the 
               probability of transitioning to that state

        Attributes:
        m = number of state-action pairs
        n = number of states
        X = list of states
        A = dictionary of lists of available actions; keys are states, values are 
            sets of actions
        r = dictionary of one-step rewards; keys are state-action pairs, values 
            are rewards
        p = dictionary of transition probabilities; keys are state-action pairs, 
            values are dictionaries where each key is a state and the corresponding 
            value is the probability of transitioning to that state
        """
        self.m = len(mdp_dict)
        
        states = set()
        for state_action_pair in mdp_dict.keys():
            states.add(state_action_pair[0])
        self.n = len(states)        
        self.X = list(states)
        
        actions = {}
        for state in states:
            act_set = set()
            for key in mdp_dict:
                if key[0] == state:
                    act_set.add(key[1])
            actions[state] = list(act_set)
        self.A = actions

        rewards = {}
        for state_action_pair in mdp_dict.keys():
            rewards[state_action_pair] = mdp_dict[state_action_pair][0]
        self.r = rewards

        transition_probs = {}
        for state_action_pair in mdp_dict.keys():
            transition_probs[state_action_pair] = mdp_dict[state_action_pair][1]
        self.p = transition_probs

def policy_eval(pi, mdp, discount_factor):
    """ Generate the transition matrix. """
    P_pi = np.fromfunction(np.vectorize(lambda i,j : mdp.p[mdp.X[i],
               pi[mdp.X[i]]][mdp.X[j]]), (mdp.n, mdp.n), dtype=int)
    """ Generate the reward vector. """
    r_pi = np.fromfunction(np.vectorize(lambda i,j : mdp.r[mdp.X[i],
               pi[mdp.X[i]]]), (mdp.n, 1), dtype=int)
    return np.linalg.inv(np.eye(mdp.n) - discount_factor * P_pi).dot(r_pi)

def greedy_action(state, v, mdp, discount_factor):
    state_action_values = [(mdp.r[state, action] +
                            discount_factor*sum([mdp.p[state, action][mdp.X[j]] * v[j]
                                for j in range(len(mdp.X))]))[0]
                            for action in mdp.A[state]]
    return mdp.A[state][np.argmax(state_action_values)]

def greedy_policy(v, mdp, discount_factor):
    # print 'start greedy policy'
    pi = {}
    for state in mdp.X:
        pi[state] = greedy_action(state, v, mdp, discount_factor)
    return pi

def optimality_operator(v, mdp, discount_factor):
    pi = greedy_policy(v, mdp, discount_factor)
    # print 'got greedy policy'
    Tv = np.zeros((mdp.n, 1))
    for i in range(mdp.n):
        Tv[i] = mdp.r[mdp.X[i], pi[mdp.X[i]]] + discount_factor*sum([mdp.p[mdp.X[i], pi[mdp.X[i]]][mdp.X[j]]*v[j] for j in range(mdp.n)])
    return Tv

def policy_operator(v, mdp, discount_factor, pi):
    Tv = np.zeros((mdp.n, 1))
    for i in range(mdp.n):
        Tv[i] = mdp.r[mdp.X[i], pi[mdp.X[i]]] + discount_factor*sum([mdp.p[mdp.X[i], pi[mdp.X[i]]][mdp.X[j]]*v[j] for j in range(mdp.n)])
    return Tv

def policy_iteration(mdp, discount_factor):
    start = time.time()
    """Select an initial policy."""
    pi = {state : random.choice(mdp.A[state]) for state in mdp.X}
    iteration_count = 0
    while True:
        iteration_count += 1
        """Evaluate the current policy."""
        v_pi = policy_eval(pi, mdp, discount_factor)
        """Try to improve the current policy."""
        updated = False
        for state in mdp.X:
            a_star = greedy_action(state, v_pi, mdp, discount_factor)
            if a_star != pi[state]:
                pi[state] = a_star
                updated = True
        if not(updated):
            print 'Number of policy iterations: {}'.format(iteration_count)
            print 'Policy iteration took {} seconds.'.format(time.time() - start)
            return pi

def value_iteration(mdp, discount_factor, epsilon=0.001):
    start = time.time()
    """Initialization"""
    v = np.zeros((mdp.n, 1))
    # print 'initialized value iteration'
    Tv = optimality_operator(v, mdp, discount_factor)
    # print 'did first value iteration'
    """Iterate until epsilon-convergence"""
    iteration_count = 0
    while max(np.absolute(Tv - v))[0] > epsilon*(1 - discount_factor)/discount_factor:
        iteration_count += 1
        # print 'Iteration {}'.format(iteration_count)
        v = Tv
        Tv = optimality_operator(v, mdp, discount_factor)
    print 'Number of value iterations: {}'.format(iteration_count)
    print 'Value iteration took {} seconds.'.format(time.time() - start)
    pi = greedy_policy(v, mdp, discount_factor)
    return pi

def modified_policy_iteration(mdp, discount_factor, num_vis_per_iter, epsilon=0.001):
    start = time.time()
    """Initialization"""
    v = np.zeros((mdp.n, 1))
    pi = greedy_policy(v, mdp, discount_factor)
    Tv = v
    for i in range(num_vis_per_iter):
        Tv = policy_operator(Tv, mdp, discount_factor, pi)
    """Iterate until epsilon-convergence"""
    iteration_count = 0
    while max(np.absolute(Tv - v))[0] > epsilon*(1 - discount_factor)/discount_factor:
        iteration_count += 1
        # print 'Iteration {}'.format(iteration_count)
        v = Tv # value vector from previous iteration
        pi = greedy_policy(v, mdp, discount_factor)
        """Update the value vector"""
        Tv = v
        for i in range(num_vis_per_iter):
            Tv = policy_operator(Tv, mdp, discount_factor, pi)
    print 'Number of modified policy iterations: {}'.format(iteration_count)
    print 'Modified policy iteration took {} seconds.'.format(time.time() - start)
    pi = greedy_policy(v, mdp, discount_factor)
    return pi
