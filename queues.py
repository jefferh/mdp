from __future__ import division
from math import *
from collections import defaultdict
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import time
import MDP

class TwoParallel_WithDet:
    """
        Two parallel queues, controlled in discrete-time via a single server 
        that deteriorates. At most one event (i.e., an arrival, service 
        completion, or change in server state) can occur during each time slot.

        Attributes:
        buffer_size_1 = maximum number of customers (including any in service)
        that can be in queue 1
        buffer_size_2 = maximum number of customers (including any in service)
        that can be in queue 2

        service_list = list where the indices are the server states, and
        each value is a tuple whose first element is the queue 1 service rate 
        for that state, whose second element is the queue 2 service rate for 
        that state, and whose third element is the rate at which the server
        changes state
        
        arrival_prob_1 = arrival rate to queue 1
        arrival_prob_2 = arrival rate to queue 2
        cost_rate_1 = holding cost rate for queue 1
        cost_rate_2 = holding cost rate for queue 2
        maint_cost = fixed cost to initiate maintenance
    """
    def __init__(self, N1, N2, slist, lambda1, lambda2, c1, c2, K):
        self.buffer_size_1 = N1
        self.buffer_size_2 = N2

        Phi = sum([sum([slist[s][0] for s in range(len(slist))]),
                      sum([slist[s][1] for s in range(len(slist))]),
                      sum([slist[s][2] for s in range(len(slist))]),
                      lambda1, lambda2])
        slist_uniformized = [(slist[s][0]/Phi, slist[s][1]/Phi, slist[s][2]/Phi) for s in range(len(slist))]
                 
        self.service_list = slist_uniformized

        self.arrival_prob_1 = lambda1 / Phi
        self.arrival_prob_2 = lambda2 / Phi
        self.cost_rate_1 = c1
        self.cost_rate_2 = c2
        self.maint_cost = K

    def generate_mdp_dict(self):
        mdp_dict = {}
        N = [0, self.buffer_size_1, self.buffer_size_2]
        S = len(self.service_list) # number of server states (including state 0)
        B = S-1
        
        mu1 = [self.service_list[s][0] for s in range(S)]
        mu2 = [self.service_list[s][1] for s in range(S)]
        det = [self.service_list[s][2] for s in range(S)]
        
        arr = [0, self.arrival_prob_1, self.arrival_prob_2]
        c = [0, self.cost_rate_1, self.cost_rate_2]
        K = self.maint_cost
        A = [0, 1, 2, 'm'] # set of all actions; 0 = idle, 1 = serve queue 1, 2 = serve queue 2, 'm' = maintain

        ## Action 0 (idle the server) #########################################################################################
        mdp_dict[((0, 0, 1), 0)] = (-K*det[1],
                                        defaultdict(lambda : 0, {(1, 0, 1) : arr[1],
                                                                     (0, 1, 1) : arr[2],
                                                                     (0, 0, 0) : det[1],
                                                                     (0, 0, 1) : 1 - arr[1] - arr[2] - det[1]}))
        for s in range(2,S):
            mdp_dict[((0, 0, s), 0)] = (0,
                                            defaultdict(lambda : 0, {(1, 0, s) : arr[1],
                                                                         (0, 1, s) : arr[2],
                                                                         (0, 0, s-1) : det[s],
                                                                         (0, 0, s) : 1 - arr[1] - arr[2] - det[s]}))
        # for i in range(N[1]): # transitions when neither queue is full
        #     for j in range(N[2]):
        #         mdp_dict[((i, j, 1), 0)] = (-(c[1]*i + c[2]*j + K*det[1]),
        #                                         defaultdict(lambda : 0, {(i+1, j, 1) : arr[1],
        #                                                                      (i, j+1, 1) : arr[2],
        #                                                                      (i, j, 0) : det[1],
        #                                                                      (i, j, 1) : 1 - arr[1] - arr[2] - det[1]}))
        #         for s in range(2,S):
        #                 mdp_dict[((i, j, s), 0)] = (-(c[1]*i + c[2]*j),
        #                                                 defaultdict(lambda : 0, {(i+1, j, s) : arr[1],
        #                                                                              (i, j+1, s) : arr[2],
        #                                                                              (i, j, s-1) : det[s],
        #                                                                              (i, j, s) : 1 - arr[1] - arr[2] - det[s]}))
        # for j in range(N[2]): # transitions when queue 1 is full
        #     mdp_dict[((N[1], j, 1), 0)] = (-(c[1]*N[1] + c[2]*j + K*det[1]),
        #                                        defaultdict(lambda : 0, {(N[1], j+1, 1) : arr[2],
        #                                                                     (N[1], j, 0) : det[1],
        #                                                                     (N[1], j, 1) : 1 - arr[2] - det[1]}))
        #     for s in range(2,S):
        #         mdp_dict[((N[1], j, s), 0)] = (-(c[1]*N[1] + c[2]*j),
        #                                            defaultdict(lambda : 0, {(N[1], j+1, s) : arr[2],
        #                                                                         (N[1], j, s-1) : det[s],
        #                                                                         (N[1], j, s) : 1 - arr[2] - det[s]}))
        # for i in range(N[1]): # transitions when queue 2 is full
        #     mdp_dict[((i, N[2], 1), 0)] = (-(c[1]*i + c[2]*N[2] + K*det[1]),
        #                                        defaultdict(lambda : 0, {(i+1, N[2], 1) : arr[1],
        #                                                                     (i, N[2], 0) : det[1],
        #                                                                     (i, N[2], 1) : 1 - arr[1] - det[1]}))
        #     for s in range(2,S):
        #         mdp_dict[((i, N[2], s), 0)] = (-(c[1]*i + c[2]*N[2]),
        #                                            defaultdict(lambda : 0, {(i+1, N[2], s) : arr[1],
        #                                                                         (i, N[2], s-1) : det[s],
        #                                                                         (i, N[2], s) : 1 - arr[1] - det[s]}))
        # # transitions when both queues are full
        # mdp_dict[((N[1], N[2], 1), 0)] = (-(c[1]*N[1] + c[2]*N[2] + K*det[1]),
        #                                       defaultdict(lambda : 0, {(N[1], N[2], 0) : det[1],
        #                                                                    (N[1], N[2], 1) : 1 - det[1]}))
        # for s in range(2,S):
        #     mdp_dict[((N[1], N[2], s), 0)] = (-(c[1]*N[1] + c[2]*N[2]),
        #                                           defaultdict(lambda : 0, {(N[1], N[2], s-1) : det[s],
        #                                                                        (N[1], N[2], s) : 1 - det[s]}))
        ## Action 1 (serve queue 1) ###############################################################################################
        for i in range(1,N[1]): # transitions when neither queue is full
            for j in range(N[2]):
                mdp_dict[((i, j, 1), 1)] = (-(c[1]*i + c[2]*j + K*det[1]),
                                                defaultdict(lambda : 0, {(i+1, j, 1) : arr[1],
                                                                             (i, j+1, 1) : arr[2],
                                                                             (i-1, j, 1) : mu1[1],
                                                                             (i, j, 0) : det[1],
                                                                             (i, j, 1) : 1 - arr[1] - arr[2] - mu1[1] - det[1]}))
                for s in range(2,S):
                    mdp_dict[((i, j, s), 1)] = (-(c[1]*i + c[2]*j),
                                                    defaultdict(lambda : 0, {(i+1, j, s) : arr[1],
                                                                                 (i, j+1, s) : arr[2],
                                                                                 (i-1, j, s) : mu1[s],
                                                                                 (i, j, s-1) : det[s],
                                                                                 (i, j, s): 1 - arr[1] - arr[2] - mu1[s] - det[s]}))
        for j in range(N[2]): # transitions when queue 1 is full
            mdp_dict[((N[1], j, 1), 1)] = (-(c[1]*N[1] + c[2]*j + K*det[1]),
                                               defaultdict(lambda : 0, {(N[1], j+1, 1) : arr[2],
                                                                            (N[1]-1, j, 1) : mu1[1],
                                                                            (N[1], j, 0) : det[1],
                                                                            (N[1], j, 1) : 1 - arr[2] - mu1[1] - det[1]}))
            for s in range(2,S):
                mdp_dict[((N[1], j, s), 1)] = (-(c[1]*i + c[2]*j),
                                                   defaultdict(lambda : 0, {(N[1], j+1, s) : arr[2],
                                                                                (N[1]-1, j, s) : mu1[s],
                                                                                (i, j, s-1) : det[s],
                                                                                (i, j, s) : 1 - arr[2] - mu1[s] - det[s]}))
        for i in range(1,N[1]): # transitions when queue 2 is full
            mdp_dict[((i, N[2], 1), 1)] = (-(c[1]*i + c[2]*N[2] + K*det[1]),
                                               defaultdict(lambda : 0, {(i+1, N[2], 1) : arr[1],
                                                                            (i-1, N[2], 1) : mu1[1],
                                                                            (i, N[2], 0) : det[1],
                                                                            (i, N[2], 1) : 1 - arr[1] - mu1[1] - det[1]}))
            for s in range(2,S):
                mdp_dict[((i, N[2], s), 1)] = (-(c[1]*i + c[2]*j),
                                                   defaultdict(lambda : 0, {(i+1, N[2], s) : arr[1],
                                                                                (i-1, N[2], s) : mu1[s],
                                                                                (i, j, s-1) : det[s],
                                                                                (i, j, s) : 1 - arr[1] - mu1[s] - det[s]}))
        # transitions when both queues are full
        mdp_dict[((N[1], N[2], 1), 1)] = (-(c[1]*N[1] + c[2]*N[2] + K*det[1]),
                                              defaultdict(lambda : 0, {(N[1]-1, N[2], 1) : mu1[1],
                                                                           (N[1], N[2], 0) : det[1],
                                                                           (N[1], N[2], 1) : 1 - mu1[1] - det[1]}))
        for s in range(2,S):
            mdp_dict[((N[1], N[2], s), 1)] = (-(c[1]*N[1] + c[2]*N[2]),
                                                defaultdict(lambda : 0, {(N[1]-1, N[2], s) : mu1[s],
                                                                             (N[1], N[2], s-1) : det[s],
                                                                             (N[1], N[2], s) : 1 - mu1[s] - det[s]}))
        ## Action 2 (serve queue 2)
        for i in range(N[1]): # transitions when neither queue is full
            for j in range(1,N[2]):
                mdp_dict[((i, j, 1), 2)] = (-(c[1]*i + c[2]*j + K*det[1]),
                                                defaultdict(lambda : 0, {(i+1, j, 1) : arr[1],
                                                                             (i, j+1, 1) : arr[2],
                                                                             (i, j-1, 1) : mu2[1],
                                                                             (i, j, 0) : det[1],
                                                                             (i, j, 1) : 1 - arr[1] - arr[2] - mu2[1] - det[1]}))
                for s in range(2,S):
                    mdp_dict[((i, j, s), 2)] = (-(c[1]*i + c[2]*j),
                                                    defaultdict(lambda : 0, {(i+1, j, s) : arr[1],
                                                                                 (i, j+1, s) : arr[2],
                                                                                 (i, j-1, s) : mu2[s],
                                                                                 (i, j, s-1) : det[s],
                                                                                 (i, j, s) : 1 - arr[1] - arr[2] - mu2[s] - det[s]}))
        for j in range(1, N[2]): # transitions when queue 1 is full
            mdp_dict[((N[1], j, 1), 2)] = (-(c[1]*N[1] + c[2]*j + K*det[1]),
                                               defaultdict(lambda : 0, {(N[1], j+1, 1) : arr[2],
                                                                            (N[1], j-1, 1) : mu2[1],
                                                                            (N[1], j, 0) : det[1],
                                                                            (N[1], j, 1) : 1 - arr[2] - mu2[1] - det[1]}))
            for s in range(2,S):
                mdp_dict[((N[1], j, s), 2)] = (-(c[1]*i + c[2]*j),
                                                   defaultdict(lambda : 0, {(N[1], j+1, s) : arr[2],
                                                                                (N[1], j-1, s) : mu2[s],
                                                                                (N[1], j, s-1) : det[s],
                                                                                (N[1], j, s) : 1 - arr[2] - mu2[s] - det[s]}))
        for i in range(N[1]): # transitions when queue 2 is full
            mdp_dict[((i, N[2], 1), 2)] = (-(c[1]*i + c[2]*N[2] + K*det[1]),
                                               defaultdict(lambda : 0, {(i+1, N[2], 1) : arr[1],
                                                                            (i, N[2]-1, 1) : mu2[1],
                                                                            (i, N[2], 0) : det[1],
                                                                            (i, N[2], 1) : 1 - arr[1] - mu2[1] - det[1]}))
            for s in range(2,S):
                mdp_dict[((i, N[2], s), 2)] = (-(c[1]*i + c[2]*N[2]),
                                                   defaultdict(lambda : 0, {(i+1, N[2], s) : arr[1],
                                                                                (i, N[2]-1, s) : mu2[s],
                                                                                (i, N[2], s-1) : det[s],
                                                                                (i, N[2], s) : 1 - arr[1] - mu2[s] - det[s]}))
        # transitions when both queues are full
        mdp_dict[((N[1], N[2], 1), 2)] = (-(c[1]*N[1] + c[2]*N[2] + K*det[1]),
                                              defaultdict(lambda : 0, {(N[1], N[2]-1, 1) : mu2[1],
                                                                           (N[1], N[2], 0) : det[1],
                                                                           (N[1], N[2], 1) : 1 - mu2[1] - det[1]}))
        for s in range(2,S):
            mdp_dict[((N[1], N[2], s), 2)] = (-(c[1]*N[1] + c[2]*N[2]),
                                                  defaultdict(lambda : 0, {(N[1], N[2]-1, s) : mu2[s],
                                                                               (N[1], N[2], s-1) : det[s],
                                                                               (N[1], N[2], s) : 1 - mu2[s] - det[s]}))
        ## Action m (perform maintenance)
        for s in range(1, S): # when maintenance is initiated preventively
            for i in range(N[1]): # when neither queue is full
                for j in range(N[2]):
                    mdp_dict[((i, j, s), 'm')] = (-(K + c[1]*i + c[2]*j),
                                                      defaultdict(lambda : 0, {(i+1, j, 0) : arr[1],
                                                                                   (i, j+1, 0) : arr[2],
                                                                                   (i, j, B) : det[0],
                                                                                   (i, j, 0) : 1 - arr[1] - arr[2] - det[0]}))
            for j in range(N[2]): # when queue 1 is full
                mdp_dict[((N[1], j, s), 'm')] = (-(K + c[1]*N[1] + c[2]*j),
                                                       defaultdict(lambda : 0, {(N[1], j+1, 0) : arr[2],
                                                                                    (N[1], j, B) : det[0],
                                                                                    (N[1], j, 0) : 1 - arr[2] - det[0]}))
            for i in range(N[1]): # when queue 2 is full
                mdp_dict[((i, N[2], s), 'm')] = (-(K + c[1]*i + c[2]*N[2]),
                                                     defaultdict(lambda : 0, {(i+1, N[2], 0) : arr[1],
                                                                                  (i, N[2], B) : det[0],
                                                                                  (i, N[2], 0) : 1 - arr[1] - det[0]}))
            # when both queues are full
            mdp_dict[((N[1], N[2], s), 'm')] = (-(K + c[1]*N[1] + c[2]*N[2]),
                                                    defaultdict(lambda : 0, {(N[1], N[2], B) : det[0],
                                                                                 (N[1], N[2], 0) : 1 - det[0]}))
        # maintenance is the only action avaialble when the server state is 0
        for i in range(N[1]): # transitions when neither queue is full
            for j in range(N[2]):
                mdp_dict[((i, j, 0), 'm')] = (-(c[1]*i + c[2]*j),
                                                  defaultdict(lambda : 0, {(i+1, j, 0) : arr[1],
                                                                               (i, j+1, 0) : arr[2],
                                                                               (i, j, B) : det[0],
                                                                               (i, j, 0) : 1 - arr[1] - arr[2] - det[0]}))
        for j in range(N[2]): # transitions when queue 1 is full
            mdp_dict[((N[1], j, 0), 'm')] = (-(c[1]*N[1] + c[2]*j),
                                                 defaultdict(lambda : 0, {(N[1], j+1, 0) : arr[2],
                                                                              (N[1], j, B) : det[0],
                                                                              (N[1], j, 0) : 1 - arr[2] - det[0]}))
        for i in range(N[1]): # transitions when queue 2 is full
            mdp_dict[((i, N[2], 0), 'm')] = (-(c[1]*i + c[2]*N[2]),
                                                 defaultdict(lambda : 0, {(i+1, N[2], 0) : arr[1],
                                                                              (i, N[2], B) : det[0],
                                                                              (i, N[2], 0) : 1 - arr[1] - det[0]}))
        # transitions when both queues are full
        mdp_dict[((N[1], N[2], 0), 'm')] = (-(c[1]*N[1] + c[2]*N[2]),
                                                defaultdict(lambda : 0, {(N[1], N[2], B) : det[0],
                                                                             (N[1], N[2], 0) : 1 - det[0]}))
        return mdp_dict
                                                                
    def plot_switch_surface(self, pi):        
        N = [0, self.buffer_size_1, self.buffer_size_2]
        S = len(self.service_list) # number of server states
        
        fig = pyplot.figure()
        ax = Axes3D(fig)
        
        i_idle = []
        j_idle = []
        s_idle = []

        i_q1 = []
        j_q1 = []
        s_q1 = []

        i_q2 = []
        j_q2 = []
        s_q2 = []

        i_m = []
        j_m = []
        s_m = []
        
        for i in range(N[1]+1):
            for j in range(N[2]+1):
                for s in range(S):
                    if pi[(i, j, s)] == 0: # state where pi idles
                        i_idle.append(i)
                        j_idle.append(j)
                        s_idle.append(s)
                    elif pi[(i, j, s)] == 1: # state where pi serves queue 1
                        i_q1.append(i)
                        j_q1.append(j)
                        s_q1.append(s)
                    elif pi[(i, j, s)] == 2: # state where pi serves queue 2
                        i_q2.append(i)
                        j_q2.append(j)
                        s_q2.append(s)
                    elif pi[(i, j, s)] == 'm': # state where pi maintains
                        i_m.append(i)
                        j_m.append(j)
                        s_m.append(s)
        idle = ax.scatter(i_idle, j_idle, s_idle, c='red', marker='o')
        q1 = ax.scatter(i_q1, j_q1, s_q1, c='blue', marker='s')
        q2 = ax.scatter(i_q2, j_q2, s_q2, c='orange', marker='s')
        m = ax.scatter(i_m, j_m, s_m, c='green', marker='^')

        ax.legend((idle, q1, q2, m),
                      ("Idle", "Serve Queue 1", "Serve Queue 2", "Maintain"),
                      scatterpoints=1)

        ax.set_xticks(range(N[1]+1))
        ax.set_yticks(range(N[2]+1))
        ax.set_zticks(range(S))

        ax.set_xlabel("Number of Class 1 Jobs")
        ax.set_ylabel("Number of Class 2 Jobs")
        ax.set_zlabel("Server State")
        
        pyplot.show()
        
class TwoParallel:
    """
        Two parallel queues, controlled in discrete-time via a single server. 
        At most one event (i.e., an arrival or service completion) can occur 
        during each time slot.

        Attributes:
        buffer_size_1 = maximum number of customers (including any in service) 
        that can be in queue 1
        buffer_size_2 = maximum number of customers (including any in service)
        that can be in queue 2
        service_prob_1 = probability of service completion if queue 1 is non-
        empty and the server is assigned to it
        service_prob_2 = probability of service completion if queue 2 is non-
        empty and the server is assigned to it
        arrival_prob_1 = probability that there's an arrival to queue 1
        arrival_prob_2 = probability that there's an arrival to queue 2
        cost_rate_1 = holding cost rate for queue 1
        cost_rate_2 = holding cost rate for queue 2
    """
    def __init__(self, N1, N2, mu1, mu2, lambda1, lambda2, c1, c2):
        self.buffer_size_1 = N1
        self.buffer_size_2 = N2

        Phi = mu1 + mu2 + lambda1 + lambda2
        
        self.service_prob_1 = mu1 / Phi
        self.service_prob_2 = mu2 / Phi
        self.arrival_prob_1 = lambda1 / Phi
        self.arrival_prob_2 = lambda2 / Phi
        self.cost_rate_1 = c1
        self.cost_rate_2 = c2

    def generate_mdp_dict(self):
        mdp_dict = {}
        N = [0, self.buffer_size_1, self.buffer_size_2]
        mu = [0, self.service_prob_1, self.service_prob_2] # mu[0] = 0 is the service rate when the server idles
        arr = [0, self.arrival_prob_1, self.arrival_prob_2]
        c = [0, self.cost_rate_1, self.cost_rate_2]
        A = [0, 1, 2] # set of all actions; 0 = idle, 1 = serve queue 1, 2 = serve queue 2

        # Action 0 (idle the server) ############################################################################
        for i in range(N[1]): # transitions when neither queue is full
            for j in range(N[2]):
                mdp_dict[((i, j), 0)] = (-(c[1]*i + c[2]*j),
                                             defaultdict(lambda : 0, {(i+1, j) : arr[1],
                                                                          (i, j+1) : arr[2],
                                                                          (i, j) : 1 - arr[1] - arr[2]}))
        for j in range(N[2]): # transitions when queue 1 is full
            mdp_dict[((N[1], j), 0)] = (-(c[1]*N[1] + c[2]*j),
                                             defaultdict(lambda : 0, {(N[1], j+1) : arr[2],
                                                                          (N[1], j) : 1 - arr[2]}))
        for i in range(N[1]): # transitions when queue 2 is full
            mdp_dict[((i, N[2]), 0)] = (-(c[1]*i + c[2]*N[2]),
                                             defaultdict(lambda : 0, {(i+1, N[2]) : arr[1],
                                                                          (i, N[2]) : 1 - arr[1]}))
        # transitions when both queues are full
        mdp_dict[((N[1], N[2]), 0)] = (-(c[1]*N[1] + c[2]*N[2]),
                                           defaultdict(lambda : 0, {(N[1], N[2]) : 1}))
        # Action 1 (serve queue 1) ############################################################################
        for i in range(N[1]): # transitions when neither queue is full
            for j in range(N[2]):
                if i==0:
                    mdp_dict[((i, j), 1)] = (-(c[1]*i + c[2]*j),
                                                 defaultdict(lambda : 0, {(i+1, j) : arr[1],
                                                                              (i, j+1) : arr[2],
                                                                              (i, j) : 1 - arr[1] - arr[2]}))
                else:
                    mdp_dict[((i, j), 1)] = (-(c[1]*i + c[2]*j),
                                                 defaultdict(lambda : 0, {(i+1, j) : arr[1],
                                                                              (i, j+1) : arr[2],
                                                                              (i-1, j) : mu[1],
                                                                              (i, j) : 1 - arr[1] - arr[2] - mu[1]}))
        for j in range(N[2]): # transitions when queue 1 is full
            mdp_dict[((N[1], j), 1)] = (-(c[1]*N[1] + c[2]*j),
                                            defaultdict(lambda : 0, {(N[1], j+1) : arr[2],
                                                                         (N[1]-1, j) : mu[1],
                                                                         (i, j) : 1 - arr[2] - mu[1]}))
        for i in range(N[1]): # transitions when queue 2 is full
            if i == 0:
                mdp_dict[((i, N[2]), 1)] = (-(c[1]*i + c[2]*N[2]),
                                                defaultdict(lambda : 0, {(i+1, N[2]) : arr[1],
                                                                             (i, N[2]) : 1 - arr[1]}))
            else:
                mdp_dict[((i, N[2]), 1)] = (-(c[1]*i + c[2]*N[2]),
                                                defaultdict(lambda : 0, {(i+1, N[2]) : arr[1],
                                                                            (i-1, N[2]) : mu[1],
                                                                            (i, N[2]) : 1 - arr[1] - mu[1]}))
        # transitions when both queues are full
        mdp_dict[((N[1], N[2]), 1)] = (-(c[1]*N[1] + c[2]*N[2]),
                                           defaultdict(lambda : 0, {(N[1]-1, N[2]) : mu[1],
                                                                        (N[1], N[2]) : 1 - mu[1]}))
        # Action 2 (serve queue 2) ############################################################################
        for i in range(N[1]): # transitions when neither queue is full
            for j in range(N[2]):
                if j==0:
                    mdp_dict[((i, j), 2)] = (-(c[1]*i + c[2]*j),
                                                 defaultdict(lambda : 0, {(i+1, j) : arr[1],
                                                                              (i, j+1) : arr[2],
                                                                              (i, j) : 1 - arr[1] - arr[2]}))
                else:
                    mdp_dict[((i, j), 2)] = (-(c[1]*i + c[2]*j),
                                                 defaultdict(lambda : 0, {(i+1, j) : arr[1],
                                                                              (i, j+1) : arr[2],
                                                                              (i, j-1) : mu[2],
                                                                              (i, j) : 1 - arr[1] - arr[2] - mu[2]}))
        for j in range(N[2]): # transitions when queue 1 is full
            if j == 0:
                mdp_dict[((N[1], j), 2)] = (-(c[1]*N[1] + c[2]*j),
                                                defaultdict(lambda : 0, {(N[1], j+1) : arr[2],
                                                                             (N[1], j) : 1 - arr[2]}))
            else:
                mdp_dict[((N[1], j), 2)] = (-(c[1]*N[1] + c[2]*j),
                                                defaultdict(lambda : 0, {(N[1], j+1) : arr[2],
                                                                             (N[1], j-1) : mu[2],
                                                                             (N[1], j) : 1 - arr[2] - mu[2]}))
        for i in range(N[1]): # transitions when queue 2 is full
            mdp_dict[((i, N[2]), 2)] = (-(c[1]*i + c[2]*N[2]),
                                            defaultdict(lambda : 0, {(i+1, N[2]) : arr[1],
                                                                         (i, N[2]-1) : mu[2],
                                                                         (i, N[2]) : 1 - arr[1] - mu[2]}))
        # transitions when both queues are full
        mdp_dict[((N[1], N[2]), 2)] = (-(c[1]*N[1] + c[2]*N[2]),
                                           defaultdict(lambda : 0, {(N[1], N[2]-1) : mu[2],
                                                                        (N[1], N[2]) : 1 - mu[2]}))
        return mdp_dict
                                                                             

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
