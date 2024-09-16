# -*- coding: utf-8 -*-
# Copyright (C) 2001-2020
# Author: Biyang Ma <biyangma@stu.xmu.edu.cn> <mabiyang001@hotmail.com>
# builtins
import pysmile
import pysmile_license
import numpy as np
import re
#import datetime
#import math
import graphviz as G
from graphviz import Source
from graphviz import Digraph
import pydotplus
import collections

# self package
from Node import Node
class PolicyTree(object):
    def __init__(self,policy_tree_id=None,action_list=None,observation_list=None, policy_pathes=None,prior=None, policy_dict = None):
        # tree attributes
        self.id = []
        self.name = ''
        self.horizon = 3
        self.node_string_head = 'mod'
        self.belief = []
        self.num_path = 0
        self.len_path = 0
        self.show = False
        self.merge_top = True


        # for plotting
        self.nodelist = []
        self.node_labels = dict()
        self.edgelist = []
        self.edge_labels = dict()
        self.pos = dict()
        # policy tree
        self.nodes = list()

        self.roots_map = dict()


        self.root = Node('0','root',0)
        self.nodes.append(self.root)

        self.leaf = Node('inf','leaf',-1)
        self.nodes.append(self.leaf)

        self.policy_dict = dict()
        self.node_dict = dict()


        #dataset
        self.policy_mat = np.zeros(1)
        self.adjacent_mat = np.zeros(1)

        # initialize
        self.set_name(policy_tree_id)
        self.set_policytree_id(policy_tree_id)
        self.action_list = action_list
        self.observation_list = observation_list
        self.set_policy_tree()
        self.set_policy_pathes(policy_pathes)
        self.set_prior_belief(prior)
        self.set_policy_dict(policy_dict)

        #self.initialize()

    def initialize(self):
        self.get_horizon()
        self.get_len_path()
        self.get_num_path()
        #self.gen_policygraph()
        #self.build_tree()


    def __display__(self):
        print(self.name)
    def print_summary(self):
        print(self.policytree)
    ### set attributes
    def set_name(self, name):
        self.name = name
    def set_policytree_id(self, id):
        self.id = id
    def set_action_list(self, action_list):
        if ( action_list is None or len(action_list) ==0):
            self.action_list = list()
            print('errrrrrrrrrrrrrrrrrr')
        else:
            self.action_list =  action_list
    def set_observation_list(self, observation_list):
        if (observation_list is None or len(observation_list) ==0):
            self.observation_list = list()
        else:
            self.observation_list = observation_list
    def set_policy_tree(self):
        self.policytree = G.Digraph(comment='T'+self.name)
        return self.policytree
    def set_policy_pathes(self, policy_pathes):
        if (policy_pathes is None or len(policy_pathes) ==0):
            self.policy_pathes =list()
        else:
            self.policy_pathes = policy_pathes
    def set_prior_belief(self, prior):
        if prior is None or len(prior)==0 :
            self.belief = list()
        else:
            self.belief = prior
    def set_horizon(self,horizon):
        self.horizon = horizon
    def set_policy_mat(self, mat):
        self.policy_mat = mat
    def set_node_string_head(self, node_string_head):
        self.node_string_head = node_string_head + '_' +self.id
    def set_policy_dict(self,policy_dict):
        self.policy_dict = dict()
        if  (policy_dict is None or len(policy_dict) ==0) :
            self.policy_dict = dict()
        else:
            self.policy_dict = policy_dict
    def set_root(self,root):
        if root is None:
            self.root = Node('0','root',0)
        else:
            self.root = root
        self.nodes.append(self.root)
    def set_leaf(self,leaf):
        if leaf is None:
            self.leaf = Node('inf', 'leaf', -1)
        else:
            self.leaf = leaf
        self.nodes.append(self.leaf)

    # get attributes
    def get_name(self):
        return self.name
    def get_policytree_id(self):
        return self.id
    def get_action_list(self):
        return self.action_list
    def get_observation_list(self):
        return self.observation_list
    def get_policy_tree(self):
        return self.policytree
    def get_policy_pathes(self):
        return self.policy_pathes
    def get_num_path(self):
        if not (self.policy_pathes is None or len(self.policy_pathes) == 0):
            self.num_path = len(self.policy_pathes)
        if not (self.policy_dict is None or len(self.policy_dict) ==0):
            pathes = self.policy_dict.get(1)
            self.num_path = len(pathes)
        return self.num_path
    def get_len_path(self):
        if not (self.policy_pathes is None or len(self.policy_pathes) == 0):
            self.len_path = len(self.policy_pathes[0])
        if not (self.policy_dict is None or len(self.policy_dict) == 0):
            pathes = self.policy_dict.get(1)
            self.len_path = len(pathes[0])
        return self.len_path
    def get_horizon(self):
        if not (self.policy_pathes is None or len(self.policy_pathes) ==0):
            self.horizon = int((len(self.policy_pathes[0])+1)/2)
        if not (self.policy_dict is None or len(self.policy_dict) ==0):
            for key in self.policy_dict.keys():
                key_s = key
                break
            pathes = self.policy_dict.get(key_s)
            self.horizon = int((len(pathes[0]) + 1) / 2)
        return self.horizon
    def get_belief_str(self):
        return self.belief
    def get_root(self):
        return self.root
    def get_nodes(self):
        return self.nodes
    def get_leaf(self):
        return self.leaf

    def get_nodelist(self):
        return  self.nodelist
    def get_node_labels(self):
        return self.node_labels
    def get_edgelist(self):
        return self.edgelist
    def get_edge_labels(self):
        return self.edge_labels
    def get_node_string_head(self):
        return self.node_string_head
    def get_policy_dict(self ):
        return self.policy_dict

    #generate method
    def gen_policy_tree(self,key,pathes):
        Horizon = self.get_horizon()
        for pathi in pathes:
            act_values = [pathi[(Horizon - hi) * 2] for hi in range(Horizon, 0, -1)]
            ob_values = [pathi[(Horizon - hi) * 2 + 1] for hi in range(Horizon, 1, -1)]
            parent_index = self.nodes.index(self.root)
            for hi in range(Horizon, 0, -1):
                #print(self.action_list)
                act_str = self.action_list[int(act_values[Horizon - hi])]
                parent = self.nodes.pop(parent_index)
                if hi < Horizon:
                    ob_str = self.observation_list[int(ob_values[Horizon - 1 - hi])]
                else:
                    ob_str = self.node_string_head + str(key)
                if not parent.arcs_to_children_contains(ob_str):
                    Num_nodes = len(self.nodes) + 2
                    node_id = self.node_string_head + '_' + str(key) + '_' + str(Num_nodes)
                    node = Node(node_id, act_str, hi)
                    node.insert_parent(parent)
                    node.set_arc_from_parent(parent, ob_str)
                    parent.insert_child(node)
                    parent.set_arc_to_child(node, ob_str)
                else:
                    node = parent.arcs_to_children.get(ob_str)
                    node_index = self.nodes.index(node)
                    node = self.nodes.pop(node_index)
                if hi == 1:
                    leaf_index = self.nodes.index(self.leaf)
                    self.leaf = self.nodes.pop(leaf_index)
                    ob_str = 'inf'
                    self.leaf.insert_parent(node)
                    self.leaf.set_arc_from_parent(node, ob_str)
                    node.insert_child(self.leaf)
                    node.set_arc_to_child(self.leaf, ob_str)
                    self.nodes.append(self.leaf)
                self.nodes.append(parent)
                self.nodes.append(node)
                parent_index = self.nodes.index(node)
        if self.show:
            print('-------------')
            for ns in self.nodes:
                ns.display()
            print('-------------')
    def gen_policy_trees(self):
        # create nodes in first time, then merge them
        for key in self.policy_dict.keys():
            pathes = self.policy_dict.get(key)
            self.gen_policy_tree( key, pathes)
        self.self_merge()
        self.gen_node_and_edge()
        self.build_tree()
    def gen_policy_trees_memorysaved(self):
        # not preserve every tree
        # create a public and final merged tree, i.e. the self is the public one
        # for each mod not create a new policy tree,
        # add the nodes and edges in the public one, then self merge
        for key in self.policy_dict.keys():
            pathes = self.policy_dict.get(key)
            self.gen_policy_tree(key, pathes)
            self.self_merge()
        self.policy_tree_roots_map()
        self.gen_node_and_edge()
        self.build_tree()

    def gen_policy_trees_incremental(self,pathes):
        # for new generate policy tree: pathes
        # not preserve every tree
        # create a public and final merged tree, i.e. the self is the public one
        # for each mod not create a new policy tree,
        # add the nodes and edges in the public one, then self merge
        key = np.max(list(self.policy_dict.keys()))+1
        self.policy_dict[key] = pathes
        self.gen_policy_tree(key, pathes)
        self.self_merge()
        self.policy_tree_roots_map()
        self.gen_node_and_edge()
        self.set_policy_tree()
        self.build_tree()
    def merge_other(self,root_of_next_policy_tree,leaf_of_next_policy_tree):
        # transfer the root node and leaf node of other policy treee
        # to build the tree
        print()
    def remove_node(self, node):
        if node not in self.nodes:
            return False
        else:
            arcs_from_parents = node.get_arcs_from_parents()
            for arc in arcs_from_parents.keys():
                parents = arcs_from_parents.get(arc)
                for pa in parents:
                    pa.remove_arc_to_children(node, arc)
                    pa.remove_child(node)
            arcs_to_children = node.get_arcs_to_children()
            for arc in arcs_to_children.keys():
                child = arcs_to_children.get(arc)
                child.remove_parent(node)
                child.remove_arc_from_parent(node, arc)
            self.nodes.pop(self.nodes.index(node))
            del node
            return True
    def remove_dominated_tree(self, modi):
        # after we have evaluate every policy tree's reward, we want to remove the dominated one
        # go through the root via arc modi to node with horizon max of it
        # remove it, check and remove the nodes not root and not leaf  without parents
        node = self.get_root()
        arc_to_mod_i = self.get_node_string_head() + str(modi)
        node = node.get_arcs_to_children().get(arc_to_mod_i)
        self.remove_node( node)
        for ns in self.nodes:
            if len(ns.get_parents())==0 and not ns.is_leaf() and not ns.is_root:
                self.remove_node(ns)
    def gen_node_dict(self):
        self.node_dict = dict()
        for ns in self.nodes:
            if ns.is_leaf() or ns.is_root():
                continue
            # ns.display()
            node_id = ns.get_id()
            level = ns.get_level()
            if not self.node_dict.__contains__(level):
                self.node_dict[level] = [node_id]
            else:
                nodes =   self.node_dict.get(level)
                nodes.append(node_id)
                self.node_dict[level] = nodes
    def get_node_dict(self):
        self.gen_node_dict()
        return self.node_dict
    def gen_node_and_edge(self):
        #self.gen_node_dict()
        self.nodelist = list()
        self.node_labels = dict()
        self.edgelist = list()
        self.edge_labels = dict()
        for ns in self.nodes:
            if ns.is_leaf() or ns.is_root():
                continue
            #ns.display()
            node_id = ns.get_id()
            if not self.node_labels.__contains__(node_id):
                self.nodelist.append(node_id)
                self.node_labels[node_id] = ns.get_label()
            for arc in ns.arcs_to_children.keys():
                child = ns.arcs_to_children.get(arc)
                if child.is_leaf():
                   continue
                child_id = child.get_id()
                if  not self.edge_labels.__contains__((node_id, child_id)):
                    self.edgelist.append((node_id, child_id))
                    self.edge_labels[(node_id, child_id)] = arc
                else:
                    old_arcs = self.edge_labels.get((node_id, child_id))
                    self.edge_labels[(node_id, child_id)] = old_arcs + '|'+ arc
    def build_tree(self):
        self.set_policy_tree()
        nodelabels  = self.get_node_labels()
        for ni in self.get_nodelist():
            self.policytree.node(ni, nodelabels.get(ni))
        for ei in self.edgelist:
            edge_label_list = self.edge_labels.get(ei)
            edge_label_list = edge_label_list.split('|')

            if len(edge_label_list) == 1:
                self.edge_labels[ei] = edge_label_list[0]
                continue
            if len(edge_label_list) == len(self.observation_list):
                self.edge_labels[ei] = '*'
                while self.edgelist.count(ei) > 1:
                    self.edgelist.remove(ei)
                continue

        for ei in self.get_edgelist():
            #self.policytree.edge(ei[0], ei[1], label = self.edge_labels[ei])
            self.policytree.edge(ei[0], ei[1], label=self.edge_labels.get(ei))
    def save_policytree(self,filename):
        # print(filename +self.name)
        # self.policytree.save(filename +self.name + '.gv')
        # self.policytree.save(filename +self.name + '.dot')
        # self.policytree.render(filename + self.name + '.gv')
        self.policytree.render(filename=filename + self.name, format='pdf')
    def self_merge(self):
        # from leaf of trees, merging from bottom-up
        waiting_list = list()
        waiting_list.append(self.leaf)
        deleted_list = dict()
        while not len(waiting_list)==0:
            node = waiting_list.pop(0)
            if not self.merge_top and node.get_level()>self.horizon-1:#
                continue
            # node.display()
            arcs = [key for key in node.get_arcs_from_parents().keys()]
            for arc in arcs:
                if not node.get_arcs_from_parents().__contains__(arc):
                    print(arc)
                    node.display_dict(node.get_arcs_from_parents())
                    continue
                parents = node.get_arcs_from_parents().get(arc)
                parents_id = [pa.get_id() for pa in parents]
                parents_dict = dict()
                for pa in parents:
                    parents_dict[pa.get_id()] =  pa
                for i in  range(0,len(parents_id)):
                    pa_id = parents_id[i]
                    if not parents_dict.__contains__(pa_id) or deleted_list.__contains__(pa_id):
                        continue
                    pa =  parents_dict.get(pa_id)
                    # pa.display()
                    for j in range(0,len(parents_id)):
                        pb_id = parents_id[j]
                        if pa_id == pb_id :
                            continue
                        if not parents_dict.__contains__(pb_id) or deleted_list.__contains__(pb_id):
                            continue
                        pb = parents_dict.get(pb_id)
                        # pb.display()
                        flag = self.merge_eq_node(pa,pb)
                        if flag:
                            self.nodes.pop(self.nodes.index(pb))
                            deleted_list[pb_id] = flag
            for pa in node.get_parents():
                if not pa.is_root():
                    waiting_list.append(pa)
    def policy_tree_roots_map(self):
        # print('=======================')
        self.roots_map = dict()
        arcs_to_children =self.root.get_arcs_to_children()
        for arc in arcs_to_children.keys():
            child = arcs_to_children.get(arc)
            child_id = child.get_id()
            key = arc[3:len(arc)]
            self.roots_map[key] = child_id
            # print(key,child_id)
        # print('=======================')
    def merge_eq_node(self,node1,node):
        if node1.is_with_eq_label(node) and node1.is_goto_eq_children(node):
            # can merge
            # print('=========can merge=============')
            arcs_from_parents = node.get_arcs_from_parents()
            for arc in arcs_from_parents.keys():
                parents = arcs_from_parents.get(arc)
                for pa in parents:
                    # pa.display()
                    pa.set_arc_to_child(node1,arc)
                    pa.remove_child(node)
                    pa.insert_child(node1)
                    # pa.display()
                    node1.set_arc_from_parent(pa,arc)
                    node1.insert_parent(pa)
                    # node1.display()
            arcs_to_children = node.get_arcs_to_children()
            for arc in arcs_to_children.keys():
                child = arcs_to_children.get(arc)
                # child.display()
                child.remove_parent(node)
                child.remove_arc_from_parent(node,arc)
                # child.display()
            return True
            # print('=========can merge=============')
        else:
            return False


# pathes = [np.array([2., 0., 2., 0., 2.]),
#           np.array([2., 0., 2., 1., 2.]),
#           np.array([2., 0., 2., 2., 1.]),
#           np.array([2., 0., 2., 3., 2.]),
#           np.array([2., 0., 2., 4., 2.]),
#           np.array([2., 0., 2., 5., 2.]),
#           np.array([2., 1., 2., 0., 2.]),
#           np.array([2., 1., 2., 1., 2.]),
#           np.array([2., 1., 2., 2., 1.]),
#           np.array([2., 1., 2., 3., 2.]),
#           np.array([2., 1., 2., 4., 2.]),
#           np.array([2., 1., 2., 5., 2.]),
#           np.array([2., 2., 2., 0., 2.]),
#           np.array([2., 2., 2., 1., 2.]),
#           np.array([2., 2., 2., 2., 1.]),
#           np.array([2., 2., 2., 3., 2.]),
#           np.array([2., 2., 2., 4., 2.]),
#           np.array([2., 2., 2., 5., 2.]),
#           np.array([2., 3., 2., 0., 2.]),
#           np.array([2., 3., 2., 1., 2.]),
#           np.array([2., 3., 2., 2., 2.]),
#           np.array([2., 3., 2., 3., 2.]),
#           np.array([2., 3., 2., 4., 2.]),
#           np.array([2., 3., 2., 5., 0.]),
#           np.array([2., 4., 2., 0., 2.]),
#           np.array([2., 4., 2., 1., 2.]),
#           np.array([2., 4., 2., 2., 2.]),
#           np.array([2., 4., 2., 3., 2.]),
#           np.array([2., 4., 2., 4., 2.]),
#           np.array([2., 4., 2., 5., 0.]),
#           np.array([2., 5., 2., 0., 2.]),
#           np.array([2., 5., 2., 1., 2.]),
#           np.array([2., 5., 2., 2., 2.]),
#           np.array([2., 5., 2., 3., 2.]),
#           np.array([2., 5., 2., 4., 2.]),
#           np.array([2., 5., 2., 5., 0.])]
# policy_dict = dict()
# policy_dict[1] = pathes
# policy_tree_id = str(1)
# action_list = ['OL', 'OR', 'L']
# observation_list =['GLCL', 'GLCR', 'GLS', 'GRCL', 'GRCR', 'GRS']
# pt = PolicyTree(str(1)+''+str(2),action_list,observation_list)
# pt.set_policy_dict(policy_dict)
# pt.gen_policy_trees_memorysaved()
# pt.save_policytree('test-output/')
#
# policy_pathes = [np.array([2., 0., 2., 0., 2.]), np.array([2., 0., 2., 1., 1.]), np.array([2., 1., 2., 0., 2.]), np.array([2., 1., 2., 1., 0.])]
# policy_tree_id = str(1)
# action_list = ['OL','OR','L']
# observation_list = ['GL','GR']
# prior = [0.3,0.7]
# policy_dict = dict()
# policy_dict[1] = policy_pathes
#
# pt = PolicyTree(str(1)+''+str(2),action_list,observation_list)
# pt.set_policy_dict(policy_dict)
# pt.gen_policy_trees_memorysaved()
# pt.save_policytree('test-output/2')


# policy_pathes2 = [[2,0,2,0,1],[2,0,2,1,1],[2,1,1,0,1],[2,1,1,1,0]]
# policy_tree_id2 = str(2)
# policy_pathes = [[2,0,2,0,1],[2,0,2,1,2],[2,1,1,0,1],[2,1,1,1,1]]
# policy_tree_id = str(1)
# action_list = ['OL','OR','L']
# observation_list = ['GL','GR']
# prior = [0.3,0.7]
# policy_dict = dict()
# policy_dict[1] = policy_pathes
# policy_dict[2] = policy_pathes2
#
#
# pt = PolicyTree(str(1)+''+str(2),action_list,observation_list)
# pt.set_policy_dict(policy_dict)
# # pt.gen_policy_trees()
# pt.gen_policy_trees_memorysaved()
# # pt.gen_policy_trees()
# # pt.gen_policy_trees_incremental(policy_pathes2)
# pt.save_policytree('test-output/')
'''

policy_pathes2 = [[2,0,2,0,1],[2,0,2,1,2],[2,1,1,0,2],[2,1,1,1,0]]
policy_tree_id2 = str(2)
policy_pathes = [[2,0,2,0,1],[2,0,2,1,2],[2,1,1,0,1],[2,1,1,1,1]]
policy_tree_id = str(1)
action_list = ['OL','OR','L']
observation_list = ['GL','GR']
prior = [0.3,0.7]
policy_dict = dict()
policy_dict[1] = policy_pathes
policy_dict[2] = policy_pathes2

pt = PolicyTree(str(1)+''+str(2),action_list,observation_list)
pt.set_policy_dict(policy_dict)
pt.gen_policy_trees()
#pt.self_merge()
pt.self_merge()
pt.gen_node_and_edge()
pt.build_tree()
pt.save_policytree('test-output/')

'''
