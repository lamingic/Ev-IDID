# -*- coding: utf-8 -*-
# Copyright (C) 2001-2020
# Author: Biyang Ma <biyangma@stu.xmu.edu.cn> <mabiyang001@hotmail.com>
# builtins
import numpy as np
import pysmile
import re
from tqdm import tqdm
# self package
from PolicyTree import PolicyTree
from namespace import Name
import namespace as nsp
import tkinter as tk
class DBN(object):
    def __init__(self,filepath,name,step,scr_message =None):
        self.filepath = filepath
        self.name = name
        self.pnames = Name()
        self.type = name
        self.net = pysmile.Network()
        self.net.__copy__()
        self.horizon = ''
        self.num_ss = ''
        self.num_os = ''
        self.num_as = ''
        self.action_list = list()
        self.observation_list = list()
        self.evidences = list()
        self.result = dict()
        self.read(step)
        self.expa_policy_tree = PolicyTree('The Policy Tree of EXPANSION @' + self.type)
        for key in self.pnames.Result:
            if key.__contains__('policy_tree') :
                self.result[key] = PolicyTree('The Policy Tree of'+  self.type,self.action_list,self.observation_list)
                # self.result[key].save_policytree(self.pnames.Save_filepath)
            else:
                self.result[key] = dict()
            if not scr_message is None:
                self.scr_message = scr_message
            else:
                self.scr_message = None
            # method

    def print(self, string):
        if not self.scr_message is None:
            self.scr_message.insert(tk.END, string + '\n')
            self.scr_message.see(tk.END)
            self.scr_message.update()
        else:
            print(string)

    # other
    def __copy__(self,dbn):
        self.filepath = self.dbn.filepath
        self.name = self.dbn.name
        self.pnames = self.dbn.pnames
        self.type = self.dbn.name
        self.net = pysmile.Network()
        self.net.__copy__(self.dbn.net)
        self.evidences = [e for e in self.dbn.evidences]
        self.result = self.dbn.result
        self.update_parameters()
    def copy_result(self,dbn):
        for key in self.pnames.Result:
            if key.__contains__('policy_tree'):
                self.result[key] = PolicyTree('The Policy Tree of' + self.type, self.action_list, self.observation_list)
            else:
                for keyi in dbn.result[key].keys():
                    self.result[key][keyi] = dbn.result[key][keyi]
        self.result['policy_tree'].set_policy_dict(self.result['policy_dict'])
    def write(self, step):
        file = self.filepath + self.pnames.Prefixs[self.type][step] + self.name + self.pnames.Postfix
        # print(file)
        self.net.write_file(file)
    def read(self,step):
        file = self.filepath + self.pnames.Prefixs[self.type][step] + self.name + self.pnames.Postfix
        self.net.read_file(file);  # read j's DID or IDID prior model
        self.update_parameters()
    def display(self,node):
        print('----------' + node+'-----------------')
        mat = np.array(self.net.get_node_definition(node))
        node_states = len(self.net.get_outcome_ids(node))
        Total_cls = int(len(mat) / node_states)
        mat = np.transpose(np.reshape(mat, [Total_cls, node_states]))
        print(mat)
        return mat
    def eucliDist(self, A, B):
        return np.sqrt(sum(np.power((A - B), 2)))
    def mat_mult(self, A, B):
        mat = np.zeros([len(A), len(B)])
        for rw in range(0, len(A)):
            mat[rw, :] = A[rw] * B
        return mat
    def norm_prob(self, sub_mat, prob_s):
        size_mat = np.shape(sub_mat)
        copy = int(size_mat[1] / len(prob_s))
        prob_sb = list()
        if copy > 1:
            for si in range(0, len(prob_s)):
                for ci in range(0, copy):
                    prob_sb.append(prob_s[si])
            prob_s = np.array(prob_sb)
        prob_s = np.matmul(sub_mat, prob_s)
        prob_s = prob_s / np.sum(prob_s)
        return prob_s
    def update_parameters(self):
        node_list = self.net.get_all_node_ids()
        node_list_count = [re.sub(r'\d+', '', e) for e in node_list]
        node_list_e = set([re.sub(r'\d+', '', e) for e in node_list])
        Steps_list = [node_list_count.count(ni) for ni in node_list_e]
        self.horizon = np.max(Steps_list)
        self.num_ss = self.net.get_outcome_count("S" + str(self.horizon))  # number of state value
        self.num_os = self.net.get_outcome_count("O" + str(self.horizon - 1))  # number of observation value
        self.num_as = self.net.get_outcome_count("D" + str(self.horizon))  # number of action value
        self.action_list = self.net.get_outcome_ids("D" + str(self.horizon))
        self.observation_list = self.net.get_outcome_ids("O" + str(self.horizon - 1))
    # extend the network
    def extend(self,horizon_max,step):
        if self.horizon < horizon_max:
            Steps_gap = horizon_max - self.horizon
            node_list = self.net.get_all_node_ids()
            node_list_e_count = [re.sub(r'\d+', '', e) for e in node_list]
            node_list_e = set([re.sub(r'\d+', '', e) for e in node_list])
            Steps_dict = dict([(ni, node_list_e_count.count(ni)) for ni in node_list_e])
            # change the index of node from Horizon-1 to Horizon_Max-[Horizon_Max-Horizon]
            for hi in range(self.horizon, 0, -1):
                for ni in node_list_e:
                    if hi <= Steps_dict.get(ni):
                        node_id = ni + str(hi)
                        node_id_index = node_list.index(node_id)
                        node_id_n = ni + str(hi + Steps_gap)
                        self.net.set_node_id(node_id_index, node_id_n)
                        self.net.set_node_name(node_id_n, node_id_n)
            # add node from [Horizon-1] to 1
            node_list = self.net.get_all_node_ids()  # the order of add the arc
            for sti in range(Steps_gap, 0, -1):
                for ni in node_list_e:
                    node_id = ni + str(sti)
                    node_id_copy = ni + str(sti + 1)  # parent
                    node_id_copy2 = ni + str(sti + 2)  # grandparent
                    node_type = pysmile.NodeType(self.net.get_node_type(node_id_copy))
                    self.net.add_node(node_type, node_id)
                    self.net.set_node_name(node_id, node_id)

                    position = self.net.get_node_position(node_id_copy)
                    position_dif = self.net.get_node_position(node_id_copy2)
                    position[0] = position[0] * 2 - position_dif[0]
                    self.net.set_node_position(node_id, position)

                    bgcolor = self.net.get_node_bg_color(node_id_copy)
                    self.net.set_node_bg_color(node_id, bgcolor)
                    if node_type != pysmile.NodeType.UTILITY:
                        outcomes = self.net.get_outcome_ids(node_id_copy)
                        initial_outcome_count = self.net.get_outcome_count(node_id)
                        for i in range(0, initial_outcome_count):
                            self.net.set_outcome_id(node_id, i, outcomes[i])
                        for i in range(initial_outcome_count, len(outcomes)):
                            self.net.add_outcome(node_id, outcomes[i])
            # add arc
            for sti in range(Steps_gap, 0, -1):
                for ni in node_list_e:
                    node_id = ni + str(sti)
                    node_id_copy = ni + str(sti + 1)  # parent
                    parent = self.net.get_parent_ids(node_id_copy)
                    if len(parent) > 0:
                        parent_step = [str(int(re.sub(r'[A-Z]+', '', e)) - 1) for e in parent]
                        parent_id = [re.sub(r'\d+', '', e) for e in parent]
                        parent_n = [parent_id[i] + parent_step[i] for i in range(0, len(parent))]
                        for pi in parent_n:
                            self.net.add_arc(pi, node_id)
            # set definition
            for sti in range(Steps_gap, 0, -1):
                for ni in node_list_e:
                    node_id = ni + str(sti)
                    node_type = pysmile.NodeType(self.net.get_node_type(node_id))
                    # if node_type != pysmile.NodeType.UTILITY and node_type != pysmile.NodeType.TRUTH_TABLE:
                    if node_type != pysmile.NodeType.TRUTH_TABLE:
                        node_id_copy = ni + str(sti + 1)  # parent
                        mat = self.net.get_node_definition(node_id_copy)
                        self.net.set_node_definition(node_id, mat)
        self.update_parameters()
        self.write(step)
    def generate_evidence(self,step):
        net = pysmile.Network()
        file = self.filepath + self.pnames.Prefixs[self.type][step-1] + self.name+ self.pnames.Postfix
        # print(file)
        net.read_file(file);  # prior model
        node_list = net.get_all_node_ids()
        node_list_e = set([re.sub(r'\d+', '', e) for e in node_list])
        node_list_e.remove("O")
        node_list_e.remove("D")
        for ni in node_list:
            Flag = False
            for e in node_list_e:
                if ni.startswith(e):
                    Flag = True
            if Flag == True:  # delete node S, U ...
                net.delete_node(ni)
                continue
            if ni.startswith("O"):  # Change the definition matrix of O to uniform
                mat = net.get_node_definition(ni)
                prob = 1 / net.get_outcome_count(ni)
                mat = [prob for x in range(0, len(mat))]
                net.set_node_definition(ni, mat)

            if ni.startswith("D"):  # Change the node type of D to deterministic
                child_ids = net.get_child_ids(ni)  # delete the arc between Ds
                if len(child_ids) != 0:
                    for ci in child_ids:
                        if ci.startswith("D"):
                            net.delete_arc(ni, ci)
                parent_ids = net.get_parent_ids(ni)
                net.set_node_type(ni, int(pysmile.NodeType.TRUTH_TABLE))
                net.add_arc(parent_ids[0], ni)
        net.delete_node("O" + str(self.horizon))  # delete node 1st O
        file = self.filepath + self.pnames.Prefixs[self.type][step] + self.name + self.pnames.Postfix
        net.write_file(file)
        self.evidences = net.get_all_node_ids()
    # update
    def normalize(self,_d, copy=True):
        # d is a (n x dimension) np array
        d = _d if not copy else np.copy(_d)
        d /= np.sum(d)
        return d
    def get_reward(self,weight_off = None,modi = None):  # only for one policy tree
        # self.MODPrefix = self.pnames.MODPrefix.get(self.type).get(self.pnames.Steps.get(self.type)[step])
        # print(self.MODPrefix)
        policy_dict = self.expa_policy_tree.get_policy_dict()
        for key in policy_dict.keys():
            pathes = policy_dict.get(key)
            break
        # need to modify
        if modi is None:
            mat_rows = self.pnames.MAX
            mat = np.random.rand(mat_rows, self.num_ss)
            [self.normalize(mat[i, :], copy=False) for i in range(0, self.pnames.P_MAX)]
        else:
            prior_belief_mat = self.result.get('prior_belief')
            # print(prior_belief_mat.keys())
            modi = modi % (len(prior_belief_mat))
            prob_prior_s = prior_belief_mat[modi]
            mat_rows = 1
            mat = np.random.rand(mat_rows, self.num_ss)
            mat[0, :] = prob_prior_s

        reward_ptree = list()
        policy_path_weights = dict()
        # print('get reward')
        for i in range(0,mat_rows):
            rewards = list()
            # set the prior distribution of S
            prior_belief = mat[i, :]
            ei = "S" + str(self.horizon)
            # self.net.set_node_definition(ei, prior_belief)
            self.net.set_virtual_evidence(ei, prior_belief)
            e_mod = self.MODPrefix + str(self.horizon)
            ei_values = self.net.get_outcome_ids(e_mod)
            self.net.set_evidence(e_mod, ei_values[0])
            self.net.update_beliefs()
            # get utility
            reward = self.get_path_reward()
            # clear evidences
            self.net.clear_all_evidence()
            path_weight = list()
            if weight_off is None:
                for pathi in range(0, len(pathes)):
                    path = pathes[pathi]
                    # get prob
                    prob_path = self.get_path_prob(prior_belief, path)
                    path_weight.append(prob_path)
                policy_path_weights[i] = path_weight
            reward_ptree.append(np.sum(np.array(reward)))
        # reward_ptree_mean = np.mean(np.array(reward_ptree))
        # reward_ptree_mean_abs = [np.abs(e - reward_ptree_mean) for e in reward_ptree]
        # choose_index = np.argmin(np.array(reward_ptree_mean_abs))
        choose_index = np.argmax(np.array(reward_ptree))
        if weight_off is None:
            ptree_path_weight = policy_path_weights.get(choose_index)
        if weight_off:
            ptree_path_weight = [1/len(pathes) for i in range(0,len(pathes))]
        prior_belief = mat[choose_index, :]
        return np.max(np.array(reward_ptree)), ptree_path_weight, prior_belief
    #OK
    def get_reward_bk(self):# only for one policy tree
        # self.MODPrefix = self.pnames.MODPrefix.get(self.type).get(self.pnames.Steps.get(self.type)[step])
        # print(self.MODPrefix)
        policy_dict = self.expa_policy_tree.get_policy_dict()
        for key in policy_dict.keys():
            pathes = policy_dict.get(key)
            break
        mat = np.random.rand(self.pnames.MAX, self.num_ss)
        [self.normalize(mat[i, :], copy=False) for i in range(0, self.pnames.P_MAX)]
        reward_ptree = list()
        policy_path_weights = dict()
        # print('get reward')
        for i in range(0, self.pnames.MAX):
            rewards = list()
            # set the prior distribution of S
            prior_belief = mat[i, :]
            ei = "S" + str(self.horizon)
            self.net.set_node_definition(ei, prior_belief)
            e_mod = self.MODPrefix + str(self.horizon)
            ei_values = self.net.get_outcome_ids(e_mod)
            self.net.set_evidence(e_mod, ei_values[0])
            self.net.update_beliefs()
            # enter the evidence
            reward = list()
            path_weight = list()
            for pathi in range(0, len(pathes)):
                path = self.partial_clear_evidences(self.evidences, pathi, pathes)
                # set evidences
                self.enter_evidences(self.evidences, path)
                # get utility
                reward_path = self.get_path_reward()
                # get prob
                prob_path = self.get_path_prob(prior_belief,path)
                reward.append(np.sum(reward_path) * prob_path)
                path_weight.append(prob_path)
                # clear evidences
                [self.net.clear_evidence(ei) for ei in self.evidences]
            policy_path_weights[i]= path_weight
            reward_ptree.append(np.sum(np.array(reward)))
        reward_ptree_mean = np.mean(np.array(reward_ptree))
        reward_ptree_mean_abs = [np.abs(e - reward_ptree_mean) for e in reward_ptree]
        choose_index = np.argmin(np.array(reward_ptree_mean_abs))
        choose_index = np.argmax(np.array(reward_ptree))
        ptree_path_weight = policy_path_weights.get(choose_index)
        prior_belief = mat[choose_index, :]
        return np.max(np.array(reward_ptree)),ptree_path_weight,prior_belief
    def partial_clear_evidences(self,evidences, pathi,pathes):
        path = pathes[pathi]
        if pathi!=0:
            delta_path = path - pathes[pathi - 1]
            for ei_index in range(0, len(evidences)):
                if  delta_path[ei_index] == 0:
                    continue
                ei = evidences[ei_index]
                self.net.clear_evidence(ei)
        return path
    def enter_evidences(self,evidences,path):
        for ei_index in range(0, len(evidences)):
            ei = evidences[ei_index]
            self.net.set_evidence(ei, int(path[ei_index]))
        self.net.update_beliefs()
    def gen_pathes(self):
        Obs_id_matrix,Total_Num = self.gen_Obs_id_matrix()
        pathes = list()
        for pathi in range(0, Total_Num):
            path = np.zeros(len(self.evidences)) - 1
            path_obs = Obs_id_matrix[pathi, :]
            len_obs = len(path_obs)
            action_size = len(self.action_list)
            horizon = self.horizon - 1
            for obi in range(len_obs - 1, -1, -1):
                path[2 * (len_obs - 1 - obi) + 1] = path_obs[obi]
            if pathi == 0:
                for ei in range(0, horizon+1):
                    path[2 * ei] = np.random.randint(0,action_size,1)
            else:
                path[0] = pathes[0][0]
                pathj = pathes[len(pathes) - 1]
                for ei in range(0, horizon):# 3steps horizon==2
                    obi = 2 * (ei) + 2# 2 4
                    if np.sum(np.abs(np.array(path[0:obi]) - np.array(pathj[0:obi]))) == 0:
                        path[obi] = pathj[obi]# do the same action with the same ob
                    else:
                        path[obi] = np.random.randint(0,action_size,1)
            pathes.append(path)
        return pathes
    # solver main
    # def exact_solver(self):

    def exact_solver(self):
        num_mod = len(list(self.result.get('prior_belief').keys()))
        Obs_id_matrix, Total_Num = self.gen_Obs_id_matrix()
        for modi in range(0,num_mod):
            self.print('> @ sovling mod:' + str(modi + 1))
            p_mat, prob_prior_s = self.set_prior_S(modi,self.result.get('prior_belief'))

            # enter the evidence
            pathes = list()
            reward = list()
            path_weight = list()
            bar = tqdm(total=int(Total_Num))
            pprob_o = np.array(self.net.get_node_value("O" +str(self.horizon)))
            for pathi in range(0,Total_Num):
                path = self.preset_path(Obs_id_matrix,pathi,pathes,self.evidences)
                # set evidences
                path = self.enter_evidences_s(self.evidences, path,p_mat,prob_prior_s,pprob_o)
                # get utility
                reward_path = self.get_path_reward()
                # get prob
                prob_path = self.get_path_prob( prob_prior_s,path)
                reward.append(np.sum(reward_path) * prob_path)
                path_weight.append(prob_path)
                pathes.append(path)
                bar.update(1)
            bar.close()
            [self.net.clear_evidence(ei) for ei in self.evidences]
            self.result.get('policy_dict')[modi] = pathes
            print(pathes)
            self.result.get('reward')[modi] = np.sum(np.array(reward))
            self.result.get('policy_path_weight')[modi] = path_weight
        self.result['policy_tree'] = PolicyTree('The merged Policy Tree of ' + self.type)
        self.result.get('policy_tree').set_action_list(self.action_list)
        self.result.get('policy_tree').set_observation_list(self.observation_list)
        self.result.get('policy_tree').set_policy_dict(self.result.get('policy_dict'))
        # self.result.get('policy_tree').save_policytree(self.pnames.Save_filepath)
    # solver method
    def gen_Obs_id_matrix(self):
        horizon = self.horizon
        Obs_list = [self.num_os for i in range(0, horizon - 1)]
        Total_Num = int(np.product(Obs_list))
        Obs_id_matrix = np.zeros([Total_Num, horizon - 1])
        mod = np.zeros(horizon - 1)
        for j in range(horizon - 1):
            mod[j] = np.product(Obs_list[0:j + 1]) / Obs_list[j]
        for j in range(horizon - 1):
            k = 0
            clc = 0
            for ai in range(Total_Num):
                # print(mod[j],clc)
                if clc == mod[j]:
                    k = k + 1
                    clc = 0
                    if k == Obs_list[j]:
                        k = 0
                Obs_id_matrix[ai, j] = k
                clc = clc + 1
        return Obs_id_matrix,Total_Num
    def set_prior_S(self, modi,prior_belief_mat):
        p_mat = self.create_prior_belief_mat_improve()
        prob_prior_s = prior_belief_mat[modi]
        self.net.set_node_definition("S" + str(self.horizon), prob_prior_s)
        # self.net.set_virtual_evidence("S" + str(self.horizon), prob_prior_s)
        self.net.update_beliefs()
        return p_mat, prob_prior_s
    def create_prior_belief_mat_improve(self):
        mat = np.random.rand(self.pnames.P_MAX, self.num_ss)
        [self.normalize(mat[i,:], copy=False) for i in range(0,self.pnames.P_MAX)]
        return mat
    def preset_path(self, Obs_id_matrix, pathi, pathes, evidences):
        path_obs = Obs_id_matrix[pathi, :]
        len_obs = len(path_obs)
        path = np.zeros(len(self.evidences)) - 1
        for obi in range(len_obs - 1, -1, -1):
            path[2 * (len_obs - 1 - obi) + 1] = path_obs[obi]
        if pathi == 0:
            return path
        path[0] = pathes[0][0]
        horizon = self.horizon - 1
        for ei in range(0, horizon):
            obi = 2 * (ei) + 2
            pathj = pathes[len(pathes) - 1]
            if np.sum(np.abs(np.array(path[0:obi]) - np.array(pathj[0:obi]))) == 0:
                path[obi] = pathj[obi]
            else:
                obi = 2 * (ei - 1) + 2
                break
        for ei_index in range(0, len(evidences)):
            if ei_index < obi:
                continue
            ei = evidences[ei_index]
            self.net.clear_evidence(ei)
        return path
    def enter_evidences_s(self,evidences,path,p_mat,prob_prior_s,post_prob_o):
        for ei_index in range(0, len(evidences)):
            ei = evidences[ei_index]
            ei_values = self.net.get_outcome_ids(ei)
            if path[ei_index] != -1:
                choose_index = int(path[ei_index])
                self.net.set_evidence(ei, ei_values[choose_index])
                self.net.update_beliefs()
                continue
            if ei.startswith("D"):
                choose_index =self.argmax_utility(ei,ei_values,post_prob_o)
                # choose_index = self.choose_action(ei,ei_values,p_mat,prob_prior_s)
            path[ei_index] = choose_index
        self.net.update_beliefs()
        return path
    def argmax_utility(self,ei,ei_values,post_prob_o):
        utilitys = np.array(self.net.get_node_value(ei))
        rows = len(ei_values)
        cls = int(len(utilitys)/rows)
        utilitys = utilitys.reshape([cls,rows]).transpose()
        utilitys = np.matmul(utilitys,post_prob_o)
        choose_index = utilitys.argmax()
        self.net.set_evidence(ei, ei_values[choose_index])
        # self.net.update_beliefs()
        return choose_index
    def choose_action(self,ei,ei_values,p_mat,prob_prior_s):
        if self.type == 'DID':
            # compute the utility of each action, then choose the best one
            ei_utility = np.zeros([1, len(ei_values)])
            for evi_index in range(0, len(ei_values)):
                self.net.set_evidence(ei, ei_values[evi_index])
                self.net.update_beliefs()
                beliefs = self.net.get_node_value("U" + ei[1:len(ei)])  # replace
                ei_utility[0, evi_index] = np.mean(beliefs)
                self.net.clear_evidence(ei)
            choose_index = ei_utility.argmax()
            self.net.set_evidence(ei, ei_values[choose_index])
            self.net.update_beliefs()
        if self.type == 'IDID':
            # compute the utility of each action, then choose the best one
            horizon = self.horizon
            if int(ei[1:len(ei)]) < horizon:
                for i in range(0, 1):
                # for i in range(0, self.pnames.P_MAX):
                    # set the prior distribution of S
                    ps =  p_mat[i, :]
                    self.net.set_node_definition("S" + str(horizon), prob_prior_s)
                    # self.net.set_virtual_evidence("S" + str(horizon), ps)
                    # enter the evidence
                    reward = dict()
                    ei_utility = np.zeros([1, len(ei_values)])
                    for evi_index in range(0, len(ei_values)):
                        self.net.set_evidence(ei, ei_values[evi_index])
                        self.net.update_beliefs()
                        beliefs = self.net.get_node_value("U" + ei[1:len(ei)])  # replace
                        ei_utility[0, evi_index] = np.mean(beliefs)
                        self.net.clear_evidence(ei)
                    choose_index = ei_utility.argmax()
                    rw_max = ei_utility.max()
                    if reward.__contains__(choose_index):
                        rw = reward.get(choose_index)
                        rw.append(rw_max)
                        reward[choose_index] = rw
                    else:
                        rw = list()
                        rw.append(rw_max)
                        reward[choose_index] = rw
                ei_utility = np.zeros([1, len(ei_values)])-np.inf
                for key in reward.keys():
                    rw = reward.get(key)
                    ei_utility[0, key] =np.mean(rw)
                self.net.set_node_definition("S" + str(horizon), prob_prior_s)
            else:
                ei_utility = np.zeros([1, len(ei_values)])
                for evi_index in range(0, len(ei_values)):
                    self.net.set_evidence(ei, ei_values[evi_index])
                    self.net.update_beliefs()
                    beliefs = self.net.get_node_value("U" + ei[1:len(ei)])  # replace
                    ei_utility[0, evi_index] = np.mean(beliefs)
                    self.net.clear_evidence(ei)
            choose_index = ei_utility.argmax()
            self.net.set_evidence(ei, ei_values[choose_index])
            self.net.update_beliefs()

        return choose_index
    def get_path_reward(self):
        reward_path = list()
        for hi in range(0, self.horizon):
            reward_path.append(np.mean(np.array(self.net.get_node_value("U" + str(hi + 1)))))
        return reward_path
    def get_path_prob(self,prob_s,path):
        prob_path = 1.0
        for ei_index in range(0, len(self.evidences)):
            ei = self.evidences[ei_index]
            if ei.startswith("D"):
                if int(ei[1:len(ei)]) - 1 == 0:
                    continue
                choose_index_act = int(path[ei_index])
                sub_mat = self.get_sub_definition("S" + str(int(ei[1:len(ei)]) - 1), choose_index_act)
                prob_s = self.norm_prob(sub_mat, prob_s)
            if ei.startswith("O"):  # choose one by one in the path
                choose_index = int(path[ei_index])
                sub_mat = self.get_sub_definition(ei, choose_index_act)
                prob_o = self.norm_prob(sub_mat, prob_s)
                prob_path = prob_path * prob_o[choose_index]
        return prob_path
    def get_sub_definition(self,node,choose_index):
        nums = self.num_ss
        subcolumns = self.num_as
        mat = np.array(self.net.get_node_definition(node))
        node_states = len(self.net.get_outcome_ids(node))
        Total_cls = int(len(mat) / node_states)
        mat = np.transpose(np.reshape(mat,[Total_cls,node_states]))
        c =  int(Total_cls/(subcolumns*nums))
        cls = range(choose_index, int(Total_cls/c), subcolumns) # begin in 0
        cls = [int(ci*c) for ci in cls]
        if c>1:
            cls_b = list()
            for cl in cls:
                for i in range(0,c):
                    cls_b.append(cl+i)
            cls = cls_b
            # print(cls)
        sub_mat = mat[:,cls]
        return  sub_mat
    def set_to_elimit(self,Ps):
        for si in range(0,len(Ps)):
            if Ps[si] ==0:
                Ps[si] = self.pnames.Elimit
        return Ps
    # expansion
    def expansion(self,step, expansion_flag = None):
        # self.step = step
        progress = self.pnames.Steps.get(self.type).get(step)
        self.Node_D = self.pnames.Expansion_nodes.get(self.type).get(progress).get('Node_D')
        self.Node_O = self.pnames.Expansion_nodes.get(self.type).get(progress).get('Node_O')
        self.MODPrefix = self.pnames.MODPrefix.get(self.type).get(self.pnames.Steps.get(self.type)[step])
        # print(self.MODPrefix)
        # node_list and node_labels for OD or D
        # node_list, node_labels, edge and edge_list for MOD
        if expansion_flag is None:
            self.copy_node_dict()
            self.copy_level_dict()
            self.add_node()  #
            self.add_arc()  #
            self.set_state()
            self.set_definition()
            self.normalize_definition()
            self.normalize_O_definition()
            self.write(step)
        else:
            if expansion_flag:
                self.copy_node_dict()
                self.copy_level_dict()
                self.add_node()  #
                self.add_arc()  #
                self.write(step)
            else:
                self.read(step)
                # read file for new tree
                self.copy_node_dict()
                self.copy_level_dict()
                self.initialize_definition()
                self.set_state()
                self.set_definition()
                self.normalize_definition()
                self.normalize_O_definition()
                # self.write(step)
    def copy_node_dict(self):
        action_list, observation_list, node_labels, edgelists, edge_labels, self.node_dict = self.get_tree_parameters()
        for hi in self.node_dict.keys():
            node_states = self.node_dict.get(hi)
            if len(node_states) ==1:
                node_states.append(node_states[0] + '_copy')
            self.node_dict[hi] = node_states
    def get_tree_parameters(self):
        action_list = self.expa_policy_tree.get_action_list()
        observation_list = self.expa_policy_tree.get_observation_list()
        node_labels = self.expa_policy_tree.get_node_labels()
        edgelists = self.expa_policy_tree.get_edgelist()
        edge_labels = self.expa_policy_tree.get_edge_labels()
        node_dict = self.expa_policy_tree.get_node_dict()
        return action_list, observation_list,node_labels, edgelists, edge_labels, node_dict
    def copy_level_dict(self):
        self.level_dict = dict()
        for hi in self.node_dict.keys():
            node_states = self.node_dict.get(hi)
            for ns in node_states:
                self.level_dict[ns] = hi
    def add_node(self):
        # add mod node
        position_dif = self.net.get_node_position(self.Node_O + str(1))
        for hi in range(self.horizon, 0, -1):
            node_type = pysmile.NodeType.CPT
            position = self.net.get_node_position(self.Node_D + str(hi))
            position[1] = position[1] * 2 - position_dif[1]
            self.net.add_node(node_type, self.MODPrefix + str(hi))
            self.net.set_node_name(self.MODPrefix + str(hi), self.MODPrefix + str(hi))
            self.net.set_node_position(self.MODPrefix + str(hi), position)
            node_type = pysmile.NodeType.TRUTH_TABLE
            self.net.set_node_type(self.Node_D + str(hi), node_type)
    def add_arc(self):
        # add arc
        self.net.add_arc("S" + str(self.horizon), self.MODPrefix + str(self.horizon))
        for hi in range(self.horizon, 0, -1):  # add arc from MOD to OD
            node_id_od = self.Node_D + str(hi)
            node_id_mod = self.MODPrefix + str(hi)
            self.net.add_arc(node_id_mod, node_id_od)
            if hi > 1:  # add arc from  MOD to MOD
                node_id_mod_next = self.MODPrefix + str(hi - 1)
                self.net.add_arc(node_id_mod, node_id_mod_next)
            if hi < self.horizon:  # add arc from  OD,OO to MOD
                node_id_oo = self.Node_O + str(hi)
                self.net.add_arc(node_id_oo, node_id_mod)
    def set_state(self):
        # add mod state
        for hi in self.node_dict.keys():
            node_states = self.node_dict.get(hi)
            for ns in node_states:
                self.net.add_outcome(self.MODPrefix + str(hi), ns)
        for hi in self.node_dict.keys():
            node_states = self.net.get_outcome_ids(self.MODPrefix + str(hi))
            for ns in node_states:
                if ns.__contains__('State'):
                    self.net.delete_outcome(self.MODPrefix + str(hi), ns)
        # set others to zeros
        for hi in range(self.horizon, 0, -1):
            node_id_od = self.Node_D + str(hi)
            node_id_mod = self.MODPrefix + str(hi)
            mat = self.net.get_node_definition(node_id_od)
            mat = [0 for e in mat]
            self.net.set_node_definition(node_id_od, mat)
            mat = self.net.get_node_definition(node_id_mod)
            mat = [0 for e in mat]
            self.net.set_node_definition(node_id_mod, mat)
        # set prior belief of MOD
        S_states = len(self.net.get_outcome_ids("S" + str(self.horizon)))
        node_id_mod = self.MODPrefix + str(self.horizon)
        mat = self.net.get_node_definition(node_id_mod)
        mat = [S_states / len(mat) for e in mat]
        self.net.set_node_definition(node_id_mod, mat)
    def initialize_definition(self):
        for hi in range(1, self.horizon):  # from OD MOD 1 to Horizon
            node_id_mod = self.MODPrefix + str(hi)
            mat = self.net.get_node_definition(node_id_mod)
            mat = [0 for e in mat]
            self.net.set_node_definition(node_id_mod, mat)

            node_id_od = self.Node_D + str(hi)
            mat = self.net.get_node_definition(node_id_od)
            mat = [0 for e in mat]
            self.net.set_node_definition(node_id_od, mat)
    def display_dict(self,dict):
        print('-------------- dict-----------')
        for key in dict.keys():
            print("> key     : " + str(key) )
            print("> values  : ")
            print(dict.get(key))
        print('-------------------------------')
    def display(self,node):
        print('----------' + node+'-----------------')
        mat = np.array(self.net.get_node_definition(node))
        node_states = len(self.net.get_outcome_ids(node))
        Total_cls = int(len(mat) / node_states)
        mat = np.transpose(np.reshape(mat, [Total_cls, node_states]))
        print(mat)
    def set_definition(self):
        action_list, observation_list, node_labels, edgelists, edge_labels, self.node_dict = self.get_tree_parameters()
        Num_as_j = len(action_list)
        Num_obs_j = len(observation_list)
        # print('===dsafsdafd=========')
        # print(action_list)
        # self.display_dict(node_labels)
        # self.display_dict(self.node_dict)
        # print(edgelists)
        # self.display_dict(edge_labels)
        # print('===dsafsdafd=========')
        for hi in self.node_dict.keys():
            # set OD_H definition
            node_id_od = self.Node_D + str(hi)
            # self.display(node_id_od)
            node_states = self.net.get_outcome_ids(self.MODPrefix + str(hi))
            mat = np.zeros([len(self.net.get_outcome_ids(node_id_od)), len(node_states)])
            len_mat = len(self.net.get_outcome_ids(node_id_od)) * len(node_states)
            for ns in node_states:
                cl_index = node_states.index(ns)
                if not ns.endswith('_copy'):
                    # print(ns)
                    # print(node_labels.get(ns))
                    row_index = action_list.index(node_labels.get(ns))
                else:
                    row_index = action_list.index(node_labels.get(ns.replace('_copy', '')))
                    # print(ns)
                    # print(node_labels.get(ns.replace('_copy', '')))
                mat[row_index, cl_index] = 1
            mat = np.reshape(mat, [1, len_mat], order='F')
            mat = mat[0]
            self.net.set_node_definition(node_id_od, mat)
        # set MOD_H definition
        for node_pair in edgelists:
            mod_value = node_pair[1]
            edge_label = edge_labels.get(node_pair)
            hi = self.level_dict.get(mod_value)
            if not hi < self.horizon:
                continue
            parent = node_pair[0]
            node_id_mod = self.MODPrefix + str(hi)
            row_shifts = list()
            pre_mod_ids = list()
            node_states = self.net.get_outcome_ids(self.MODPrefix + str(hi))
            node_states_pa = self.net.get_outcome_ids(self.MODPrefix + str(hi + 1))
            total_rows = len(node_states)
            row_shift = self.node_dict.get(hi).index(mod_value)  # dont have copy node
            pre_mod_id = self.node_dict.get(hi + 1).index(parent)  # dont have copy node
            row_shifts.append(row_shift)
            pre_mod_ids.append(pre_mod_id)
            if node_states.__contains__(mod_value + '_copy'):
                row_shifts.append(node_states.index(mod_value + '_copy'))
            if node_states_pa.__contains__(parent + '_copy'):
                pre_mod_ids.append(node_states_pa.index(parent + '_copy'))
            for row_shift in row_shifts:
                for pre_mod_id in pre_mod_ids:
                    # act_value = action_list.index(node_labels.get(parent))# not need any more
                    if edge_label != '*' and edge_label.count('|') == 0:
                        ob_values = list()
                        ob_values.append(observation_list.index(edge_label))
                    if edge_label == '*':
                        ob_values = range(0, Num_obs_j)
                    if edge_label.count('|') > 0:
                        ob_values = list()
                        edge_label_list = edge_label.split('|')
                        for label in edge_label_list:
                            ob_value = observation_list.index(label)
                            ob_values.append(ob_value)
                    for ob_value in ob_values:
                        base_index = int((pre_mod_id) * Num_obs_j * total_rows)
                        column_shift = int(ob_value * total_rows)
                        total_shift = base_index + column_shift + row_shift
                        mat = self.net.get_node_definition(node_id_mod)
                        mat[total_shift] = 1
                        self.net.set_node_definition(node_id_mod, mat)
    def normalize_definition(self):
        for hi in range(1, self.horizon):  # from MOD 1 to Horizon-1
            node_id_mod = self.MODPrefix + str(hi)
            mat = self.net.get_node_definition(node_id_mod)
            node_states = len(self.net.get_outcome_ids(node_id_mod))
            # print(node_states)
            Total_cls = int(len(mat) / node_states)
            for cl in range(0, Total_cls):
                start_index = (cl) * node_states
                end_index = (cl + 1) * node_states
                column = mat[start_index:end_index]
                # print(column )
                cl_sum = np.sum(np.abs(column))
                if cl_sum == 0:
                    column = np.zeros(node_states)
                    column[0] = 1
                    mat[start_index:end_index] = column
                else:
                    column = [i / cl_sum for i in column]
                    mat[start_index:end_index] = column
            self.net.set_node_definition(node_id_mod, mat)
    def normalize_O_definition(self):  # NOT FOR EXPANSION for considering all obervation
        for hi in range(1, self.horizon):  # from MOD 1 to Horizon-1
            node_o = "O" + str(hi)
            mat = self.net.get_node_definition(node_o)
            node_states = len(self.net.get_outcome_ids(node_o))
            # print(node_states)
            Total_cls = int(len(mat) / node_states)
            for cl in range(0, Total_cls):
                start_index = (cl) * node_states
                end_index = (cl + 1) * node_states
                column = mat[start_index:end_index]
                # print(column )
                cl_sum = np.sum(np.abs(column))
                if cl_sum != 0:
                    o_zeros = column.count(0)
                    share = o_zeros / (len(column) - o_zeros)
                    for ci in range(0, len(column)):
                        if column[ci] == 0:
                            column[ci] = self.pnames.Elimit
                        else:
                            column[ci] = column[ci] - self.pnames.Elimit * share
                    cl_sum = np.sum(np.abs(column))
                    column = column / cl_sum
                else:
                    column = np.ones(node_states) / node_states
                mat[start_index:end_index] = column
            self.net.set_node_definition(node_o, mat)