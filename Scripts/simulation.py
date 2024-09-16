# -*- coding: utf-8 -*-

#
# Copyright (C) 2001-2020
# Author: Biyang Ma <biyangma@stu.xmu.edu.cn> <mabiyang001@hotmail.com>
import pysmile
# pysmile_license is your license key
import pysmile_license
# builtins
import random
import numpy as np
import re
import datetime
import os
import sys
# 3rd party
from sklearn.preprocessing import normalize
from graphviz import Source
import graphviz as G
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
import matplotlib.colors as mcolors
from icecream import ic
# self package
# from Model import Model
from DID import DID
from IDID import IDID
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
class simulation():
    #sim
    Test_style = {1:'Offline test',2:'Online test'}
    Sim_mode = {1:'random_path',2:'most_possible_path',3:'tree'}
    Domain = {1:'Tiger',2:'UAV_IPTG',3:'UAV_ORG',4:'DUAV'}
    # did
    Gen_method = {1: 'Null', 2: 'GA', 3: 'GAP', 4: 'GAG',5:'MD',6:'MD'}
    Fitness_method = {1: 'distance', 2: 'diversity', 3: 'reward'}
    Group_criterion_method = {1: 'distance', 2: 'diversity', 3: 'reward'}
    #exp
    Models = {1:'IDID',2:'IDID-GA',3:'IDID-GAP',4:'IDID-GAG',5: 'IDID-MDP',6:'IDID-MDT'}
    def __init__(self):
        # attributes
        self.domain = simulation.Domain[1]
        self.test_style = simulation.Test_style[1]
        self.save_filepath = "./Results/"
        self.outputlog = "outputlog.txt"
        self.gui_mod = False
        self.output_handler = None
        self.log_mod = False
        self.outputlog_handler = None
        self.buff = list()

        self.model = 'IDID'
        self.num_mod_idid = 2
        self.num_mod_did = 5
        self.num_mod_idid_ini = self.num_mod_idid
        self.num_mod_did_ini =  self.num_mod_did
        self.horizon_max = 6
        self.num_test = 10
        self.compare_models = [2,4,1]
        self.main_model_id = 1
        self.compare_idids = list()
        self.preset_belief = True
        self.rewards = dict()
        self.idids = dict()

        self.idid = IDID(self.domain,num_mod = self.num_mod_idid,num_mod_did=self.num_mod_did,
                         horizon_max = self.horizon_max,preset_belief = self.preset_belief)
        self.did = DID(self.domain,num_mod = self.num_mod_did,horizon_max = self.horizon_max,
                       preset_belief = self.preset_belief)
        self.idid.did.gen_method = simulation.Gen_method[4]
        self.idid.did.fitness_method = simulation.Fitness_method[3]
        self.idid.did.group_criterion_method = simulation.Group_criterion_method[2]

        self.set_parameters()
        self.ini_ga()

        self.rewards_mean = np.zeros([self.num_mod_idid,self.num_mod_did])
        self.rewards_var = np.zeros([self.num_mod_idid, self.num_mod_did])
        self.rewards_std = np.zeros([self.num_mod_idid, self.num_mod_did])
        self.mode_i = simulation.Sim_mode[3]
        self.mode_j = simulation.Sim_mode[3]
        self.policytrees_i = dict()
        self.policytrees_j = dict()

        self.create_prior_belief_mat()
        self.display_parameters()
    # experiments
    def set_main_model(self,main_model_id):
        self.main_model_id = main_model_id
        ic(self.main_model_id)
    def ini_ga(self):
        self.fitness_method = dict()
        self.group_criterion_method = dict()
        self.fitness_method['IDID-GA'] = simulation.Fitness_method[3]
        self.group_criterion_method['IDID-GA'] = simulation.Group_criterion_method[2]

        self.fitness_method['IDID-GAG'] = simulation.Fitness_method[3]
        self.group_criterion_method['IDID-GAG'] = simulation.Group_criterion_method[2]

        self.parameters_dict = dict()
        self.parameters_dict['IDID-GA'] = self.parameters
        self.parameters_dict['IDID-GAG'] = self.parameters

        self.did_filenames = dict()
    def set_parameters(self):
        self.parameters = dict()
        self.parameters['pop_size'] = 10
        self.parameters['generations'] = 50
        self.parameters['tournament_size'] = 5
        self.parameters['crossover_rate'] = 0.8
        self.parameters['mutation_rate'] = 0.1
        self.parameters['pelite_mode'] = True
        self.parameters['elite_mode']= True
        self.parameters['odd_crossoperate']= True
        self.parameters['weight_flag']= True

        self.parameters['group_size'] = 5
        self.parameters['emigrate_rate'] = 0.1
    def set_did_parameters(self,did_filenames):
        if did_filenames is not None:
            self.did_filenames = did_filenames
    def set_ga_parameters(self,fitness_method,group_criterion_method,parameters_dict ):
        if fitness_method is not None:
            self.fitness_method = fitness_method
        if group_criterion_method is not None:
            self.group_criterion_method = group_criterion_method
        if parameters_dict is not None:
            self.parameters_dict = parameters_dict
    def set_sim_mode(self,mode_i,mode_j):
        self.mode_i = mode_i
        self.mode_j = mode_j
    def set_compare_models(self,compare_models):
        self.compare_models =  compare_models
    def create_prior_belief_mat_NS(self,N,S):
        mat = np.random.rand(N,S)
        mat = normalize(mat, axis=1, norm='l1')
        return mat
    def plot_exp_reward(self):
        num_mod_did_list = list()
        keys = list()
        for key in self.rewards.keys():
            rewards_mean = self.rewards.get(key)[0]
            values = rewards_mean[0, :]
            num_mod_did_list.append(len(values))
            keys.append(key)
        num_mod_did_max = np.max(num_mod_did_list)
        index = np.argmax(num_mod_did_list)
        indexs = [keys[index]]
        for i in range(0,len(keys)):
            if i !=index:
                indexs.append(keys[i])
        bar_width = 1/(len(indexs)+1)
        for modi in range(1, self.num_mod_idid + 1):
            fig = plt.figure()
            axis = fig.gca()
            count = 0
            for key in indexs:
                rewards_mean = self.rewards.get(key)[0]
                rewards_std = self.rewards.get(key)[2]
                values = rewards_mean[modi - 1, :]
                SD = rewards_std[modi - 1, :]
                num_mod_did = len(values)
                index = np.arange(num_mod_did)
                color_set = mcolors.CSS4_COLORS
                ckeys = list(color_set.keys())
                color = []
                for i in range(0,len(values)):
                    if values[i] < 0:
                        color.append(color_set[ckeys[0+3*count+10]])
                    if values[i] == 0:
                        color.append(color_set[ckeys[1+3*count+10]])
                    if values[i] > 0:
                        color.append(color_set[ckeys[2+3*count+10]])
                color = tuple( color)
                plt.bar(index + count*bar_width, values, width= bar_width,yerr=SD,
                        color =color,error_kw={'ecolor': '0.2', 'capsize': 6},
                        alpha=0.7, label='First')
                # # 添加数据标签
                # for a, b in zip(index, values):
                #    plt.text(a+ count*bar_width, 1.1*b, '%.0f' % b, ha='center', va='bottom', fontsize=10)
                # plt.legend(key, loc='center')
                count =  count +1
            title = 'i\'s mod ' + str(modi) + ': i\'s ' + self.mode_i + ' vs ' + ' j\'s ' + self.mode_j
            plt.title(title)
            ticks = list()
            index = np.arange(num_mod_did_max)
            for modj in range(1, num_mod_did_max  + 1):
                ticks.append('m' + str(modj))
            plt.xticks(index + 0.5, ticks)
            plt.legend(indexs, loc='best')

            plt.setp(axis.get_xticklabels(), rotation=30, horizontalalignment='right')

            axis.set_ylabel('Reward')
            axis.set_xlabel('Agent j\'s policy trees')
            plt.show()
            title = 'Reward @ ' + str(modi) + ' agent i\'s ' + self.mode_i + ' vs ' + ' agent j\'s ' + self.mode_j+'_'+str(self.horizon_max)
            fig.savefig(self.save_filepath + title + '_ALL.pdf')
    def exp_gui(self):
        print('-------------- Testing now --------------------------')
        # self.create_prior_belief_mat()
        self.display_parameters()
        # # create_prior_belief_mat of idid model for testing
        # prior_belief_idid = self.create_prior_belief_mat_NS(self.num_mod_idid_ini, self.idid.get_num_ss())
        # prior_belief_did = self.create_prior_belief_mat_NS(self.num_mod_did_ini, self.idid.did.get_num_ss())
        # #
        ic((self.num_mod_idid_ini, self.idid.get_num_ss()))
        if self.compare_models.index(self.main_model_id)!= 0                    :
            models = list()
            models.append(self.main_model_id)
            [models.append(ei) for ei in self.compare_models if ei !=self.main_model_id]
            self.compare_models = models
        ic(self.compare_models)
        for i in range(0,len(self.compare_models)):
            ei = self.compare_models[i]
            print('--------------  run model: '+ simulation.Models[ei]+'   -------------------')
            self.idid = IDID(self.domain, num_mod=self.num_mod_idid_ini, num_mod_did=self.num_mod_did_ini,
                        horizon_max=self.horizon_max, preset_belief = self.preset_belief)
            if ei == self.main_model_id:
                self.create_prior_belief_mat()
                # create_prior_belief_mat of idid model for testing
                prior_belief_idid = self.create_prior_belief_mat_NS(self.num_mod_idid_ini, self.idid.get_num_ss())
                prior_belief_did = self.create_prior_belief_mat_NS(self.num_mod_did_ini, self.idid.did.get_num_ss())
                #

            # self.idid.set_preset_belief( self.preset_belief)
            # self.idid.did.set_preset_belief( self.preset_belief)
            self.idid.set_prior_belief_mat(prior_belief_idid)
            self.idid.did.set_prior_belief_mat(prior_belief_did)

            self.idid.did.gen_method = simulation.Gen_method[ei]
            self.gen_method = simulation.Gen_method[ei]
            self.idid.did.fitness_method = self.fitness_method.get(simulation.Models[ei])
            self.idid.did.group_criterion_method = self.group_criterion_method.get(simulation.Models[ei])
            self.idid.did.parameters= self.parameters_dict.get(simulation.Models[ei])
            self.idid.did.filename = self.did_filenames.get(simulation.Models[ei])
            ic( self.idid.did.filename)
            #self.prior_belief_mat = prior_belief_idid
            #self.num_test = self.num_mod_idid
            if ei != self.main_model_id:
                self.idid.did.copy_did(self.idids.get(simulation.Models[self.main_model_id]).did,self.num_mod_did_ini)
            # self.create_prior_belief_mat()
            # self.simulate()
            starttime = datetime.datetime.now()
            # get policy tree
            if ei == self.main_model_id :
                load = None
            else:
                load = True
            self.idid.gen_idid_pathes(load)
            self.idids[simulation.Models[ei]] = self.idid
            # self.create_prior_belief_mat()
            # self.did.gen_did_pathes()
            # self.offline_test()
            endtime = datetime.datetime.now()
            print('------- Duration ------- ')
            print((endtime - starttime))
            print((endtime - starttime).seconds)
            # self.rewards[simulation.Models[ei]] = [self.rewards_mean,self.rewards_var,self.rewards_std]
            self.policytrees_i[simulation.Models[ei]] = self.idid.policy_tree
            self.policytrees_j[simulation.Models[ei]] = self.idid.did.policy_tree
            # self.display()
            # self.plot_reward()
            # # self.plot_reward_3d()
            # self.plot_reward_heatmap()
            # self.compare_idids.append(self.idid)
        # self.plot_exp_reward()
    def offline_test_gui(self):
        print('-------------- analysis now --------------------------')
        for ei in self.compare_models:
            print('-------------  run model: ' + simulation.Models[ei] + '   -------------------')
            self.idid =  self.idids[simulation.Models[ei]]
            #self.prior_belief_mat = self.idid.prior_belief_mat
            #self.num_test =  self.num_test
            ic(self.num_test)
            ic(self.prior_belief_mat)
            self.num_mod_idid =  self.idid.num_mod
            self.num_mod_did =  self.idid.did.num_mod
            self.offline_test_gui_x()
            self.rewards[simulation.Models[ei]] = [self.rewards_mean, self.rewards_var, self.rewards_std]
            self.policytrees_i[simulation.Models[ei]] = self.idid.policy_tree
            self.policytrees_j[simulation.Models[ei]] = self.idid.did.policy_tree
        print('-------------- analysis done --------------------------')

    def offline_test_gui_x(self):
        self.initialize_reward()
        print('===================== offline test ==================')
        print('agent i simulate @: ' + self.mode_i)
        print('agent j simulate @: ' + self.mode_j)
        bar = tqdm(total=int(self.num_mod_idid * self.num_mod_did))
        for modi in range(1, self.num_mod_idid + 1):
            pathes_i = self.idid.policy_dict.get(modi)
            weights_i = self.idid.policy_path_weight.get(modi)
            pathes_i, weights_i = self.gen_pathes_weights(pathes_i, weights_i, self.mode_i)
            # ic(pathes_i)
            for modj in range(1, self.num_mod_did + 1):
                pathes_j = self.idid.did.policy_dict.get(modj)
                weights_j = self.idid.did.policy_path_weight.get(modj)
                pathes_j, weights_j = self.gen_pathes_weights(pathes_j, weights_j, self.mode_j)
                # ic(pathes_j)
                reward = self.get_rewards_gui(pathes_i, pathes_j, weights_i, weights_j)
                self.rewards_mean[modi - 1, modj - 1] = np.mean(reward)
                self.rewards_var[modi - 1, modj - 1] = np.var(reward)
                self.rewards_std[modi - 1, modj - 1] = np.std(reward)
                bar.update(1)
        bar.close()
    def get_rewards_gui(self, pathes_i, pathes_j, weights_i, weights_j):
        reward = list()
        path_i_pre = []
        for pi in range(0, len(pathes_i)):
            path_i = pathes_i[pi]
            weight_i = weights_i[pi]
            self.enter_evidences_gui(self.idid.evidences, path_i)
            for pj in range(0, len(pathes_j)):
                path_j = pathes_j[pj]
                weight_j = weights_j[pj]
                self.enter_evidences_gui(['O' + ei for ei in self.idid.did.evidences], path_j)
                for i in range(0, self.num_test):
                    prior_belief = self.prior_belief_mat[i, :]
                    ei = "S" + str(self.get_horizon_max())
                    self.idid.net.set_node_definition(ei, prior_belief)
                    self.idid.net.update_beliefs()
                    reward_path = list()
                    for hi in range(0, self.get_horizon_max()):
                        reward_path.append(np.mean(np.array(self.idid.net.get_node_value("U" + str(hi + 1)))))
                    prob_path = 1.0
                    reward.append(np.sum(reward_path) * prob_path)
        self.idid.net.clear_all_evidence()
        return reward
    def enter_evidences_gui(self,evidences,path):
        [self.idid.net.clear_evidence(ei) for ei in evidences]
        for ei_index in range(0, len(evidences)):
            ei = evidences[ei_index]
            self.idid.net.set_evidence(ei,int(path[ei_index]))
        self.idid.net.update_beliefs()
    def exp(self):

        print('-------------- Testing now --------------------------')
        # create_prior_belief_mat of idid model for testing
        prior_belief_idid = self.create_prior_belief_mat_NS(self.num_mod_idid_ini, self.idid.get_num_ss())
        prior_belief_did = self.create_prior_belief_mat_NS(self.num_mod_did_ini, self.idid.did.get_num_ss())
        #
        for ei in self.compare_models:
            print('--------------  run model: '+ simulation.Models[ei]+'   -------------------')
            self.idid = IDID(self.domain, num_mod=self.num_mod_idid_ini, num_mod_did=self.num_mod_did_ini,
                        horizon_max=self.horizon_max, preset_belief = self.preset_belief)
            # self.idid.set_preset_belief( self.preset_belief)
            # self.idid.did.set_preset_belief( self.preset_belief)
            self.idid.set_prior_belief_mat(prior_belief_idid)
            self.idid.did.set_prior_belief_mat(prior_belief_did)

            self.idid.did.gen_method = simulation.Gen_method[ei]
            self.gen_method = simulation.Gen_method[ei]
            self.idid.did.fitness_method = self.fitness_method.get(simulation.Models[ei])
            self.idid.did.group_criterion_method = self.group_criterion_method.get(simulation.Models[ei])
            self.idid.did.parameters= self.parameters_dict.get(simulation.Models[ei])
            self.prior_belief_mat = prior_belief_idid
            self.num_test = self.num_mod_idid

            # self.create_prior_belief_mat()
            self.simulate()
            self.rewards[simulation.Models[ei]] = [self.rewards_mean,self.rewards_var,self.rewards_std]
            self.policytrees_i[simulation.Models[ei]] = self.idid.policy_tree
            self.policytrees_j[simulation.Models[ei]] = self.idid.did.policy_tree
            # self.display()
            # self.plot_reward()
            # # self.plot_reward_3d()
            # self.plot_reward_heatmap()
            # self.compare_idids.append(self.idid)
        self.plot_exp_reward()
    # method
    def open_gui_mod(self):
        if self.gui_mod:
            self.set_log_mod(True)
            self.open_log_mod()
    def close_gui_mod(self):
        self.close_log_mod()
        self.set_gui_mod(False)
    def open_log_mod(self):
        if self.log_mod:
            self.output_handler =  sys.stdout  # 记录当前输出指向，默认是consle
            if os.path.isfile(self.outputlog):
                print('文件已存在,进行重写覆盖')
                os.remove(self.outputlog)
                return 0
            else:
                print('文件不存在,进行重新创建')
            self.outputlog_handler =  open(self.outputlog, 'w')
            sys.stdout = self.outputlog_handler # 输出指向txt文件
            print("filepath:", __file__,"\nfilename:", os.path.basename(__file__))
             # print(self.outputlog_handler.readlines())  # 将记录在文件中的结果输出到屏幕
            self.display_parameters()
    def close_log_mod(self):
        sys.stdout = self.output_handler  # 输出重定向回consle
        self.outputlog_handler.close() #关闭文件句柄
        self.set_log_mod(False)
    def display_parameters(self):
        print('--------------   parameters    ---------------------')
        print('>> Model name: ' + self.model)
        print('>> Domain name: '+self.domain)
        print('>> Maximun horizon: ' +str(self.horizon_max))
        print('>> Size of DID models: ' +str(self.num_mod_did))
        print('>> Size of IDID models: ' + str(self.num_mod_idid))
        print('>> Size of tets: ' +str(self.num_test))
        print('----------------------------------------------------')
    def display(self):
        print('-----------------------------------------------------')
        for rw in range(0,self.num_mod_idid):
            print('>@ mod:' + str(rw+1))
            print(self.rewards_mean[rw,:])
            print(self.rewards_var[rw, :])
            print(self.rewards_std[rw, :])
        print('-----------------------------------------------------')
    def random_set_evidence(self,net_node_str):
        outcomes = self.idid.net.get_outcome_ids(net_node_str)
        outcome = random.sample(outcomes, 1)[0]
        self.idid.net.set_evidence(net_node_str, outcome)
        return outcome
    # get
    def get_model(self):
        return self.model
    def get_domain(self):
        return self.domain
    def get_num_mod_idid(self):
        return self.num_mod_idid
    def get_num_mod_did(self):
        return self.num_mod_did
    def get_horizon_max(self):
        return self.horizon_max
    def get_num_test(self):
        return self.num_test
    def get_gui_mod(self):
        return self.gui_mod
    def get_log_mod(self):
        return self.log_mod
    # set
    def set_model(self,model):
        self.model = model
    def set_domain(self,domain):
        self.domain = domain
        self.idid.domain = domain
        self.idid.did.domain = domain
    def set_num_mod_idid(self,num_mod_idid):
        self.num_mod_idid = num_mod_idid
        self.idid.num_mod = num_mod_idid
        self.num_mod_idid_ini = self.num_mod_idid
    def set_num_mod_did(self,num_mod_did):
        self.num_mod_did = num_mod_did
        self.idid.did.num_mod = num_mod_did
        self.num_mod_did_ini = self.num_mod_did
    def set_horizon_max(self,horizon_max):
        self.horizon_max = horizon_max
        self.idid.set_horizon_max(self.horizon_max)
        self.idid.did.set_horizon_max(self.horizon_max)
    def set_num_test(self,num_test):
        self.num_test = num_test
    def set_gui_mod(self,gui_mod):
        self.gui_mod = gui_mod
    def set_log_mod(self,log_mod):
        self.log_mod = log_mod
    # function
    def eucliDist(self,A, B):
        return np.sqrt(sum(np.power((A - B), 2)))
    def plot_states(self):
        fig = plt.figure()
        axis = fig.gca()
        words =  list()
        for modj in range(1, self.num_mod_did + 1):
            words.append('mj' + str(modj))
        for modi in range(1, self.num_mod_idid + 1):
            words.append('mi' + str(modi))
        for i in range(1, self.num_test + 1):
            words.append('t' + str(i))

        word2Ind = dict()
        colors = dict()
        num_total = self.num_mod_did + self.num_mod_idid + self.num_test
        M_total = list()
        for j,word in enumerate(words):
            word2Ind[word] = j
            if j <= self.num_mod_did-1:
                colors[word] = 'red'
                M_total.append(self.idid.did.prior_belief_mat[j])
            if j >= self.num_mod_did and j <= self.num_mod_did + self.num_mod_idid-1:
                colors[word] = 'blue'
                M_total.append(self.idid.prior_belief_mat[j-self.num_mod_did])
            if j >= self.num_mod_did + self.num_mod_idid:
                colors[word] = 'green'
                M_total.append(self.prior_belief_mat[j - self.num_mod_did - self.num_mod_idid])
        M_total = np.array(M_total)
        gap = np.array(range(0,len(words)))/len(words)

        posy = list()
        for word in words:
            x, y = M_total[word2Ind[word]]
            posy.append(y)
        posy.sort(reverse=False)
        pos = dict()
        for word in words:
            x, y = M_total[word2Ind[word]]
            ind = posy.index(y)
            # if ind >= len(words)/2:
            if gap[ind]>=y:
                pos[word] = (x + 0.2,gap[ind])
            else:
                pos[word] = (x - 0.2 ,gap[ind])
        for word in words:
            x, y = M_total[word2Ind[word]]
            plt.scatter(x, y, marker='x', color=colors[word])
            axis.annotate(word, (x, y), xytext=pos[word], color=colors[word], fontsize=6,
                          arrowprops=dict(arrowstyle="->",
                                          connectionstyle="angle3,angleA=0,angleB=-90"))
        axis.set_ylabel('$S_2$')
        axis.set_xlabel('$S_1$')
        axis.set_xlim(-0.25, 1.25)  # lower limit (0)
        axis.set_ylim(0, 1.1)  # lower limit (0)
        title = 'i\'s mod ' + str(modi) + ': i\'s ' + self.mode_i + ' vs ' + ' j\'s ' + self.mode_j+'_'+str(self.horizon_max)
        # plt.title(title)
        plt.show()
        fig.savefig(self.save_filepath  + 'Prior states.pdf')
        # fig.savefig(self.save_filepath + title + '_states.pdf')
    def plot_states_heatmap(self):
        fig = plt.figure(figsize=(20,20))
        ax = fig.gca()
        words = list()
        for modj in range(1, self.num_mod_did + 1):
            words.append('mj' + str(modj))
        for modi in range(1, self.num_mod_idid + 1):
            words.append('mi' + str(modi))
        for i in range(1, self.num_test + 1):
            words.append('t' + str(i))
        M_total = list()
        for j, word in enumerate(words):
            if j <= self.num_mod_did - 1:
                M_total.append(self.idid.did.prior_belief_mat[j])
            if j >= self.num_mod_did and j <= self.num_mod_did + self.num_mod_idid - 1:
                M_total.append(self.idid.prior_belief_mat[j - self.num_mod_did])
            if j >= self.num_mod_did + self.num_mod_idid:
                M_total.append(self.prior_belief_mat[j - self.num_mod_did - self.num_mod_idid])
        M_total = np.array(M_total)
        num_total = self.num_mod_did + self.num_mod_idid + self.num_test
        data = np.zeros([num_total,num_total])
        for i in range(0,num_total):
            for j in range(0, num_total):
                data[i,j] = self.eucliDist(M_total[i],M_total[j])
        ax.set_ylabel('Prior states')
        ax.set_xlabel('Prior states')#, labelpad=40)
        title = 'Adjcent Matrix'
        ax.set_title(title)
        im, cbar = heatmap(data, words, words, ax=ax,
                           cmap="magma_r", cbarlabel="Reward")#"YlGn""Wistia""magma_r""PuOr"
        texts = annotate_heatmap(im, valfmt="{x:.1f}")
        plt.show()
        title = title+'_'+str(self.horizon_max)
        fig.savefig(self.save_filepath + title + '_heatmap.pdf')
    def plot_reward_heatmap(self):
        fig, ax = plt.subplots()
        xticks = list()
        for modj in range(1, self.num_mod_did + 1):
            xticks.append('m' + str(modj))
        yticks = list()
        for modi in range(1, self.num_mod_idid + 1):
            yticks.append('m' + str(modi))
        ax.set_ylabel('Agent i\'s policy trees')
        ax.set_xlabel('Agent j\'s policy trees')#, labelpad=40)
        title = ' i\'s ' + self.mode_i + ' vs ' + 'j\'s ' + self.mode_j
        ax.set_title(title)
        im, cbar = heatmap(self.rewards_mean, yticks, xticks, ax=ax,
                           cmap="magma_r", cbarlabel="Reward")#"YlGn""Wistia""magma_r""PuOr"
        texts = annotate_heatmap(im, valfmt="{x:.1f}")

        title = 'Reward @ ' +' i\'s ' + self.mode_i + ' vs ' + 'j\'s ' + self.mode_j+'_'+str(self.horizon_max)
        plt.show()
        fig.savefig(self.save_filepath + title + '_heatmap.pdf')
    def plot_reward_3d(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        data = list()
        minvalue = np.min(np.min(self.rewards_mean))
        maxvalue = np.max(np.max(self.rewards_mean))
        if  np.min(np.min(self.rewards_mean)) == 0:
            minvalue = -1
        if  np.max(np.max(self.rewards_mean)) == 0:
            maxvalue = 1
        if  np.min(np.min(self.rewards_mean)) > 0 :
            minvalue = 0
            maxvalue = maxvalue * 1.2
        if  np.max(np.max(self.rewards_mean))  < 0:
            minvalue =  minvalue* 1.2
            maxvalue = 0
        if  np.min(np.min(self.rewards_mean)) < 0 and np.max(np.max(self.rewards_mean)) > 0:
            minvalue = minvalue * 1.2
            maxvalue = maxvalue * 1.2
        for modi in range(1, self.num_mod_idid + 1):
            values = self.rewards_mean[modi - 1, :]
            data.append(list(values))
        color_set = ['black', 'red', 'blue']
        color = []
        for ej in data:
            for eji in ej:
                if eji < 0:
                    color.append(color_set[1])
                if eji == 0:
                    color.append(color_set[0])
                if eji > 0:
                    color.append(color_set[2])
        color = tuple(color)
        xticks = list()
        for modj in range(1, self.num_mod_did + 1):
            xticks.append('m' + str(modj))
        yticks = list()
        for modi in range(1, self.num_mod_idid + 1):
            yticks.append('m' + str(modi))
        title = 'Reward @ ' + ' agent i\'s ' + self.mode_i + ' vs ' + ' agent j\'s ' + self.mode_j+'_'+str(self.horizon_max)

        x = list(range(0, len(data[0]))) * len(data)
        y = sum([[i] * len(data[0]) for i in range(len(data))], [])
        z = [0] * len(data[0]) * len(data)

        dx = 0.5
        dy = 0.5
        dz = sum(data, [])

        ax.w_xaxis.set_ticks([i + dx / 2 for i in range(len(data[0]))])
        ax.w_xaxis.set_ticklabels(xticks)
        plt.setp(ax.get_xticklabels(), rotation=60, horizontalalignment='right')

        ax.w_yaxis.set_ticks([i + dy / 2 for i in range(len(data))])
        ax.w_yaxis.set_ticklabels(yticks)
        plt.setp(ax.get_yticklabels(), rotation=-30, horizontalalignment='right')

        ax.set_title(title)
        ax.set_zlabel('Reward')
        ax.set_ylabel('Agent i\'s policy trees')
        ax.set_xlabel('Agent j\'s policy trees', labelpad=40)


        # print( minvalue,maxvalue)
        ax.set_zlim(minvalue, maxvalue)  # lower limit (0)

        # stretch axis, thanks to stackoverflow.com/q/30223161
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, len(data) / len(data[0]), 1, 1]))

        # color = cm.rainbow([0.2 + (1 - 0.2) / (len(x) - 1) * i for i in range(len(x))])

        ax.bar3d(x, y, z, dx, dy, dz, color)
        # rotate the axes and update

        plt.show()
        title = title+'_'+str(self.horizon_max)
        fig.savefig(self.save_filepath + title + '_3d.pdf')
    def plot_reward(self):
        for modi in range(1, self.num_mod_idid + 1):
            fig = plt.figure()
            axis = fig.gca()
            index = np.arange(self.num_mod_did)
            ticks = list()
            for modj in range(1, self.num_mod_did + 1):
                ticks.append('m'+str(modj))
            values = self.rewards_mean[modi - 1, :]
            SD = self.rewards_std[modi - 1, :]
            title = 'i\'s mod '+ str(modi)+ ': i\'s '+ self.mode_i + ' vs ' + ' j\'s '+ self.mode_j
            plt.title(title)
            color_set = ['black','red','blue']
            color = []
            for i in range(0,len(values)):
                if values[i] < 0:
                    color.append(color_set[1])
                if values[i] == 0:
                    color.append(color_set[0])
                if values[i] > 0:
                    color.append(color_set[2])
            color = tuple( color)
            plt.bar(index, values, yerr=SD, color =color,error_kw={'ecolor': '0.2', 'capsize': 6}, alpha=0.7, label='First')
            plt.xticks(index + 0.2, ticks)
            # 添加数据标签
            for a, b in zip(index, values):
                plt.text(a , 1.1 * b, '%.0f' % b, ha='center', va='bottom', fontsize=10)
            plt.setp(axis.get_xticklabels(), rotation=30, horizontalalignment='right')
            # plt.legend('reward', loc='center')
            axis.set_ylabel('Reward')
            axis.set_xlabel('Agent j\'s policy trees')
            plt.show()
            title = 'Reward @ ' + str(modi) + ' agent i\'s ' + self.mode_i + ' vs ' + ' agent j\'s ' + self.mode_j+'_'+str(self.horizon_max)
            fig.savefig(self.save_filepath + title + '.pdf')
    def all_mode_simulate(self):
        # self.create_prior_belief_mat()
        self.plot_states()
        self.plot_states_heatmap()
        for mode_i in  simulation.Sim_mode.keys():
            self.mode_i = simulation.Sim_mode[mode_i]
            for mode_j in simulation.Sim_mode.keys():
                self.mode_j = simulation.Sim_mode[mode_j]
                self.offline_test()
                self.display()
                self.plot_reward()
                # self.plot_reward_3d()
                self.plot_reward_heatmap()

        print('=================================================')
    def simulate1(self):
        starttime = datetime.datetime.now()
        self.idid.gen_idid_pathes()
        #self.did.gen_did_pathes()
        for modi in range(1,self.num_mod_idid+1):
            for modj in range(1,self.num_mod_did+1):
                reward = list()
                evidences = list()
                for i in range(0,self.num_test):
                    node_i = self.idid.policy_tree.get_root()
                    arc_to_mod_i = self.idid.policy_tree.get_node_string_head() + str(modi)
                    node_i = node_i.get_arcs_to_children().get(arc_to_mod_i)

                    self.random_set_evidence('O' + str(self.horizon_max))
                    evidences.append('O' + str(self.horizon_max))


                    node_j = self.idid.did.policy_tree.get_root()
                    arc_to_mod_j = self.idid.did.policy_tree.get_node_string_head() + str(modj)
                    node_j = node_j.get_arcs_to_children().get(arc_to_mod_j)
                    reward_path = 0
                    while not node_i.is_leaf():
                        # according to i's policytree, set agent i's action and receive observation in IDID model
                        # according to j's policytree, set agent j's action and receive observation in DID model
                        # then, set agent j's action and  receive observation in IDID model
                        # calculate the untility, then get the total reward of j's policy tree
                        # OD impossible to choose new policy tree‘s action
                        # err:pysmile.SMILEException: SMILE Error Occured in: Network.SetEvidence # ErrNo=-26
                        hi = node_i.get_level()
                        #node_i.display()
                        #node_j.display()
                        evidences.append('D' + str(hi))
                        evidences.append('MODD' + str(hi))

                        self.idid.net.set_evidence('D' + str(hi), node_i.get_label())
                        self.idid.net.set_evidence('MOD' + str(hi), node_j.get_id())
                        if hi >1:
                            evidences.append('O' + str(hi-1))
                            evidences.append('OO' + str(hi-1))
                            observation_i = self.random_set_evidence('O' + str(hi-1))
                            observation_j = self.random_set_evidence('OO'+ str(hi-1))
                            # move to next step

                            node_i = node_i.get_arcs_to_children().get(observation_i)
                            node_j = node_j.get_arcs_to_children().get(observation_j)
                        if node_i.get_level()==1:
                            hi = node_i.get_level()
                            self.idid.net.set_evidence('D' + str(hi), node_i.get_label())
                            self.idid.net.set_evidence('MOD' + str(hi), node_j.get_id())
                            evidences.append('D' + str(hi))
                            evidences.append('MODD' + str(hi))
                            node_i = self.idid.policy_tree.get_leaf()

                    self.idid.net.update_beliefs()
                    # print(set(evidences))

                    for hi in range(1,self.get_horizon_max()):
                        beliefs = self.idid.net.get_node_value("U" + str(hi))
                        # print(beliefs)
                        reward_path = reward_path + beliefs[0]
                    self.idid.net.clear_all_evidence()
                    reward.append(reward_path)

                self.rewards_mean[modi-1,modj-1] = self.rewards_mean[modi-1,modj-1] + np.mean(reward)
                self.rewards_var[modi-1,modj-1] = self.rewards_var[modi-1, modj-1] + np.var(reward)
                self.rewards_std[modi-1,modj-1] = self.rewards_std[modi-1, modj-1] + np.std(reward)
        endtime = datetime.datetime.now()
        print('------- Duration ------- ')
        print((endtime - starttime))
        print((endtime - starttime).seconds)
    def simulate(self):
        starttime = datetime.datetime.now()
        # get policy tree
        self.idid.gen_idid_pathes()
        # self.create_prior_belief_mat()
        #self.did.gen_did_pathes()
        self.offline_test()
        endtime = datetime.datetime.now()
        print('------- Duration ------- ')
        print((endtime - starttime))
        print((endtime - starttime).seconds)
    def initialize_reward(self):
        self.num_mod_idid = len([key for key in self.idid.policy_dict.keys()])
        self.num_mod_did = len([key for key in self.idid.did.policy_dict.keys()])
        self.rewards_mean = np.zeros([self.num_mod_idid,self.num_mod_did])
        self.rewards_var = np.zeros([self.num_mod_idid, self.num_mod_did])
        self.rewards_std = np.zeros([self.num_mod_idid, self.num_mod_did])
    def offline_test(self):
        self.initialize_reward()
        print('===================== offline test ==================')
        print('agent i simulate @: ' + self.mode_i)
        print('agent j simulate @: ' + self.mode_j)
        bar = tqdm(total=int( self.num_mod_idid*self.num_mod_did))
        for modi in range(1, self.num_mod_idid + 1):
            pathes_i = self.idid.policy_dict.get(modi)
            weights_i = self.idid.policy_path_weight.get(modi)
            pathes_i,weights_i = self.gen_pathes_weights(pathes_i, weights_i, self.mode_i)

            for modj in range(1, self.num_mod_did + 1):
                pathes_j = self.idid.did.policy_dict.get(modj)
                weights_j = self.idid.did.policy_path_weight.get(modj)
                pathes_j, weights_j = self.gen_pathes_weights(pathes_j, weights_j, self.mode_j)

                reward = self.get_rewards(pathes_i, pathes_j, weights_i, weights_j)
                self.rewards_mean[modi - 1, modj - 1] = np.mean(reward)
                self.rewards_var[modi - 1, modj - 1] = np.var(reward)
                self.rewards_std[modi - 1, modj - 1] = np.std(reward)
                bar.update(1)
        bar.close()
    def russian_roulette(self,weights):
        sum_w = np.sum(weights)
        sum_ = 0
        u = random.random() * sum_w
        for pos in range(0, len(weights)):
            sum_ += weights[pos]
            if sum_ > u:
                break
        return pos
    def gen_pathes_weights(self,pathes,weights,mode):
        if mode == simulation.Sim_mode[3]:
            return pathes,weights
        if mode == simulation.Sim_mode[2]:
            choose = np.argmax(np.array(weights))
            pathes = [pathes[choose]]# compared one
            weights = [weights[choose]]# compared one
            return pathes, weights
        if mode == simulation.Sim_mode[1]:
            # choose = np.random.randint(low=0,high=len(weights), size=1)[0]
            choose = self.russian_roulette(weights)
            pathes = [pathes[choose]]# compared one
            weights = [weights[choose]]# compared one
            return pathes, weights
    def create_prior_belief_mat(self):
        # get path with weight
        ic(self.idid.get_num_ss())
        mat = np.random.rand(self.num_test, self.idid.get_num_ss())
        mat = normalize(mat, axis=1, norm='l1')
        print('-------------- prior belief matrix ------------------')
        self.prior_belief_mat = mat
        print(self.prior_belief_mat)
    def get_rewards(self,pathes_i, pathes_j, weights_i, weights_j):
        reward = list()
        path_i_pre = []
        for pi in range(0, len(pathes_i)):
            path_i = pathes_i[pi]
            weight_i = weights_i[pi]
            self.enter_evidences(self.idid.evidences, path_i,path_i_pre)
            path_j_pre = []
            for pj in range(0, len(pathes_j)):
                path_j = pathes_j[pj]
                weight_j = weights_j[pj]
                self.enter_evidences(['O' + ei for ei in self.idid.did.evidences], path_j,path_j_pre)
                for i in range(0, self.num_test):
                    # set the prior distribution of S
                    prior_belief = self.prior_belief_mat[i, :]
                    ei = "S" + str(self.get_horizon_max())
                    self.idid.net.set_node_definition(ei, prior_belief)
                    self.idid.net.update_beliefs()
                    # get utility
                    reward_path = list()
                    for hi in range(0, self.get_horizon_max()):
                        reward_path.append(np.mean(np.array(self.idid.net.get_node_value("U" + str(hi + 1)))))
                    # path prob
                    # prob_path = weight_i * weight_j
                    prob_path = 1.0
                    reward.append(np.sum(reward_path) * prob_path)
                # clear evidences
                path_j_pre = path_j
                # [self.idid.net.clear_evidence('O' + ei) for ei in self.idid.did.evidences]
            path_i_pre = path_i
            # [self.idid.net.clear_evidence(ei) for ei in self.idid.evidences]
        [self.idid.net.clear_evidence('O' + ei) for ei in self.idid.did.evidences]
        [self.idid.net.clear_evidence(ei) for ei in self.idid.evidences]
        return reward
    def enter_evidences(self,evidences,path,path_pre):
        if len(path_pre) ==0 or path_pre ==None:
            [self.idid.net.clear_evidence(ei) for ei in evidences]
            for ei_index in range(0, len(evidences)):
                ei = evidences[ei_index]
                self.idid.net.set_evidence(ei, int(path[ei_index]))
            self.idid.net.update_beliefs()
            # [self.idid.net.set_evidence(ei,int(path[self.idid.evidences.index(ei)])) for ei in self.idid.evidences]
        else:
            for ei_index in range(0,len(evidences)):
                if path[ei_index] != path_pre[ei_index]:
                    pos = ei_index
                    break
            for ei_index in range(pos, len(evidences)):
                ei = evidences[ei_index]
                self.idid.net.clear_evidence(ei)
            for ei_index in range(pos, len(evidences)):
                ei = evidences[ei_index]
                self.idid.net.set_evidence(ei, int(path[ei_index]))
            self.idid.net.update_beliefs()

# sim = simulation()
# sim.exp()
# # sim.set_log_mod(True)
# # sim.open_log_mod()
# # sim.simulate()
# # sim.display()
# # sim.close_log_mod()
# print('==================================')
# sim.all_mode_simulate()



