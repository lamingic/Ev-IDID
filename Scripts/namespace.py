# Author: Biyang Ma <biyangma@stu.xmu.edu.cn> <mabiyang001@hotmail.com>
# -*- coding: utf-8 -*-
# Copyright (C) 2001-2020
# Author: Biyang Ma <biyangma@stu.xmu.edu.cn> <mabiyang001@hotmail.com>
# builtins
import tkinter as tk
import numpy as np
# from namespace import Name,DomainParameters,Result,SimModels,ModelParameters,GA_ps,GAG_ps

class Name(object):
    def __init__(self):        
        self.Save_filepath = "./Results/"
        self.Elimit = 1 / np.power(10, 10)
        self.MAX = 20
        self.P_MAX = 20
        self.sufix_sc = '_sc'

        self.Postfix = "3.xdsl"
        self.Prefixs ={'DID':{0:'',1:'extn_',2:'gene_',3:'solv_',4:'extp_',6:'merg_'},\
                       'IDID': {0:'',1:'extn_',2:'gene_',3:'solvd_',4:'expa_',5:'solv_',6:'extp_',7:'merg_'}}
        self.Step = {1:'extending network',2:'generating evidence',3:'solving DID',\
                     4:'expanding DID',5:'solving network',6:'extending policy trees',7:'merging policy trees'}
        self.Steps = { 'DID':dict(),'IDID': dict((key,self.Step.get(key)) for key in self.Step.keys()) }
        self.RSteps = {'DID': dict(), 'IDID': dict((self.Step.get(key),key) for key in self.Step.keys())}
        index = 1
        for key in range(1,len(self.Step.keys())+1):
            if not self.Step.get(key).__contains__('DID'):
                self.Steps.get('DID')[index] = self.Step.get(key)
                self.RSteps.get('DID')[self.Step.get(key)] = index
                index = index +1
        self.MODPrefix = { 'DID':dict.fromkeys(self.Steps.get('DID').values(),'') ,'IDID': dict.fromkeys(self.Steps.get('IDID').values(),'') }
        self.MODPrefix['DID']['solving network'] = 'SMOD'
        self.MODPrefix['DID']['extending policy trees'] = 'EXMOD'
        self.MODPrefix['IDID']['solving network'] = 'SMOD'
        self.MODPrefix['IDID']['extending policy trees'] = 'EXMOD'
        self.MODPrefix['IDID']['expanding DID'] = 'MOD'
        self.Expansion_nodes = {'DID': dict.fromkeys(self.Steps.get('DID').values(), ''),\
                                'IDID': dict.fromkeys(self.Steps.get('IDID').values(), '')}
        self.Expansion_nodes['DID']['solving network'] = {'Node_D': 'D', 'Node_O': 'O'}
        self.Expansion_nodes['DID']['extending policy trees'] = {'Node_D': 'D', 'Node_O': 'O'}
        self.Expansion_nodes['IDID']['solving network'] = {'Node_D': 'D', 'Node_O': 'O'}
        self.Expansion_nodes['IDID']['extending policy trees'] = {'Node_D': 'D', 'Node_O': 'O'}
        self.Expansion_nodes['IDID']['expanding DID'] = {'Node_D': 'OD', 'Node_O': 'OO'}

        self.Solver_parameters = {'pointer','type','name', 'parameters', 'values','step','prestep'}
        self.Solver_types = {"DID":{'solving network':'','extending policy trees':''}, "IDID":{'solving network':'','extending policy trees':''}}
        self.Solver = {1: 'Exact', 2: 'GA', 3: 'PGA', 4: 'GGA',5:'MGA',6:'MPGA',7:'MGGA'}
        self.Extension = {1: '', 2: 'GA', 3: 'PGA', 4: 'GGA', 5: 'MDP', 6: 'MDT',7:'MGA',8:'MPGA',9:'MGGA'}
        self.Fitness_method = {1: 'distance', 2: 'diversity', 3: 'reward'}
        self.Group_criterion_method = {1: 'distance', 2: 'diversity', 3: 'reward'}
        self.Emigrate_method = {1:'Random',2:'Random-bound',3:'Sigmoid-bound',4:'Sigmoid-cutoff',5:'Sigmoid-group-cutoff'}
        self.Test_style = {1: 'Offline test', 2: 'Online test'}
        self.Sim_mode = {0:'all', 1: 'random_path', 2: 'most_possible_path', 3: 'tree'}
        self.Sim_mode_abrv = { 'random_path':'RP', 'most_possible_path':'MPP',  'tree':'T'}
        self.Domain = {1: 'Tiger', 2: 'UAV_IPTG', 3: 'UAV_ORG', 4: 'DUAV'}
        self.Result = {'policy_tree','policy_dict','reward','policy_path_weight','prior_belief','Plot'}
        self.domain_parameters = ['domain_name', 'num_mod_did', 'num_mod_idid', 'num_mod_test', 'horizon_size']
        self.ga_parameters = ["fitness_method","pop_size","generation_size",\
                          "tournament_size","crossover_rate","mutation_rate",\
                          "oddcross_mode","weight_mode","pelite_mode",\
                              "elite_mode","cover_mode"]
        self.gga_parameters = ["group_criterion_method","group_size","emigrate_rate","emigrate_method"]
        self.mga_parameters = ["genomes_size"]
        self.GUI_Tabs = ['Main','Plot','Diversity','Diversity','PolicyTree']
        self.GUI_Tabs_sc = ['','','',self.sufix_sc,'']
        self.GUI_Frames = ['Domain', 'Models','Simulation','Play','Meessage','Figure','Plot','Alg','Table','PolicyTree']
        cb =  ['Domain', 'Play']
        [cb.append(self.GUI_Tabs[ei]+self.GUI_Tabs_sc[ei]) for ei in range(1,len(self.GUI_Tabs))]
        self.CB_list_Domains = ['CB_'+ei for ei in cb]
        cb = ['Models', 'Play']
        [cb.append(self.GUI_Tabs[ei] + self.GUI_Tabs_sc[ei]) for ei in range(2, len(self.GUI_Tabs))]
        self.CB_list_Models = ['CB_' + ei for ei in cb]
        self.messeage_clues = ['coverage','plot_coverage','analysis','Plot','Alg','table','Diversity','PolicyTree']
        self.simmod_pt = ['Agent i\'s policy tree','Agent j\'s policy tree']
        self.table_files = ['Offline test','Online test','Runtime']
class DomainParameters(object):
    def __init__(self):
        self.pnames = Name()
        self.parameters = self.pnames.domain_parameters
        self.parameters_setting = {self.parameters[0]:list(self.pnames.Domain.values()),\
                                   self.parameters[1]:10,self.parameters[2]:1,\
                                   self.parameters[3]:5,self.parameters[4]:3 }
        self.Name = ''
        self.beliefs = {'DID':'','IDID':'','test':''}
        self.DBNS = {'DID':'','IDID':''}
        self.filepathes = {'DID':'','IDID':''}
        self.parameters_type =  dict()
        for key in self.parameters:
             self.parameters_type[key] = tk.IntVar()
        self.parameters_type[self.parameters[0]] = tk.StringVar()
        self.values = dict.fromkeys(self.parameters, '')
    def update(self):
        for key in self.parameters:
            self.values[key] = self.parameters_type.get(key).get()
            # print(key,self.values[key])
        for key in self.DBNS.keys():
            self.filepathes[key] = "./" + key + "/" + self.values[self.parameters[0]] + '/'
            from DBN import DBN
            self.DBNS[key] = DBN(self.filepathes[key], key,0)
        for key in self.beliefs.keys():
            if key.__contains__('test'):
                k = self.DBNS.get('IDID').num_ss
            else:
                k = self.DBNS.get(key).num_ss
            belief = dict()
            for i in range(0,self.values.get('num_mod_' + key.lower())):
                # belief[i] = normalize(np.random.rand(1, k), axis=1, norm='l1')[0]
                # print(belief[i])
                belief[i] = self.normalize(np.random.rand(1, k), copy=False)[0]
                # print(belief[i])
            self.beliefs[key] = belief
    def normalize(self,_d, copy=True):
        # d is a (n x dimension) np array
        d = _d if not copy else np.copy(_d)
        d /= np.sum(d)
        return d
    def Tostring(self):
        id_str = "_".join(list(map(str,list(self.values.values()))))
        # print(id_str)
        return id_str
class SimMod():
    def __init__(self):
        self.parameters = ['Agent i\'s policy tree','Agent j\'s policy tree']
        self.parameters_type = dict()
        self.records = dict()
        for key in self.parameters:
            self.parameters_type[key] = tk.StringVar()
            self.values = dict.fromkeys(self.parameters, '')
    def update(self):
        for key in self.parameters:
            self.values[key] = self.parameters_type.get(key).get()
            # print(key, self.values[key])
    def clear(self):
        self.records = dict()
    def add(self,test_style,sim_mode_i_set, sim_mode_j_set):
        self.records[test_style] = dict()
        self.records[test_style][self.parameters[0]] = sim_mode_i_set
        self.records[test_style][self.parameters[1]] = sim_mode_j_set
class Result():
    def __init__(self):
        self.pnames = Name()
        self.rewards = dict.fromkeys(self.pnames.Test_style.values(),'')
        for kt in self.rewards.keys():
            reward = dict()
            for mi in self.pnames.Sim_mode.values():
                for mj in self.pnames.Sim_mode.values():
                    if mi !='all' and  mj !='all':
                        key = mi+' vs '+mj
                        reward[key] = ''#{'mean': '', 'std': '', 'var': ''}
            self.rewards[kt]=reward
class SimModels():
    def __init__(self, domains , models):
        self.Domains = domains
        self.Moldels = models
        self.IDID = self.initialize()
        self.DID = dict.fromkeys(self.Domains.keys(), '')  # for online test
        self.base = dict()
        self.rearrange()
    def initialize(self):
        var = dict()
        for key in self.Domains.keys():
            dictm = dict.fromkeys(self.Moldels.keys(), '')
            var[key] =  dictm
        return var
    def rearrange(self):
        for keym in self.Moldels.keys():
            if self.Moldels.get(keym).Base_model == '':
                self.base[len(self.base)] = keym
        for keym in self.Moldels.keys():
            if self.Moldels.get(keym).Base_model != '':
                self.base[len(self.base)] = keym
class ModelParameters(object):
    def __init__(self):
        self.pnames = Name()
        self.parameters = self.pnames.Solver_types
        self.parameters_setting = dict()
        self.Name = ''
        self.Base_model = ''
        self.values = dict()
        self.initialize()
    def initialize(self):
        for key in self.parameters.keys():
            dictms = dict()
            for keyi in self.parameters.get(key).keys():
                dictm = dict.fromkeys(self.pnames.Solver_parameters, '')
                dictm['type'] = key  # DID IDID
                dictm['pointer'] = keyi  # solv or extend
                dictm['step'] = self.pnames.RSteps.get(key).get(keyi)
                if key =='DID':
                    dictm['prestep'] = self.pnames.RSteps.get(key).get('extending network')
                if key =='IDID':
                    dictm['prestep'] = self.pnames.RSteps.get(key).get('expanding DID')
                dictms[keyi] =dictm
            self.values[key] = dictms
    def default(self,name):
        # {'pointer', 'type', 'name', 'parameters', 'values', 'step', 'prestep'}
        for key in self.parameters.keys():
            dictms = dict()
            for keyi in self.parameters.get(key).keys():
                dictm = dict.fromkeys(self.pnames.Solver_parameters, '')
                dictm['type'] = key  # DID IDID
                if keyi == 'solving network':
                    dictm['name'] = self.pnames.Solver.get(1) # Exact
                dictm['pointer'] = keyi  # solv or extend
                dictm['step'] = self.pnames.RSteps.get(key).get(keyi)
                if key == 'DID':
                    dictm['prestep'] = self.pnames.RSteps.get(key).get('extending network')
                if key == 'IDID':
                    dictm['prestep'] = self.pnames.RSteps.get(key).get('expanding DID')
                dictms[keyi] = dictm
            self.values[key] = dictms
        self.Name = name
    def display(self):
        print(self.Name)
class GA_ps(object):
    def __init__(self):
        self.pnames = Name()
        self.parameters = self.pnames.ga_parameters
        self.values = dict.fromkeys(self.parameters, '')
        self.parameters_type = dict()
        self.update_dict()
        self.parameters_setting = dict()
        self.parameters_setting[self.parameters[0]] = list(self.pnames.Fitness_method.values())
        self.parameters_setting[self.parameters[1]] = 10
        self.parameters_setting[self.parameters[2]] = 50
        self.parameters_setting[self.parameters[3]] = 5
        self.parameters_setting[self.parameters[4]] = 0.8
        self.parameters_setting[self.parameters[5]] = 0.1
        self.parameters_setting[self.parameters[6]] = True
        self.parameters_setting[self.parameters[7]] = True
        self.parameters_setting[self.parameters[8]] = True
        self.parameters_setting[self.parameters[9]] = True
        self.parameters_setting[self.parameters[10]] = False
    def update_dict(self):
        for key in self.parameters:
            if key.__contains__('_size'):
                self.parameters_type[key] = tk.IntVar()
            if key.__contains__('_rate') or key.__contains__('_method'):
                self.parameters_type[key] = tk.StringVar()
            if key.__contains__('_mode'):
                self.parameters_type[key] = tk.BooleanVar()
    def update(self):
        for key in self.parameters:
            if key.__contains__('_rate'):
                self.values[key] = float(self.parameters_type.get(key).get())
            else:
                self.values[key] = self.parameters_type.get(key).get()
            # print(key,self.values[key])
    def Tostring(self):
        id_str = "".join(list(map(str,list(self.values.values()))))
        # print(id_str)
class GGA_ps(GA_ps):
    def __init__(self):
        super(GGA_ps, self).__init__()
        self.gga_parameters = self.pnames.gga_parameters
        for key in self.gga_parameters:
            self.parameters.append(key)
        self.update_dict()
        self.parameters_setting[self.gga_parameters[0]] = list(self.pnames.Group_criterion_method.values())
        self.parameters_setting[self.gga_parameters[1]] = 5
        self.parameters_setting[self.gga_parameters[2]] = 0.1
        self.parameters_setting[self.gga_parameters[3]] = list(self.pnames.Emigrate_method.values())
class MGA_ps(GA_ps):
    def __init__(self):

        super(MGA_ps, self).__init__()
        self.mga_parameters = self.pnames.mga_parameters
        for key in self.mga_parameters:
            self.parameters.append(key)
        self.parameters.pop(self.parameters.index(self.parameters[7]))
        self.update_dict()
        self.parameters_setting[self.mga_parameters[0]] =  200

class MGGA_ps(MGA_ps):
    def __init__(self):
        super(MGGA_ps, self).__init__()
        self.gga_parameters = self.pnames.gga_parameters
        for key in self.gga_parameters:
            self.parameters.append(key)
        self.update_dict()
        self.parameters_setting[self.gga_parameters[0]] = list(self.pnames.Group_criterion_method.values())
        self.parameters_setting[self.gga_parameters[1]] = 5
        self.parameters_setting[self.gga_parameters[2]] = 0.1
        self.parameters_setting[self.gga_parameters[3]] = list(self.pnames.Emigrate_method.values())

