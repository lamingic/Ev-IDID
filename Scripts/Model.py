# Author: Biyang Ma <biyangma@stu.xmu.edu.cn> <mabiyang001@hotmail.com>
# -*- coding: utf-8 -*-
# Copyright (C) 2001-2020
# Author: Biyang Ma <biyangma@stu.xmu.edu.cn> <mabiyang001@hotmail.com>
import datetime
from PolicyTree import PolicyTree
from DBN import DBN
from GA import MA,GA,GGA,PGA
from MGA import MGA,MGGA,MPGA
import tkinter as tk
from namespace import Name,DomainParameters,Result,SimModels,ModelParameters,GA_ps,GGA_ps
class Model(object):
    def __init__(self, domainParameters, modelParameters,type,scr_message = None):
        self.pnames = Name()
        self.type = type
        self.DomainParameters= domainParameters
        self.ModelParameters = modelParameters
        self.result = Result()
        self.times = dict.fromkeys(Name().Steps.get(self.type).values(), '')
        self.dbn = DBN(self.DomainParameters.filepathes.get(self.type),self.type,0,scr_message)
        self.dbn.result['prior_belief'] = dict()
        for key in self.DomainParameters.beliefs.get(self.type).keys():
            self.dbn.result['prior_belief'][key] = self.DomainParameters.beliefs.get(self.type).get(key)
        self.step = 1
        self.base_model = None
        if not scr_message is None:
            self.scr_message  = scr_message
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
    def set_type(self,type):
        self.type = type
    def next_step(self,start = None,end = None):
        step_name =  self.pnames.Steps[self.type][self.step]
        if start and (end is None):
            self.print('->>>>>>>> Start: '+  step_name)
            self.starttime = datetime.datetime.now()
            return True
        if end and not start:
            self.print('--------- Finish: '+  step_name)
            self.times[step_name] = datetime.datetime.now() - self.starttime
            return True
        if not end and start:
            self.step = self.step + 1
            step_name = self.pnames.Steps[self.type][self.step]
            self.print('->>>>>>>> Start: ' + step_name)
            self.starttime = datetime.datetime.now()
            return True
        if end and start:
            self.print('--------- Finish: '+  step_name)
            self.times[step_name] = datetime.datetime.now() - self.starttime
            self.step = self.step + 1
            step_name = self.pnames.Steps[self.type][self.step]
            self.print('->>>>>>>> Start: ' + step_name)
            self.starttime = datetime.datetime.now()
            return True
        if end and (start is None):
            self.print('--------- Finish: ' + step_name)
            self.times[step_name] = datetime.datetime.now() - self.starttime
            return True
    def random_create_policy_dict(self):
        for modi in range(0, self.DomainParameters.values.get('num_mod_' + self.type.lower())):
            self.print('> @ random sovling mod:' + str(modi))
            self.dbn.result.get('policy_dict')[modi] = self.dbn.gen_pathes()
    def load(self):# copy model
        if not self.base_model is None:
            self.print('>load model from base model')
            self.dbn.copy_result(self.base_model.dbn)
    def ga_solver(self,method,solver):
        result = self.dbn.result
        filepath = self.DomainParameters.filepathes.get(self.type)
        dstr = self.DomainParameters.Tostring()
        num_mod = self.DomainParameters.values.get('num_mod_' + self.type.lower())
        if method == self.pnames.Extension[2]:
            ga_alg = GA(solver, filepath, result,dstr,num_mod)
        if method == self.pnames.Extension[3]:
            ga_alg = PGA(solver,filepath, result,dstr,num_mod)
        if method == self.pnames.Extension[4]:
            ga_alg = GGA(solver,filepath, result,dstr,num_mod)
        if method == self.pnames.Extension[5] or method == self.pnames.Extension[6]:
            ga_alg = MA(solver, filepath, result,dstr,num_mod)
        if method == self.pnames.Extension[7]:
            ga_alg = MGA(solver,filepath, result,dstr,num_mod)
        if method == self.pnames.Extension[8]:
            ga_alg = MPGA(solver,filepath, result,dstr,num_mod)
        if method == self.pnames.Extension[9]:
            ga_alg = MGGA(solver,filepath, result,dstr,num_mod)
        ga_alg.set_alg_name(self.ModelParameters.Name )
        ga_alg.ga_main()
        self.dbn.result = ga_alg.result
    def solve_mod(self):
        step_name = self.pnames.Steps[self.type][self.step]
        method = self.ModelParameters.values.get(self.type).get(step_name).get('name')
        # self.print(method)
        # self.print(step_name)
        if method == self.pnames.Solver[1]:# call exact
            self.dbn.exact_solver()
        if method != self.pnames.Solver[1]:
            solver = self.ModelParameters.values.get(self.type).get(step_name)
            self.random_create_policy_dict()
            self.ga_solver(method,solver)# call ga
    def extend(self):# extend policy tree
        step_name = self.pnames.Steps[self.type][self.step]
        method = self.ModelParameters.values.get(self.type).get(step_name).get('name')
        # self.print(method)
        # self.print(step_name)
        if method=='':
            return 'pass'# not extend
        if method != '': # GA GAG
            solver = self.ModelParameters.values.get(self.type).get(step_name)
            self.ga_solver(method,solver)
    def expansion(self, expansion_flag,policy_tree= None):
        if expansion_flag:
            self.step_sim = self.step
            step_name = self.pnames.Steps[self.type][self.step]
            solver = self.ModelParameters.values.get(self.type).get(step_name)
            filepath = self.DomainParameters.filepathes.get(self.type)
            self.dbn_sim = DBN(filepath, solver.get('type'), solver.get('prestep'))
            self.dbn_sim.expansion(solver.get('step'), expansion_flag=True)
            self.dbn_sim.result = self.dbn.result
            self.dbn_sim.evidences =self.dbn.evidences
        else:
            self.dbn_sim.expa_policy_tree = policy_tree
            # self.dbn_sim.result['policy_tree'] = policy_tree
            self.dbn_sim.expansion(self.step_sim, expansion_flag=False)
            # r, w, p = self.dbn.get_reward()
    def gen_pathes(self):
        self.print('\n')
        self.print('Building Model:  '+self.ModelParameters.Name+ '>>>>>@ DID/IDID type: ' + self.type)
        self.next_step(start=True)
        self.dbn.extend(self.DomainParameters.values['horizon_size'],self.step)

        self.next_step(start=True,end=True)
        self.dbn.generate_evidence(self.step)

        self.next_step(start=True, end=True)
        self.solve_mod()
        self.load()

        self.next_step(start=True, end=True)
        self.extend()

        self.next_step(start=True,end=True)
        self.dbn.result.get('policy_tree').gen_policy_trees_memorysaved()
        self.dbn.result.get('policy_tree').set_name( self.DomainParameters.Tostring()+ '-'+self.ModelParameters.Name +'-'+str(self.step)+'-The Merged Policy Tree of '+  self.type )
        # self.dbn.result['policy_tree'].save_policytree(self.pnames.Save_filepath)
        self.next_step(end=True)
        self.print('Finish Model:  ' + self.ModelParameters.Name + '>>>>>@ DID/IDID type: ' + self.type)
        self.print('\n')