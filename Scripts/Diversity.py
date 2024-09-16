# -*- coding: utf-8 -*-
# Copyright (C) 2001-2020
# Author: Biyang Ma <biyangma@stu.xmu.edu.cn> <mabiyang001@hotmail.com>
# builtins

import numpy as np
class Diversity(object):
    def __init__(self, policy_dict):
        self.policy_dict = policy_dict
        self.triangles = dict()
        self.genome_template  = None
        self.get_horizon()
        self.get_num_path()
        self.get_num_os()
        self.get_len_path()
        self.get_tree_template()
        self.triangles_horizon = dict()
        self.sub_genomes_abc = dict()
        # print('num os:' + str(self.num_os))
        # print('horizon: ' + str(self.horizon))
        self.calculate_diversity()
    def get_num_path(self):
        if not (self.policy_dict == None or len(self.policy_dict) == 0):
            for key in  self.policy_dict.keys():
                pathes = self.policy_dict.get(key)
                break
            self.num_path = len(pathes)
        return self.num_path
    def get_tree_template(self):
        if not (self.policy_dict == None or len(self.policy_dict) == 0):
            for key in  self.policy_dict.keys():
                self.tree_template = self.policy_dict.get(key)
                break
    def get_len_path(self):
        if not (self.policy_dict == None or len(self.policy_dict) == 0):
            for key in self.policy_dict.keys():
                pathes = self.policy_dict.get(key)
                break
            self.len_path = len(pathes[0])
        return self.len_path
    def get_horizon(self):
        self.get_len_path()
        self.horizon = int((self.len_path + 1) / 2)
        return self.horizon
    def get_num_os(self):
        y = self.get_num_path()
        x = self.get_horizon()-1
        self.num_os = int(np.power(10,np.log10(y)/x))
        return self.num_os
    def pathes_to_genome(self, pathes):
        genome = list()
        for j in range(0, self.get_len_path(), 2):
            hi = int(self.get_horizon() - (j / 2))
            step = np.power(self.get_num_os(), hi - 1)
            for rw in range(0, self.get_num_path(), step):
                genome.append(int(pathes[rw][j]))
        if self.genome_template != None:
            pass
        else:
            self.genome_template = genome
        return genome
    def gen_genome_level(self):
        if self.genome_template == None:
            return -1
        genome_level = [0 for i in range(0, len(self.genome_template))]
        start = 0
        for cl in range(0, self.get_horizon(), 1):
            step = np.power(self.get_num_os(), self.get_horizon() - cl - 1)
            num = int(self.get_num_path() / step)
            for i in range(start, start + num):
                genome_level[i] = self.get_horizon() - cl
            start = start + num
        self.genome_level = genome_level
    def gen_genome_arc(self):
        genome_arc = [-1 for i in range(0, len(self.genome_template))]
        for cl in range(0, self.get_horizon() - 1):
            step = np.power(self.get_num_os(), cl)
            num = int(self.get_num_path() / step)
            ind = range(0, num + 1, self.get_num_os())
            start = self.genome_level.index(cl + 1)
            parents_start = self.genome_level.index(cl + 2)
            for i in range(0, len(ind) - 1):
                for j in range(start + ind[i], start + ind[i + 1]):
                    genome_arc[j] = parents_start + i
        self.genome_arc = genome_arc
    def create_genomes(self):
        # create genomes
        self.genomes = dict()
        for key in self.policy_dict.keys():
            pathes = self.policy_dict.get(key)
            genome = self.pathes_to_genome(pathes)
            self.genomes[key] = genome
    def gen_genomes(self):
        self.create_genomes()
        self.gen_genome_level()
        self.gen_genome_arc()

    def calculate_diversity(self):
        self.sub_genomes = dict()
        self.sub_genomes_total = dict()
        self.gen_genomes()
        self.diversity_rate = dict()
        popindex = range(0, len(list(self.policy_dict.keys())))
        for modi in self.policy_dict.keys():
            gen  = self.genomes.get(modi)
            subtree = self.gen_genome_subtree(gen)
            self.sub_genomes[modi] = subtree
        self.diversity_pop = self.cal_diversity(popindex)
        for modi in self.policy_dict.keys():
            gen = self.genomes.get(modi)
            diversity_pop_gi = self.cal_diversity(popindex, modi)
            fits = diversity_pop_gi / self.diversity_pop  # divide
            self.diversity_rate[modi] = fits
        self.Columns = ['']
        for s in self.iter_all_strings():
            self.Columns.append(s)
            if len(self.Columns) == len(list(self.sub_genomes_total.keys()))+1:
                break
        # print('sub_genomes_total:')
        for ei in self.sub_genomes_total.keys():
            pathes,horizon = self.genome_to_pathes(ei)
            id = self.sub_genomes_total.get(ei)
            self.triangles[id] = pathes
            # print(ei,id)
            # print(pathes)
            self.triangles_horizon[id] = horizon
        # print('=============')
        # for ei in self.triangles.keys():
        #     print(ei)
        #     print(self.triangles.get(ei))
        # print('sub_genomes:')
        for modi in self.policy_dict.keys():
            # print(self.sub_genomes[modi])
            sub_genomes = [self.Columns[ei] for ei in self.sub_genomes[modi]]
            self.sub_genomes_abc[modi] = sub_genomes
            # print(self.sub_genomes_abc[modi])
        # print('=============')
    def genome_to_pathes(self, genome_str):
        # print('--------------')
        genome =[int(ei) for ei in  genome_str.split('|')]
        # print(genome)
        # print(self.genome_level)
        cl = self.genome_level[len(genome)-1]
        # print(cl)
        pathes = list()
        num_path = np.power(self.get_num_os(),(self.get_horizon()-cl))
        h = 2*(self.get_horizon()-cl)+1
        for rw in range(0, num_path):
            path = [int(ei) for ei in self.tree_template[rw][0:h]]
            pathes.append(path)
        # print(pathes)
        # print('--------------')
        horizon = self.get_horizon()-cl+1
        for cl in range(1, h, 1):
            elements = range(0, self.get_num_os())
            copy = int(num_path/np.power(self.get_num_os(),cl))
            column = [e for e in elements for i in range(0, copy)]
            for i in range(0, len(column)):
                pathes[i][cl] = int(column[i])
        ind_start = 0
        for cl in range(0, horizon, 1):
            step = np.power(self.get_num_os(), cl)
            copy = np.power(self.get_num_os(), horizon - cl - 1)
            ind_end = int(ind_start + step)
            elements = [genome[i] for i in range(ind_start, ind_end)]
            ind_start = ind_end
            column = [e for e in elements for i in range(0, copy)]
            for i in range(0, len(column)):
                pathes[i][cl * 2] = int(column[i])
        # print(pathes)
        return pathes,horizon
    def iter_all_strings(self):
        from string import ascii_lowercase
        import itertools
        for size in itertools.count(1):
            for s in itertools.product(ascii_lowercase, repeat=size):
                yield "".join(s)
    def sub_genomes_total_check(self,tree):
        # optimize it by length
        if not self.sub_genomes_total.__contains__(tree):
            value = len(self.sub_genomes_total) + 1
            self.sub_genomes_total[tree] = value
        else:
            value = self.sub_genomes_total.get(tree)
        return value
    def gen_genome_subtree(self, gen):
        subtree = set()
        for gi in range(0, len(gen)):
            cl = self.genome_level[gi]
            if cl <= 1:
                break
            tree=str(gen[gi])
            subtree.add(self.sub_genomes_total_check(tree))
            parents = [gi]
            while cl > 0:
                children = list()
                for pa in parents:
                    for gj in range(gi, len(gen)):
                        if self.genome_arc[gj] == pa:
                            children.append(gj)
                            tree = tree + '|' + str(gen[gj])
                subtree.add(self.sub_genomes_total_check(tree))
                parents = children
                # [parents.append(ch) for ch in  children]
                cl = cl - 1
        # print(subtree)
        return subtree
    def cal_diversity(self, popindex, gi=None):
        if gi != None:
            len_pop = len(popindex) - 1
        else:
            len_pop = len(popindex)
        sub_genomes = set()
        sub_genomes_gi_size = list()
        for gj in popindex:
            if gj == gi:
                continue
            subtree = self.sub_genomes.get(gj)
            sub_genomes_gi_size.append(len(subtree))
            sub_genomes = sub_genomes.union(subtree)
        if len(popindex) ==1:
            diversity =1
        else:
            diversity = len_pop * len(sub_genomes) / (np.sum(np.array(sub_genomes_gi_size)))
        # print(diversity)
        return diversity


            
