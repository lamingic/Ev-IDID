# Author: Biyang Ma <biyangma@stu.xmu.edu.cn> <mabiyang001@hotmail.com>
# builtins
import random
import math
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# import os
# import sys

# self package
from PolicyTree import PolicyTree
from DBN import DBN
from namespace import Name
class Gene:
    def __init__(self, **data):
        self.__dict__.update(data)
class MGA(object):
    def __init__(self, solver,filepath,result,dstr,num_mod,type = None):
        if type is None:
            self.type = 'MGA'
        else:
            self.type = type
        print('Initializing ' + self.type)
        self.plot_flag = False  # for GUI GEN EXE
        self.domain = dstr
        self.pnames = Name()
        self.num_mod = num_mod
        self.solver = solver
        self.step = solver.get('step')
        self.parameters = solver.get('parameters')
        self.dbn = DBN(filepath, solver.get('type'),solver.get('prestep'))
        self.dbn.expansion(solver.get('step'),expansion_flag=True)  # just set the node and arc, the sates and cpt is set in evaluation
        self.result = result
        self.dbn.result['prior_belief'] = dict()
        for key in result['prior_belief'].keys():
            self.dbn.result['prior_belief'][key] = result['prior_belief'].get(key)
        self.check_identity_flag = True
        self.progressbar = True
        self.initialize()
        self.initialize_pop()
        self.alg_name = ''
    def set_alg_name(self,name):
        self.alg_name = name
    # initialize
    def initialize(self):
        self.genomes = dict()
        self.genomes_slices = list()
        self.genome_template = None
        self.fits = {'max': list(), 'mean': list(), 'min': list(), 'std': list()}

        self.len_path = self.get_len_path()
        self.num_path = self.get_num_path()
        self.fill_tree_template()
        self.group_criterion = list()

        self.sub_genomes_total = dict()
        self.sub_genomes = dict()
        self.initialize_other()
    def get_num_path(self):
        if not (self.result['policy_dict'] == None or len(self.result['policy_dict']) == 0):
            pathes = self.result['policy_dict'].get(0)
            self.num_path = len(pathes)
        return self.num_path
    def get_len_path(self):
        if not (self.result['policy_dict'] == None or len(self.result['policy_dict']) == 0):
            pathes = self.result['policy_dict'].get(0)
            self.len_path = len(pathes[0])
        return self.len_path
    def fill_tree_template(self):
        if not (self.result['policy_dict'] == None or len(self.result['policy_dict']) == 0):
            for key in self.result['policy_dict'].keys():
                self.tree_template = self.result['policy_dict'].get(key)
                break
        # print(self.tree_template)
    def initialize_other(self):
        pass
    def initialize_pop(self):
        # initialise popluation
        self.gen_ind_template()
        self.gen_genomes()
        self.gen_pop()
        self.evaluate()  # evaluate the distance or diversity of pop
        self.bestindividual = self.selectBest(self.pop)  # store the best chromosome in the population
    def gen_ind_template(self):
        self.ind_template = {'Gene': Gene(data=[]), 'fitness': 0, 'id': 0}
    def gen_genomes(self):
        print('------ gen_genomes --------')
        self.get_tree_size()
        self.get_num_slice()
        self.create_genomes()
    def get_tree_size(self):
        tree_size = list()
        for cl in range(self.dbn.horizon, 0,-1):
            tree_size.append( np.power(self.dbn.num_os, self.dbn.horizon-cl))
        self.tree_size = int(sum(tree_size))
    def get_num_slice(self):
        self.genomes_size = self.parameters.values.get('genomes_size')
        if self.tree_size>self.genomes_size:
            self.slice_size = int(math.ceil(self.tree_size/self.genomes_size))
            self.post_null = int(self.slice_size*self.genomes_size - self.tree_size)
            while self.post_null>self.slice_size:
                self.post_null = self.post_null  - self.slice_size
                self.genomes_size = self.genomes_size -1
        else:
            self.genomes_size = self.tree_size
            self.slice_size = 1
            self.post_null = 0
        print('genomes_size:'+ str(self.genomes_size))
        print('slice_size:' + str(self.slice_size))
        print('tree_size:' + str(self.tree_size))
        print('post_null:' + str(self.post_null))
    def create_genomes(self):
        len_key = len(self.result['policy_dict'].keys())
        pop_size = self.parameters.values.get('pop_size')
        if  len_key < pop_size:
            print(len_key, pop_size)
            for i in range(len_key, pop_size):
                index = random.randint(0, len_key - 1)
                self.result['policy_dict'][i] = self.result['policy_dict'].get(index)
        # create genomes
        for key in self.result['policy_dict'].keys():
            pathes = self.result['policy_dict'].get(key)
            genome = self.pathes_to_genome(pathes)
            self.genomes[key] = genome
            # print(genome)
    def pathes_to_genome(self, pathes):
        '''
        a	o	a	o	a        a	a  a
        2	0	0	0	2        2  0  2
        2	0	0	1	2  = >   2  0  2 => 2 0 2 2 2 2 0
        2	1	2	0	2        2  2  2
        2	1	2	1	0        2  2  0
        '''
        genome = list()
        genome_slice = list()
        for j in range(0, self.get_len_path(), 2):
            hi =int(self.dbn.horizon - (j / 2))
            step = np.power(self.dbn.num_os, hi - 1)
            for rw in range(0, self.get_num_path(), step):
                genome_slice.append(int(pathes[rw][j]))
                if len(genome_slice) == self.slice_size:
                    if not self.genomes_slices.__contains__(genome_slice):
                        self.genomes_slices.append(genome_slice)
                    index =  self.genomes_slices.index(genome_slice)
                    genome.append(index)
                    genome_slice = list()
        if len(genome_slice) < self.slice_size and len(genome_slice)!=0:
            [genome_slice.append(int(random.randint(0, self.dbn.num_as - 1))) for i in range(0,self.post_null)]
            if not self.genomes_slices.__contains__(genome_slice):
                self.genomes_slices.append(genome_slice)
                genome.append(len(self.genomes_slices) - 1)
            else:
                index = self.genomes_slices.index(genome_slice)
                genome.append(index)
        if len(genome)!=self.genomes_size:
            print('errrrrrrrrrrrrrrrrrrrr-----------genome')
            print(len(genome),self.genomes_size)
        if self.genome_template != None:
            pass
        else:
            self.genome_template = genome
        return genome
    def genome_to_pathes(self, genome):
        pathes = list()
        for rw in range(0, self.num_path):
            path = [int(ei) for ei in self.tree_template[rw]]
            pathes.append(path)
            # pathes.append(self.tree_template[rw])
        index = 0
        gen = self.genomes_slices[genome[index]]
        for cl in range(0, self.dbn.horizon, 1):
            step = np.power(self.dbn.num_os, cl)
            copy = np.power(self.dbn.num_os, self.dbn.horizon - cl - 1)
            while step> len(gen):
                if len(genome) != self.genomes_size:
                    print('errrrrrrrrrrrrrrrrrrrr---22-----genome')
                    print(len(genome), self.genomes_size)
                index = index+1
                if index<len(genome):
                    gen_n = self.genomes_slices[genome[index]]
                    [gen.append(gen_n[i]) for i in range(0,self.slice_size)]
                else:
                    break
            elements = [gen[i] for i in range(0, step)]
            gen = gen[step:len(gen)]
            column = [e for e in elements for i in range(0, copy)]
            for i in range(0,len(column)):
                pathes[i][cl * 2] = column[i]
        return pathes
    def gen_pop(self):
        print('------ gen_pop --------')
        pop = []
        for key in self.genomes.keys():
            geneinfo = self.genomes.get(key)
            fits = self.result['reward'].get(key)
            ind = self.gen_ind(Gene(data=geneinfo))
            pop.append(ind)  # store the chromosome and its fitness
        self.geneinfo_dim = len(geneinfo)
        self.pop = [ind for ind in pop]
        self.pop_init = [ind for ind in pop]
    def evaluate(self):
        if self.parameters.values.get('fitness_method') == self.pnames.Fitness_method.get(1):
            self.evaluate_distance()
        if self.parameters.values.get('fitness_method') == self.pnames.Fitness_method.get(2):
            self.evaluate_diversity()
        if self.parameters.values.get('fitness_method') == self.pnames.Fitness_method.get(3):
            self.evaluate_reward()
    def evaluate_distance(self):
        # evaluate the distance of pop
        genome = self.genome_template
        genome = [0 for i in range(0, len(genome))]
        for ind in self.pop:
            gen = ind['Gene'].data
            genome = [genome[i] + gen[i] for i in range(0, len(gen))]
        genome = [g / len(self.pop) for g in genome]
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            gen = ind['Gene'].data
            fits = np.sqrt(np.sum([np.power(genome[j] - gen[j], 2) for j in range(0, len(gen))]))  # np.array
            ind['fitness'] = fits
            # print(fits)
            self.pop[gi] = ind
    def evaluate_diversity(self):
        # evaluate the diversity of pop
        # print('evaluate the diversity of pop')
        popindex = range(0, len(self.pop))
        diversity_pop = self.cal_diversity(popindex)
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            diversity_pop_gi = self.cal_diversity(popindex,gi)
            fits = diversity_pop_gi / diversity_pop  # divide
            ind['fitness'] = fits
            # print(fits)
            self.pop[gi] = ind
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
            subtree =self.pop[gj]['Gene'].data
            sub_genomes_gi_size.append(len(set(subtree)))
            sub_genomes = sub_genomes.union(subtree)
        diversity = len_pop * len(sub_genomes) / (np.sum(np.array(sub_genomes_gi_size)))
        # print(diversity)
        return diversity
    def evaluate_reward(self):
        # print('------ evaluate_reward --------')
        # evaluate the reward of pop
        # genome = self.genome_template
        # genome = [0 for i in range(0, len(genome))]
        for gi in range(0, len(self.pop)):
            # print(gi)
            ind = self.pop[gi]
            genome = ind['Gene'].data
            pathes = self.genome_to_pathes(genome)
            policy_dict = dict()
            policy_dict[1] = pathes
            policy_tree = PolicyTree('The Policy Tree of '+  self.type,self.dbn.action_list, self.dbn.observation_list)
            policy_tree.set_policy_dict(policy_dict)
            policy_tree.gen_policy_trees_memorysaved()
            # policy_tree.save_policytree(self.pnames.Save_filepath)
            self.dbn.expa_policy_tree = policy_tree
            self.dbn.expansion(self.step,expansion_flag=False)
            if self.parameters.values.get('cover_mode'):
                ind['fitness'], w, p = self.dbn.get_reward(weight_off=True, modi=gi)
            else:
                ind['fitness'], w, p = self.dbn.get_reward(weight_off=True)
            # ind['fitness'], w, p = self.dbn.get_reward(weight_off = True)
            self.pop[gi] = ind
    # ga_main
    def ga_main(self):
        print("Starting evolution")
        # Begin the evolution
        if self.progressbar:
            bar = tqdm(total=int(self.parameters.values.get('generation_size')))
        for g in range(self.parameters.values.get('generation_size')):
            if not self.progressbar:
                print("-- Generation %i --" % g)
            # Apply selection based on their converted fitness
            selectpop = self.selection(self.pop, self.parameters.values.get('pop_size'))
            nextoff = []
            while len(nextoff) != self.parameters.values.get('pop_size') :
                # Apply crossover and mutation on the offspring
                # Select two individuals
                offspring = [random.choice(selectpop) for i in range(0, 2)]
                if random.random() < self.parameters.values.get('crossover_rate'):  # cross two individuals with probability CXPB
                    crossoff = self.crossoperate(offspring)
                    if random.random() < self.parameters.values.get('mutation_rate'):  # mutate an individual with probability MUTPB
                        muteoff = self.mutation(crossoff)
                        if self.check_identity_flag:
                            if not self.check_identity(nextoff, muteoff) and not self.check_identity(self.pop, muteoff)and not self.check_identity(self.pop_init, muteoff):
                                ind = self.gen_id(nextoff, self.gen_ind(muteoff))
                                nextoff.append( ind )  # initialize
                        else:
                            ind = self.gen_id(nextoff, self.gen_ind(muteoff))
                            nextoff.append(ind)  # initialize
            # The population is entirely replaced by the offspring
            # self.next_pop(nextoff)
            if g == self.parameters.values.get('generation_size') - 1:
                self.next_pop(nextoff, final_pop=True)
            else:
                self.next_pop(nextoff)
            # Gather all the fitnesses in one list and print the stats
            fits = [ind['fitness'] for ind in self.pop]
            length = len(self.pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5
            best_ind = self.selectBest(self.pop)
            if best_ind['fitness'] > self.bestindividual['fitness']:
                self.bestindividual = best_ind
            if not self.progressbar:
                print(
                    "Best individual found is %s, %s" % (self.bestindividual['Gene'].data, self.bestindividual['fitness']))
                print("  Min fitness of current pop: %s" % min(fits))
                print("  Max fitness of current pop: %s" % max(fits))
                print("  Avg fitness of current pop: %s" % mean)
                print("  Std of currrent pop: %s" % std)
            else:
                bar.update(1)
            self.fits['max'].append(max(fits))
            self.fits['min'].append(min(fits))
            self.fits['mean'].append(mean)
            self.fits['std'].append(std)
        if self.progressbar:
            bar.close()
        print("-- End of evolution --")
        # self.next_pop(nextoff, final_pop=True)
        # if g == self.parameters.values.get('generation_size') - 1:
        #     self.next_pop(nextoff, final_pop=True)
        # else:
        #     self.next_pop(nextoff)
        self.choose_group()
        gens = [ind['Gene'].data for ind in self.pop]
        for gen in gens:
            print(gen)
        self.gen_other()
        if self.plot_flag:
            self.plot_fits()
        else:
            self.result['Plot'][self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer')] = dict()
            self.plot_fits_data()
        # ==================================================================================================
    def selection(self, individuals, k, gp=None):
        # select two individuals from pop
        # sort the pop by the reference of 1/fitness
        individuals = self.selection_group(individuals, gp)
        # print(len(individuals))
        s_inds = sorted(individuals, key=itemgetter("fitness"), reverse=True)
        min_fits = np.inf
        for ind in individuals:
            if ind['fitness'] < min_fits:
                min_fits = ind['fitness']
        # print(np.abs(min_fits)+ self.pnames.Elimit)
        min_fits = np.abs(min_fits) + self.pnames.Elimit
        sum_fits = sum(min_fits + ind['fitness'] for ind in individuals)  # sum up the fitness of the whole pop

        chosen = []
        for i in range(0, k):
            u = random.random() * sum_fits  # randomly produce a num in the range of [0, sum_fits]
            sum_ = 0
            for ind in s_inds:
                sum_ += min_fits + ind['fitness']  # sum up the fitness
                if sum_ > u:
                    # when the sum of fitness is bigger than u, choose the one, which means u is in the range of [sum(1,2,...,n-1),sum(1,2,...,n)] and is time to choose the one ,namely n-th individual in the pop
                    chosen.append(ind)
                    break
        # for ind in chosen:
        #     print(ind['id'])
        return chosen
    def selection_group(self, individuals,gp=None):
        return individuals
    def crossoperate(self, offspring):
        dim = len(offspring[0]['Gene'].data)
        geninfo1 = offspring[0]['Gene'].data  # Gene's data of first offspring chosen from the selected pop
        geninfo2 = offspring[1]['Gene'].data  # Gene's data of second offspring chosen from the selected pop
        # pos1 = random.randrange(1, dim)  # select a position in the range from 0 to dim-1,
        # pos2 = random.randrange(1, dim)
        pos1 = self.rand_pointer()  # select a position in the range from 0 to dim-1,
        pos2 = self.rand_pointer()
        newoff = Gene(data=[])  # offspring produced by cross operation
        temp = []
        if self.parameters.values.get('oddcross_mode'):
            for i in range(dim):
                if i % 2 == 1:
                    if (i >= min(pos1, pos2) and i <= max(pos1, pos2)):
                        temp.append(geninfo1[i])
                    else:
                        temp.append(geninfo2[i])
                if i % 2 == 0:
                    if (i >= min(pos1, pos2) and i <= max(pos1, pos2)):
                        temp.append(geninfo2[i])
                    else:
                        temp.append(geninfo1[i])
        else:
            for i in range(dim):
                if (i >= min(pos1, pos2) and i <= max(pos1, pos2)):
                    temp.append(geninfo2[i])
                    # the gene data of offspring produced by cross operation is from the second offspring in the range [min(pos1,pos2),max(pos1,pos2)]
                else:
                    temp.append(geninfo1[i])
                    # the gene data of offspring produced by cross operation is from the frist offspring in the range [min(pos1,pos2),max(pos1,pos2)]
        newoff.data = temp
        return newoff
    def rand_pointer(self):
        pos = random.randrange(1, self.geneinfo_dim)  # chose a position in crossoff to perform mutation.
        return pos
    def mutation(self, crossoff):
        pos = self.rand_pointer()  # chose a position in crossoff to perform mutation.
        crossoff.data[pos] = random.randint(0, len(self.genomes_slices) - 1)
        # gen = crossoff.data[pos]
        # posn = random.randrange(1, len(gen))
        # gen[posn] = random.randint(0, self.dbn.num_as - 1)
        # if not self.genomes_slices.__contains__(gen):
        #     self.genomes_slices.append(gen)
        # index = self.genomes_slices.index(gen)
        # crossoff.data[pos] = index
        return crossoff
    def check_identity(self, pop, individual):
        # if the individual is already in the pop, then we don't need to add a copy of it
        gens = [ind['Gene'].data for ind in pop]
        # print(gens)
        for gen in gens:
            sum = 0
            for gi in range(0, len(gen)):
                sum = sum + np.abs(gen[gi] - individual.data[gi])
            if sum == 0:
                return True
        return False
    def gen_ind(self,muteoff):
        ind = dict()
        for key in self.ind_template:
            ind[key] = self.ind_template.get(key)
        ind['Gene'] = muteoff
        return ind
    def gen_id(self,nextoff,ind,gp=None):
        return ind
    def next_pop(self, nextoff,final_pop = None):
        if self.parameters.values.get('pelite_mode'):
            pop_temp = [ind for ind in self.pop]
            self.pop = [ind for ind in nextoff]
            self.evaluate()
            [self.pop.append(ind) for ind in pop_temp]
            self.select_nextpop(self.parameters.values.get('pop_size'))
        else:
            self.pop = [ind for ind in nextoff]
            self.evaluate()
        if self.solver.get('pointer') == 'solving network':
            if final_pop is None:
                if self.parameters.values.get('elite_mode'):
                    if self.parameters.values.get('tournament_size') >= self.num_mod:
                        self.select_nextpop(self.parameters.values.get('tournament_size'))
                    else:
                        self.select_nextpop(self.num_mod)
            else:
                self.select_nextpop(self.num_mod)
        else:
            if self.parameters.values.get('elite_mode'):
                self.select_nextpop(self.parameters.values.get('tournament_size'))
    def select_nextpop(self,size):
        s_inds = sorted(self.pop, key=itemgetter("fitness"), reverse=True)
        nextpop=[]
        count = 0
        for ind in s_inds:
            nextpop.append(ind)  # store the chromosome and its fitness
            count = count + 1
            if count== size:
                break
        self.pop = []
        self.pop = [ind for ind in nextpop]
    def selectBest(self, pop):
        # select the best individual from pop
        s_inds = sorted(pop, key=itemgetter("fitness"), reverse=True)
        return s_inds[0]
    def choose_group(self):
        pass
    def gen_other(self):# at
        if self.solver.get('pointer') == self.pnames.Step.get(6):
            num_mod =  len(self.result['policy_path_weight'].keys())
            for key in self.pnames.Result:
                if not key.__contains__('policy_tree') and not key.__contains__('policy_path_weight'):
                    for ki in self.result.get(key).keys():
                        if ki>=num_mod:
                            self.result.get(key).pop(ki)
            weights, priors,rewards,policy_dicts = self.gen_weight_prior()
            # print(num_mod,len(weights))
            for gi in range(0, len(weights)):
                self.result['policy_path_weight'][num_mod + gi] = weights[gi]
                self.result['prior_belief'][num_mod + gi] = priors[gi]
                self.result['reward'][num_mod + gi] = rewards[gi]
                self.result['policy_dict'][num_mod + gi] = policy_dicts[gi]
        else:
            for key in self.pnames.Result:
                if not key.__contains__('policy_tree'):
                    self.result[key] = dict()
            weights, priors, rewards, policy_dicts = self.gen_weight_prior()
            for gi in range(0, len(weights)):
                self.result['policy_path_weight'][gi] = weights[gi]
                self.result['prior_belief'][gi] = priors[gi]
                self.result['reward'][gi] = rewards[gi]
                self.result['policy_dict'][ gi] = policy_dicts[gi]
        self.result['policy_tree'] = PolicyTree( self.domain+ '-'+self.alg_name+ '-'+str(self.step)+ '-The Policy Tree of '+  self.type + ' for '+ self.solver.get('type') + ' @ '+ self.solver.get('pointer') , self.dbn.action_list, self.dbn.observation_list)
        self.result['policy_tree'].set_policy_dict(self.result['policy_dict'])
        self.result['policy_tree'].gen_policy_trees_memorysaved()
        # self.result['policy_tree'].save_policytree(self.pnames.Save_filepath)
    def gen_weight_prior(self):
        genome = self.genome_template
        genome = [0 for i in range(0, len(genome))]
        weights = list()
        priors = list()
        rewards = list()
        policy_dicts = list()
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            genome = ind['Gene'].data
            pathes = self.genome_to_pathes(genome)
            if self.parameters.values.get('fitness_method') == self.pnames.Fitness_method.get(3):
                policy_dict = dict()
                policy_dict[1] = pathes
                policy_tree = PolicyTree('The Policy Tree of '+self.type,self.dbn.action_list, self.dbn.observation_list)
                policy_tree.set_policy_dict(policy_dict)
                policy_tree.gen_policy_trees_memorysaved()
                # policy_tree.save_policytree(self.pnames.Save_filepath)
                self.dbn.expa_policy_tree = policy_tree
                self.dbn.expansion(self.step,expansion_flag=False)
                # rw, w,p = self.dbn.get_reward(weight_off =True)
                if self.parameters.values.get('cover_mode'):
                    rw, w,p= self.dbn.get_reward(weight_off=True, modi=gi)
                else:
                    rw, w,p = self.dbn.get_reward(weight_off=True)
            else:
                w = [1/self.tree_size for i in range(0,len(pathes))]
                p = 0
                rw = 1
            weights.append(w)
            priors.append(p)
            rewards.append(rw)
            policy_dicts.append(pathes)
        return weights,priors,rewards,policy_dicts

    def plot_fits_data(self):
        pd = dict()
        pd['xValues'] = range(0, self.parameters.values.get('generation_size'))
        pd['yValues'] = self.fits
        # pd['ykeys'] = self.fits.keys()

        yValues0 = self.fits.get('max')
        yValues1 = self.fits.get('mean')
        yValues2 = self.fits.get('min')
        if np.min(np.array(yValues2)) > 0:
            minvalue = np.min(np.array(yValues2)) * 0.8
        if np.min(np.array(yValues2)) < 0:
            minvalue = np.min(np.array(yValues2)) * 1.2
        if np.min(np.array(yValues2)) == 0:
            minvalue = -1
        if np.max(np.array(yValues0)) > 0:
            maxvalue = np.max(np.array(yValues0)) * 1.2
        if np.max(np.array(yValues0)) < 0:
            maxvalue = np.max(np.array(yValues0)) * 0.8
        if np.max(np.array(yValues0)) == 0:
            maxvalue = 1
        # print( minvalue,maxvalue)
        pd['ylim'] = (minvalue, maxvalue)
        pd['xlim'] = (0, self.parameters.values.get('generation_size'))
        if self.parameters.values.get('group_criterion_method') is None:
            pd['ylabel'] = self.parameters.values.get('fitness_method')
        else:
            pd['ylabel'] = self.parameters.values.get('group_criterion_method') + '-' + self.parameters.values.get(
                'fitness_method')
        pd['xlabel'] = 'generation'
        pd['legend'] = self.fits.keys()
        pd['title'] = 'The Fitness Converge Line of ' + self.type
        if self.parameters.values.get('group_criterion_method') is None:
            pd['filename'] = self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
                self.step) + '-The FCL of ' + self.type + ' for ' + self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer') + '.pdf'
        else:
            pd['filename'] = self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
                self.step) + '-The FCL of ' + self.type + ' for ' + self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer') + '-' + self.parameters.values.get("emigrate_method") + '.pdf'
        self.result['Plot'][self.solver.get('type') + ' @ ' + self.solver.get(
            'pointer')]['fit'] = pd
    def plot_fits(self):
        fig = plt.figure()
        axis = fig.gca()
        xValues = range(0, self.parameters.values.get('generation_size'))
        yValues0 = self.fits.get('max')
        yValues1 = self.fits.get('mean')
        yValues2 = self.fits.get('min')
        if np.min(np.array(yValues2)) > 0:
            minvalue = np.min(np.array(yValues2))*0.8
        if np.min(np.array(yValues2)) < 0:
            minvalue = np.min(np.array(yValues2)) * 1.2
        if np.min(np.array(yValues2)) == 0:
            minvalue =-1
        if np.max(np.array(yValues0)) > 0:
            maxvalue = np.max(np.array(yValues0)) * 1.2
        if np.max(np.array(yValues0)) < 0:
            maxvalue = np.max(np.array(yValues0)) * 0.8
        if np.max(np.array(yValues0)) == 0:
            maxvalue = 1
        # print( minvalue,maxvalue)
        axis.set_ylim( minvalue,maxvalue)  # lower limit (0)
        axis.set_xlim(0, self.parameters.values.get('generation_size'))  # use same limits for x
        t0, = axis.plot(xValues, yValues0)
        t1, = axis.plot(xValues, yValues1)
        t2, = axis.plot(xValues, yValues2)
        if self.parameters.values.get('group_criterion_method') is None:
            axis.set_ylabel(self.parameters.values.get('fitness_method'))
        else:
            axis.set_ylabel(self.parameters.values.get('group_criterion_method') + '-' +self.parameters.values.get('fitness_method'))
        axis.set_xlabel('generation')
        axis.grid()
        fig.legend((t0, t1, t2), ('max', 'mean', 'min'), loc='center', fontsize=5)
        plt.title('The Fitness Converge Line of '+  self.type)
        # plt.show()
        if self.parameters.values.get('group_criterion_method') is None:
            fig.savefig( self.pnames.Save_filepath+self.domain+'-'+self.alg_name+ '-'+str(self.step)+ '-The FCL of '+  self.type + ' for '+ self.solver.get('type') + ' @ '+ self.solver.get('pointer')+ '.pdf')
        else:
            fig.savefig(self.pnames.Save_filepath+self.domain+'-'+self.alg_name+ '-'+str(self.step)+ '-The FCL of '+  self.type + ' for '+ self.solver.get('type') + ' @ '+ self.solver.get('pointer')+ '-' +self.parameters.values.get("emigrate_method")+ '.pdf')
    def display_pop(self,pop):
        for ind in pop:
            print("-----------------")
            for key in ind.keys():
                print('>key: '+ key)
                print('>value: '  )
                print( ind.get(key))
            print("-----------------")
class MPGA(object):
    def __init__(self, solver,  filepath, result,dstr,num_mod):
        self.domain = dstr
        self.solver = solver
        self.type = 'MPGA'
        self.num_mod = num_mod
        self.parameters = solver.get('parameters')
        self.dbn = DBN(filepath,solver.get('type'),solver.get('prestep'))
        self.dbn.expansion(solver.get('step'),expansion_flag=True)  # just set the node and arc, the sates and cpt is set in evaluation
        self.result = result
        self.dbn.result['prior_belief'] = dict()
        for key in result['prior_belief'].keys():
            self.dbn.result['prior_belief'][key] = result['prior_belief'].get(key)
        self.pnames = Name()
        self.policy_dict_list =  list()
        self.policy_path_weight_list = list()
        self.prior_belief_mat_list =  list()
        self.group_fits_list = list()
        self.fits = list()
        self.ga_alg = []
        self.filepath = filepath
    # @ overide
    def ga_main(self):
        for g in range(0, self.parameters.values.get('group_size')):
            print("-- Group %i --" % g)
            # TF_set = self.gen_tf(g)
            self.ga_alg = MGA(self.solver,self.filepath,self.result,self.domain,self.num_mod,self.type)
            self.ga_alg.ga_main()
            self.policy_dict_list.append(self.ga_alg.policy_dict)
            self.policy_path_weight_list.append(self.ga_alg.policy_path_weight)
            self.prior_belief_mat_list.append(self.ga_alg.prior_belief_mat)
            self.group_fits_list.append(self.ga_alg.fits)
        rerults = list()
        for r in self.group_fits_list:
            rerult = r.get('mean')
            rerults.append(rerult[len(rerults)-1])
        index = np.argmax(np.array(rerults))
        self.result['policy_dict'] = self.policy_dict_list[index]
        self.result['policy_path_weight'] = self.policy_path_weight_list[index]
        self.result['policy_belief'] = self.prior_belief_mat_list[index]
        self.fits = self.group_fits_list[index]
class MGGA(MGA):
    def __init__(self,solver, filepath, result,dstr,num_mod):
        self.domain = dstr
        self.type = 'MGGA'
        self.num_mod = num_mod
        self.parameters = solver.get('parameters')
        self.dbn = DBN(filepath, solver.get('type'),solver.get('prestep'))
        self.dbn.expansion(solver.get('step'),expansion_flag=True)  # just set the node and arc, the sates and cpt is set in evaluation
        self.result = result
        self.dbn.result['prior_belief'] = dict()
        for key in result['prior_belief'].keys():
            self.dbn.result['prior_belief'][key] = result['prior_belief'].get(key)
        self.pnames = Name()
        super(MGGA, self).__init__(solver, filepath, result,self.domain,num_mod,self.type)
    # @ overide
    def gen_pop(self):
        pop = []
        for gi in range(0, self.parameters.values.get('group_size') ):
            for key in self.genomes.keys():
                geneinfo = self.genomes.get(key)
                fits = self.result['reward'].get(key)
                ind = self.gen_ind(Gene(data=geneinfo))
                ind['id'] = gi
                pop.append(ind)  # store the chromosome and its fitness
        self.geneinfo_dim = len(geneinfo)
        self.pop = [ind for ind in pop]
        self.evaluate()
        self.evaluate_group_criterion()
        self.group_data4plot()
    def select_nextpop(self,size):
        s_inds = sorted(self.pop, key=itemgetter("fitness"), reverse=True)
        nextpop = []
        for gp in range(0, self.parameters.values.get('group_size') ):
            count = 0
            for ind in s_inds:
                if ind['id']==gp:
                    nextpop.append(ind)  # store the chromosome and its fitness
                    count = count + 1
                if count == size:
                    break
        self.pop = nextpop
    def ga_main(self):
        print("Starting evolution")
        # Begin the evolution
        if self.progressbar:
            bar = tqdm(total=int(self.parameters.values.get('generation_size')))
        for g in range(self.parameters.values.get('generation_size')):
            if not self.progressbar:
                print("-- Generation %i --" % g)
            # Apply selection based on their converted fitness
            next_pops = []
            for gp in range(0, self.parameters.values.get('group_size')):
                # print("-- Group %i --" % gp)
                selectpop = self.selection(self.pop, self.parameters.values.get('pop_size'), gp)
                nextoff = []
                # self.display_pop(selectpop)
                while len(nextoff) != self.parameters.values.get('pop_size'):
                    # Apply crossover and mutation on the offspring
                    # Select two individuals
                    offspring = [random.choice(selectpop) for i in range(0, 2)]
                    if random.random() < self.parameters.values.get('crossover_rate'):  # cross two individuals with probability CXPB
                        crossoff = self.crossoperate(offspring)
                        if random.random() < self.parameters.values.get('mutation_rate'):  # mutate an individual with probability MUTPB
                            muteoff = self.mutation(crossoff)
                            if self.check_identity_flag:
                                if not self.check_identity(nextoff, muteoff) and not self.check_identity(self.pop,
                                                                                                         muteoff):
                                    ind = self.gen_id(nextoff, self.gen_ind(muteoff), gp)
                                    nextoff.append(ind)  # initialize
                            else:
                                ind = self.gen_id(nextoff, self.gen_ind(muteoff), gp)
                                nextoff.append(ind)  # initialize
                [next_pops.append(ind) for ind in nextoff]
            # The population is entirely replaced by the offspring
            # self.next_pop(next_pops)
            if g ==  self.parameters.values.get('generation_size')-1:
                self.next_pop(next_pops, final_pop=True)
            else:
                self.next_pop(next_pops)
            # Gather all the fitnesses in one list and print the stats
            fits = [ind['fitness'] for ind in self.pop]
            length = len(self.pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5
            best_ind = self.selectBest(self.pop)
            if best_ind['fitness'] > self.bestindividual['fitness']:
                self.bestindividual = best_ind
            if not self.progressbar:
                print(
                    "Best individual found is %s, %s" % (
                    self.bestindividual['Gene'].data, self.bestindividual['fitness']))
                print("  Min fitness of current pop: %s" % min(fits))
                print("  Max fitness of current pop: %s" % max(fits))
                print("  Avg fitness of current pop: %s" % mean)
                print("  Std of currrent pop: %s" % std)
            else:
                bar.update(1)
            self.fits['max'].append(max(fits))
            self.fits['min'].append(min(fits))
            self.fits['mean'].append(mean)
            self.fits['std'].append(std)
        if self.progressbar:
            bar.close()
        print("-- End of (successful) evolution --")
        # self.next_pop(next_pops, final_pop=True)
        # self.next_pop(nextoff, final_pop=True)
        self.choose_group()
        gens = [ind['Gene'].data for ind in self.pop]
        for gen in gens:
            print(gen)
        self.gen_other()
        if self.plot_flag:
            self.plot_fits()
            self.plot_groupdatas(self.group_fits, 'group_fits')
            self.plot_groupdatas(self.group_criterions, 'group_criterions')
        else:
            self.result['Plot'][self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer')] = dict()
            self.plot_fits_data()
            self.plot_groupdatas_data(self.group_fits, 'group_fits')
            self.plot_groupdatas_data(self.group_criterions, 'group_criterions')
    def plot_groupdatas_data(self, data, ylabel):
        pd = dict()
        xlen = len(self.group_fits[0])
        pd['xValues'] = range(0, xlen)
        minvalue = np.min([np.min(data.get(key)) for key in data.keys()])
        maxvalue = np.max([np.max(data.get(key)) for key in data.keys()])
        pd['ylim'] = (minvalue, maxvalue)
        pd['xlim'] = (0, self.parameters.values.get('generation_size'))
        pd['yValues'] = data
        pd['ykeys'] = data.keys()
        t_legend = list()
        for key in data.keys():
            t_legend.append('Group @' + str(key))
        pd['legend'] = t_legend
        pd['ylabel'] = ylabel
        pd['xlabel'] = 'generation'
        pd['title'] = 'The Fitness Converge Line of ' + self.type + '@ ' + ylabel
        pd['filename'] = self.pnames.Save_filepath + self.domain + '-' + self.alg_name + '-' + str(
            self.step) + '-The FCL of ' + self.type + '@ ' + ylabel + '-' + str(
            self.parameters.values.get("emigrate_method")) + ' for ' + self.solver.get(
            'type') + ' @ ' + self.solver.get('pointer') + '.pdf'
        self.result['Plot'][self.solver.get('type') + ' @ ' + self.solver.get(
                'pointer')][ylabel] = pd
    def evaluate_diversity(self):
        # evaluate the diversity of pop
        gpop = dict()
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            id = ind['id']
            if gpop.__contains__(id):
                ids = gpop.get(id)
                ids.append(gi)
                gpop[id] = ids
            else:
                gpop[id] = [gi]
        self.group_criterion = list()
        for key in gpop.keys():
            popindex = gpop.get(key)
            diversity_pop = self.cal_diversity(popindex)
            self.group_criterion.append(diversity_pop)
            for gi in popindex:
                ind = self.pop[gi]
                diversity_pop_gi = self.cal_diversity(popindex, gi)
                fits = diversity_pop_gi / diversity_pop  # divide
                ind['fitness'] = fits
                self.pop[gi] = ind
    def evaluate_distance(self):
        # evaluate the distance of pop
        # print('evaluate the distance of pop')
        genomes = list()
        for gp in range(0,self.parameters.values.get('group_size')):
            genome = self.genome_template
            genome = [0 for i in range(0, len(genome))]
            count = 0
            for ind in self.pop:
                if ind['id'] == gp:
                    count =  count +1
                    gen = ind['Gene'].data
                    genome = [genome[i] + gen[i] for i in range(0, len(gen))]
            genome = [g / count for g in genome]
            genomes.append(genome)
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            gen = ind['Gene'].data
            genome = genomes[ind['id']]
            fits = np.sqrt(np.sum([np.power(genome[j] - gen[j], 2) for j in range(0, len(gen))]))  # np.array
            ind['fitness'] = fits
            # print(fits)
            self.pop[gi] = ind
    # pass implement
    def gen_id(self, nextoff, ind, gp):
        ind['id'] = gp
        return ind
    # def gen_id(self, nextoff, ind, gp):
    #     if random.random() >= self.parameters.values.get('emigrate_rate'):
    #         ind['id'] = gp
    #         return ind
    #     id_set = list(range(0, self.parameters.values.get('group_size')))
    #     id_set.pop(id_set.index(gp))
    #     ind['id'] = id_set[random.randrange(0, len(id_set))]
    #     return ind
    def gen_meanfits_deltas(self,nextoff):
        self.mean_fits = list()
        self.deltas = list()
        if nextoff is None:
            pop = self.pop
        else:
            pop = nextoff
        rate = 1 - self.parameters.values.get('emigrate_rate')
        if rate == 0:
            rate = 1 - self.parameters.values.get('emigrate_rate') + self.pnames.Elimit
        for gp in range(0, self.parameters.values.get('group_size')):
            fits = []
            for ind in pop:
                id = ind['id']
                if id == gp:
                    fits.append(ind['fitness'])
            fits = sorted(fits)
            midind = int(np.round(len(fits)/2))
            self.mean_fits.append(fits[midind])
            # self.mean_fits.append(np.mean(fits))
            # a = np.abs(np.max(fits) - np.min(fits))
            # a =1
            # b = np.e
            # b = math.exp(1/rate)
            # delta = a*b/(np.abs(np.max(fits) - np.mean(fits))*rate)
            if self.parameters.values.get("emigrate_method") == self.pnames.Emigrate_method.get(3):
                delta = np.log((1 - rate) / (rate))/np.abs(np.max(fits) - np.mean(fits))
            if  self.parameters.values.get("emigrate_method") == self.pnames.Emigrate_method.get(4):
                delta = np.log((1 - 0.95*rate) / (0.95*rate))/np.abs(np.max(fits) - np.mean(fits))
            # delta = a * b / (np.abs(np.max(fits) - np.mean(fits)) )
            self.deltas.append(delta)
    def group_data4plot(self):
        for gp in range(0, self.parameters.values.get('group_size')):
            fits = []
            for ind in self.pop:
                id = ind['id']
                if id == gp:
                    fits.append(ind['fitness'])
            self.group_fits[gp].append(np.mean(fits))
            self.group_criterions[gp].append(self.group_criterion[gp])
    def plot_groupdatas(self, data, ylabel):
        import matplotlib.colors as mcolors

        fig = plt.figure()
        axis = fig.gca()
        xlen = len(self.group_fits[0])
        # xValues = range(0, self.parameters.values.get('generation_size'))
        xValues = range(0, xlen)
        minvalue = np.min([np.min(data.get(key)) for key in data.keys()])
        maxvalue = np.max([np.max(data.get(key)) for key in data.keys()])
        # print( minvalue,maxvalue)
        axis.set_ylim(minvalue, maxvalue)  # lower limit (0)
        axis.set_xlim(0, self.parameters.values.get('generation_size'))  # use same limits for x
        t = list()
        t_legend = list()
        count = 0
        color_set = mcolors.CSS4_COLORS
        ckeys = list(color_set.keys())
        for key in data.keys():
            t_i, = plt.plot(xValues, data.get(key), color=color_set[ckeys[0 + 3 * count + 10]])
            t.append(t_i)
            t_legend.append('Group @' + str(key))
            count = count + 1
        axis.set_ylabel(ylabel)
        axis.set_xlabel('generation')
        axis.grid()
        plt.legend(t, t_legend, loc='best')
        plt.title('The Fitness Converge Line of ' + self.type + '@ ' + ylabel)
        # plt.show()
        fig.savefig(self.pnames.Save_filepath + self.domain +'-'+self.alg_name+ '-'+str(self.step)+  '-The FCL of ' + self.type + '@ ' + ylabel + '-' + str(
            self.parameters.values.get("emigrate_method")) + ' for ' + self.solver.get(
            'type') + ' @ ' + self.solver.get('pointer') + '.pdf')
    def next_pop(self, nextoff,final_pop= None):
        # print('next pop')
        # print(len(self.pop),len( nextoff))
        # self.emigration(self.pop)
        pop_temp = [ind for ind in self.pop]
        self.pop = [ind for ind in nextoff]
        self.evaluate()
        self.evaluate_group_criterion()
        nextoff = self.emigration(self.pop)
        self.pop = [ind for ind in pop_temp]
        if self.parameters.values.get('pelite_mode'):
            pop_temp = [ind for ind in self.pop]
            self.pop = [ind for ind in nextoff]
            [self.pop.append(ind) for ind in pop_temp]
        else:
            self.pop = [ind for ind in nextoff]
        self.select_nextpop(self.parameters.values.get('pop_size'))
        self.evaluate_group_criterion()
        self.group_data4plot()
        if self.solver.get('pointer') == 'solving network':
            if final_pop is None:
                if self.parameters.values.get('elite_mode'):
                    if self.parameters.values.get('tournament_size') >= self.num_mod:
                        self.select_nextpop(self.parameters.values.get('tournament_size'))
                    else:
                        self.select_nextpop(self.num_mod)
            else:
                self.select_nextpop(self.num_mod)
        else:
            if self.parameters.values.get('elite_mode'):
                self.select_nextpop(self.parameters.values.get('tournament_size'))
    def emigration(self, nextoff):
        md = 1
        # random
        if self.parameters.values.get("emigrate_method") == self.pnames.Emigrate_method.get(md):
            # newoffs = list()
            inds = 0
            for ind in nextoff:
                if random.random() <= self.parameters.values.get('emigrate_rate'):
                    # newoff = self.gen_ind(ind['Gene'])
                    gpn = self.russian_roulette(self.group_criterion)
                    if gpn != ind['id']:
                        ind['id'] = gpn
                        inds = inds + 1
                    # newoff['id'] = gpn
                    # newoff['fitness'] = ind['fitness']
                    # newoffs.append(newoff)

            # [nextoff.append(ind) for ind in newoffs]
            print(inds)
            return nextoff
        md = md+1
        # random-bound
        if self.parameters.values.get("emigrate_method") == self.pnames.Emigrate_method.get(md):
            inds = 0
            gprobs = dict()
            for ind in nextoff:
                fit = ind['fitness']
                gp = ind['id']
                prob = fit
                if gprobs.__contains__(gp):
                    probs = gprobs.get(gp)
                    probs[0].append(ind)
                    probs[1].append(prob)
                    gprobs[gp] = probs
                else:
                    gprobs[gp] = [[ind], [prob]]
            for g in gprobs.keys():
                number_eg = np.ceil(
                    self.parameters.values.get('emigrate_rate') * self.parameters.values.get('pop_size'))
                ind_list = gprobs.get(gp)[0]
                prob_list = gprobs.get(gp)[1]
                while number_eg>0:
                    pos = self.russian_roulette(prob_list)
                    # print(pos, prob_list[pos])
                    ind = ind_list[pos]
                    gpn = self.russian_roulette(self.group_criterion)
                    if gpn != ind['id']:
                        ind['id'] = gpn
                        inds = inds+1
                    number_eg = number_eg-1
            print(inds)
            return nextoff
        md = md+1
        # sigmoid-bound
        if  self.parameters.values.get("emigrate_method") == self.pnames.Emigrate_method.get(md):
            self.gen_meanfits_deltas(nextoff)
            newoffs = list()
            inds = 0
            gprobs = dict()
            for ind in nextoff:
                fit = ind['fitness']
                gp = ind['id']
                prob = self.stable_sigmoid((fit - self.mean_fits[gp]) * self.deltas[gp])
                if gprobs.__contains__(gp):
                    probs = gprobs.get(gp)
                    probs[0].append(ind)
                    probs[1].append(prob)
                    gprobs[gp] = probs
                else:
                    gprobs[gp] = [[ind],[prob]]
            for g in gprobs.keys():
                number_eg = np.ceil(
                    self.parameters.values.get('emigrate_rate') * self.parameters.values.get('pop_size'))
                ind_list = gprobs.get(gp)[0]
                prob_list = gprobs.get(gp)[1]
                while number_eg>0:
                    pos = self.russian_roulette(prob_list)
                    # print(pos, prob_list[pos])
                    ind = ind_list[pos]
                    gpn = self.russian_roulette(self.group_criterion)
                    if gpn != ind['id']:
                        ind['id'] = gpn
                        inds = inds+1
                    number_eg = number_eg-1
            print(inds)
            if len(newoffs)>0:
                print(len(newoffs))
                [nextoff.append(ind) for ind in newoffs]
            return nextoff
        md = md+1
        # sigmoid- cutoff
        if  self.parameters.values.get("emigrate_method") == self.pnames.Emigrate_method.get(md):
            self.gen_meanfits_deltas(nextoff)
            # pop_temp = [ind for ind in self.pop]
            # self.pop = [ind for ind in nextoff]
            # self.evaluate()
            # self.evaluate_group_criterion()
            # self.pop = [ind for ind in  pop_temp]
            newoffs = list()
            # counts = dict()
            for ind in nextoff:
                gp = ind['id']
            #     if counts.__contains__(gp):
            #         counts[gp] = counts.get(gp)+1
            #     else:
            #         counts[gp] = 1
            # if self.parameters.values.get('pelite_mode'):
            #     limit = self.parameters.values.get('pop_size')
            #     if self.parameters.values.get('elite_mode'):
            #         limit = self.parameters.values.get('tournament_size')
            # else:
            #     limit = int(np.round(self.parameters.values.get('pop_size'))*0.618)
            inds = 0
            for ind in nextoff:
                fit = ind['fitness']
                gp = ind['id']
                # if counts.get(gp) <= limit:
                #     continue
                prob = self.stable_sigmoid((fit - self.mean_fits[gp]) * self.deltas[gp])
                if prob >= 1 - self.parameters.values.get('emigrate_rate'):
                    gpn = self.russian_roulette(self.group_criterion)
                    if gpn != gp:
                        # newoff = self.gen_ind(ind['Gene'])
                        # newoff['id'] = gpn
                        # newoff['fitness'] = ind['fitness']
                        # newoffs.append(newoff)
                        ind['id'] = gpn
                        inds = inds+1
                        # counts[gp] = counts.get(gp) -1
                        # counts[gpn] = counts.get(gpn) + 1
                        # print('>>>emigrate from: ' + str(gp) + '  to: ' + str(gpn))
            print(inds)
            if len(newoffs)>0:
                print(len(newoffs))
                [nextoff.append(ind) for ind in newoffs]
            return nextoff
        md =md+1
        # sigmoid- group-cutoff
        if  self.parameters.values.get("emigrate_method") == self.pnames.Emigrate_method.get(md):
            fits = [ind['fitness'] for ind in nextoff]
            mean_fits = np.mean(fits)
            # midind = int(np.round(len(fits) / 2))
            # mean_fits =  fits[midind]
            rate = 1 - self.parameters.values.get('emigrate_rate')
            if rate == 0:
                rate = 1 - self.parameters.values.get('emigrate_rate') + self.pnames.Elimit
            # delta = np.abs(np.max(fits) -np.min(fits)) *np.e / (np.abs(np.max(fits) - mean_fits) * rate)
            delta =np.log((1 - rate) / (rate)) / np.abs(np.max(fits) - np.mean(fits))
            newoffs = list()
            inds = 0
            for ind in nextoff:
                fit = ind['fitness']
                gp = ind['id']
                prob = self.stable_sigmoid((fit - mean_fits) * delta)
                # prob = self.stable_sigmoid((fit - self.mean_fits[gp]) * self.deltas[gp])
                if prob >= 1 - self.parameters.values.get('emigrate_rate'):
                    gpn = self.russian_roulette(self.group_criterion)
                    if gpn != gp:
                        # newoff = self.gen_ind(ind['Gene'])
                        # newoff['id'] = gpn
                        # newoff['fitness'] = ind['fitness']
                        # newoffs.append(newoff)
                        ind['id'] = gpn
                        inds = inds + 1
                        # print('>>>emigrate from: ' + str(gp) + '  to: ' + str(gpn))
            print(inds)
            if len(newoffs)>0:
                print(len(newoffs))
                [nextoff.append(ind) for ind in newoffs]
            return nextoff
    def russian_roulette(self, gc):
        import random
        min_gc = np.min(gc)
        gctt = [e - min_gc+self.pnames.Elimit for e in gc]
        sum_w = np.sum(gctt)
        sum_ = 0
        u = random.random() * sum_w
        for pos in range(0, len(gctt)):
            sum_ += gctt[pos]
            if sum_ > u:
                break
        return pos
    #Sigmoid Function
    def stable_sigmoid(self,x):
        if x >= 0:
            z = math.exp(-x)
            sig = 1 / (1 + z)
            return sig
        else:
            z = math.exp(x)
            sig = z / (1 + z)
            return sig
    def choose_group(self):
        self.next_pop(self.pop)
        self.evaluate_group_criterion()
        g_choosen = np.argmax(np.array(self.group_criterion)) + 1
        self.pop_save = [ind for ind in self.pop]
        pop = [ind for ind in self.pop if g_choosen == ind['id']]
        self.pop = pop
    def selection_group(self, individuals, gp):
        individuals = [ind for ind in individuals if ind['id'] == gp]
        return individuals

    def initialize_other(self):
        self.pop_save = []
        self.group_criterion = list()
        self.group_fits = dict()
        self.group_criterions = dict()
        for gp in range(0, self.parameters.values.get('group_size')):
            self.group_fits[gp] = list()
            self.group_criterions[gp] = list()
    # new
    def evaluate_group_criterion(self):
        if self.parameters.values.get('group_criterion_method') == self.pnames.Group_criterion_method.get(1):
            self.evaluate_gc_distance()
        if self.parameters.values.get('group_criterion_method') == self.pnames.Group_criterion_method.get(2):
            self.evaluate_gc_diversity()
        if self.parameters.values.get('group_criterion_method') == self.pnames.Group_criterion_method.get(3):
            self.evaluate_gc_reward()
    def evaluate_gc_distance(self):
        genomes = list()
        for gp in range(0, self.parameters.values.get('group_size')):
            genome = self.genome_template
            genome = [0 for i in range(0, len(genome))]
            count = 0
            for ind in self.pop:
                if ind['id'] == gp:
                    count = count + 1
                    gen = ind['Gene'].data
                    genome = [genome[i] + gen[i] for i in range(0, len(gen))]
            genome = [g / count for g in genome]
            genomes.append(genome)
        distances = list()
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            gen = ind['Gene'].data
            genome = genomes[ind['id']]
            fits = np.sqrt(np.sum([np.power(genome[j] - gen[j], 2) for j in range(0, len(gen))]))  # np.array
            distances.append(fits)
        self.group_criterion = list()
        for gp in range(0, self.parameters.values.get('group_size')):
            reward_pop = []
            for gi in range(0, len(self.pop)):
                ind = self.pop[gi]
                id = ind['id']
                if id == gp:
                    reward_pop.append(distances[gi])
            self.group_criterion.append(np.mean(np.array(reward_pop)))
    def evaluate_gc_reward(self ):
        reward = list()
        genome = self.genome_template
        genome = [0 for i in range(0, len(genome))]
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            genome = ind['Gene'].data
            pathes = self.genome_to_pathes(genome)
            policy_dict = dict()
            policy_dict[1] = pathes
            policy_tree = PolicyTree('The  Policy Tree of ' + self.type,self.dbn.action_list, self.dbn.observation_list)
            policy_tree.set_policy_dict(policy_dict)
            policy_tree.gen_policy_trees_memorysaved()
            # policy_tree.save_policytree(self.pnames.Save_filepath)
            self.dbn.expa_policy_tree = policy_tree
            self.dbn.expansion(self.step,expansion_flag=False)
            # r, w, p = self.dbn.get_reward(weight_off=True)
            if self.parameters.values.get('cover_mode'):
                r, w, p = self.dbn.get_reward(weight_off=True, modi=gi)
            else:
                r, w, p = self.dbn.get_reward(weight_off=True)
            reward.append(r)
        self.group_criterion = list()
        for gp in range(0, self.parameters.values.get('group_size')):
            reward_pop = []
            for gi in range(0, len(self.pop)):
                ind = self.pop[gi]
                id = ind['id']
                if id == gp:
                    reward_pop.append(reward[gi])
            self.group_criterion.append(np.mean(np.array(reward_pop)))
    def evaluate_gc_diversity(self):
        # evaluate the distance or diversity of pop
        gpop = dict()
        for gi in range(0, len(self.pop)):
            ind = self.pop[gi]
            id = ind['id']
            if gpop.__contains__(id):
                ids = gpop.get(id)
                ids.append(gi)
                gpop[id] = ids
            else:
                gpop[id] = [gi]
        self.group_criterion = list()
        popindex = range(0, len(self.pop))
        # diversity_global = self.cal_diversity(popindex)
        for key in gpop.keys():
            popindex = gpop.get(key)
            diversity_pop = self.cal_diversity(popindex)
            # diversity_pop = diversity_pop/diversity_global
            self.group_criterion.append(diversity_pop)


