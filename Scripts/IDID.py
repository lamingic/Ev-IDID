# Copyright (C) 2001-2020
# Author: Biyang Ma <biyangma@stu.xmu.edu.cn> <mabiyang001@hotmail.com>
from Model import Model
from DID import DID
class IDID(Model):
    def __init__(self, DomainParameters, ModelParameters,scr_message = None):
        # attributes
        self.type = 'IDID'
        self.did = DID(DomainParameters, ModelParameters,scr_message)
        super(IDID, self).__init__(DomainParameters, ModelParameters,self.type,scr_message)
    # parameter
    def gen_pathes_online(self):
        self.print('\n')
        self.print('Online Building Model:  '+self.ModelParameters.Name+ '>>>>>@ DID/IDID type: ' + self.type)
        self.next_step(start=True)
        self.dbn.extend(self.DomainParameters.values['horizon_size'],self.step)

        self.next_step(start=True,end=True)
        self.dbn.generate_evidence(self.step)

        self.next_step(start=True, end=True)
        self.did.gen_pathes()

        self.next_step(start=True, end=True)
        self.dbn.expa_policy_tree = self.did.dbn.result.get('policy_tree')
        self.dbn.expansion(self.step)
        self.next_step(start=True, end=True)
        self.next_step(start=False, end=True)
        self.expansion(expansion_flag=True)
        self.next_step(start=True, end=False)
    def gen_pathes(self):
        self.print('\n')
        self.print('Building Model:  '+self.ModelParameters.Name+ '>>>>>@ DID/IDID type: ' + self.type)
        self.next_step(start=True)
        self.dbn.extend(self.DomainParameters.values['horizon_size'],self.step)

        self.next_step(start=True,end=True)
        self.dbn.generate_evidence(self.step)

        self.next_step(start=True, end=True)
        self.did.gen_pathes()

        self.next_step(start=True, end=True)
        self.dbn.expa_policy_tree = self.did.dbn.result.get('policy_tree')
        self.dbn.expansion(self.step)

        self.next_step(start=True,end=True)
        self.solve_mod()

        self.next_step(start=False, end=True)
        self.expansion(expansion_flag=True)

        self.next_step(start=True, end=False)
        self.extend()

        self.next_step(start=True,end=True)
        self.dbn.result.get('policy_tree').gen_policy_trees_memorysaved()
        self.dbn.result.get('policy_tree').set_name(self.DomainParameters.Tostring()+ '-'+self.ModelParameters.Name +'-'+str(self.step)+
            '-The Merged Policy Tree of ' + self.type )
        # self.dbn.result['policy_tree'].save_policytree(self.pnames.Save_filepath)
        self.next_step(end=True)

        self.expansion(expansion_flag=False,policy_tree= self.dbn.result.get('policy_tree'))
        self.print('Finish Model:  ' + self.ModelParameters.Name + '>>>>>@ DID/IDID type: ' + self.type)
        self.print('\n')