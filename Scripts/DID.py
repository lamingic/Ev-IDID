# Author: Biyang Ma <biyangma@stu.xmu.edu.cn> <mabiyang001@hotmail.com>
from Model import Model
###########################################################
class DID(Model):
    def __init__(self, DomainParameters, ModelParameters,scr_message = None):
        self.type = 'DID'
        super(DID, self).__init__(DomainParameters, ModelParameters,self.type,scr_message)