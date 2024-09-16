# -*- coding: utf-8 -*-

#
# Copyright (C) 2001-2020
# Author: Biyang Ma <biyangma@stu.xmu.edu.cn> <mabiyang001@hotmail.com>
# builtins

###########################################################
class Node(object):
    def __init__(self, id=None, label=None,level = None, parents=None,arcs_from_parents = None, children=None,arcs_to_children = None):
        self.id = id
        self.label = ' '
        self.children = list()#node memory add
        self.parents = list()#
        self.level = 0
        self.arcs_from_parents = dict()
        self.arcs_to_children = dict()

        self.set_label(label)
        self.set_level(level)
        self.set_parents(parents)
        self.set_arcs_from_parents(arcs_from_parents)
        self.set_children(children)#
        self.set_arcs_to_children(arcs_to_children)
    def display_dict(self,dictionary):
        print('--------Dict-----------')
        for key in dictionary.keys():
            print('key: ' + key)
            nodes = dictionary.get(key)
            if isinstance(nodes, list):
                ids = ','.join([ns.get_id() for ns in nodes])
            else:
                ids = nodes.get_id()
            print('ids: ' + ids)
        print('----------------------')
    def display(self):
        if self.parents != None:
            pa_ids =','.join( [pa.get_id() for pa in self.parents])
        else:
            pa_ids = 'Null'
        if self.children != None:
            ch_ids = ','.join([ch.get_id() for ch in self.children])
        else:
            ch_ids = 'Null'

        print('--------Node-----------')
        print('id      : '+ str(self.id))
        print('label   : ' + str(self.label))
        print('level   : ' + str(self.level))
        print('children: ' + ch_ids)
        self.display_dict(self.arcs_to_children)
        print('parents : ' + pa_ids)
        self.display_dict(self.arcs_from_parents)
        print('----------------------')
        print()
    # copy
    def copy(self):
        node = Node(self.id, self.label, self.level , self.parents, self.arcs_from_parents, self.children, self.arcs_to_children)
        return node
    # get
    def get_id(self):
        return self.id
    def get_label(self):
        return self.label
    def get_level(self):
        return self.level
    # set
    def set_id(self,id):
        if id is None:
            self.id = '-1'
        else:
            self.id = id
    def set_label(self, label):
        if label is None:
            self.label = '-1'
        else:
           self.label = label
    def set_level(self,level):
        if level is None:
            self.level = -1
        else:
         self.level =level
    # is
    def is_with_eq_id(self,node):
        if  self.get_id() == node.get_id():
            return True
        else:
            return False
    def is_leaf(self):
        return not (self.children)
    def is_root(self):
        return not (self.parents)
    def is_with_eq_label(self,node):
        if  self.get_label() == node.get_label():
            return True
        else:
            return False
    def is_goto_eq_children(self,node):
        flag = False
        if len(self.children)!= len(node.children):
            return flag
        for arc in self.arcs_to_children.keys():
            child1 = self.arcs_to_children.get(arc)
            if node.arcs_to_children_contains(arc):
                child2 = node.arcs_to_children.get(arc)
                if not child1.is_with_eq_id(child2):
                    return flag
            else:
                return flag
        flag = True
        return flag
    # get
    def get_parents(self):
        return self.parents
    def get_arcs_from_parents(self):
        return self.arcs_from_parents
    def get_children(self):#
        return self.children
    def get_arcs_to_children(self):
        return self.arcs_to_children
    # set
    def set_parents(self,parents):
        if parents  is None:
            self.parents  = list()
        else:
            self.parents = parents
    def set_arcs_from_parents(self,arcs_from_parents):
        if arcs_from_parents is None:
            self.arcs_from_parents = dict()
        else:
            self.arcs_from_parents = arcs_from_parents
    def set_arc_from_parent(self,parent,arc):
        if self.arcs_from_parents_contains(arc):
            list_arc = self.arcs_from_parents.get(arc)
            list_arc.append(parent)
            self.arcs_from_parents[arc] = list(set(list_arc))
        else:
            list_arc = list()
            list_arc.append(parent)
            self.arcs_from_parents[arc] = list_arc
    def arcs_from_parents_contains(self,arc):
        if self.arcs_from_parents is None:
            return False
        else:
            if self.arcs_from_parents.__contains__(arc):
                return True
            else:
                return False
    def remove_arc_from_parent(self,parent,arc):
        if self.arcs_from_parents_contains(arc):
            list_arc = self.arcs_from_parents.get(arc)
            list_arc.remove(parent)
            if len(list_arc)>0:
                self.arcs_from_parents[arc] = list_arc
            else:
                self.arcs_from_parents.pop(arc)
    def set_children(self, children):
        if children is None:
            self.children = list()
        else:
            self.children = children
    def set_arcs_to_children(self, arcs_to_children):
        if arcs_to_children is None:
            self.arcs_to_children = dict()
        else:
            self.arcs_to_children = arcs_to_children
    def set_arc_to_child(self, child, arc):
        self.arcs_to_children[arc] = child
    def arcs_to_children_contains(self,arc):
        if self.arcs_to_children is None:
            return False
        else:
            if self.arcs_to_children.__contains__(arc):
                return True
            else:
                return False
    def remove_arc_to_children(self,child,arc):
        if self.arcs_to_children_contains(arc):
            self.arcs_to_children.pop(arc)
    def contain_parent(self, parent):
        if self.parents is None:
            self.parents = list()
            return False
        else:
            if self.parents.__contains__(parent):
                return True
            else:
                return False
    def remove_parent(self, parent):
        if self.contain_parent(parent):
            self.parents.remove(parent)
    def insert_parent(self, parent):
        if not self.contain_parent(parent):
            self.parents.append(parent)
    def set_parents(self, parents):
        if self.parents is None:
            self.parents = parents
    def contain_child(self, child):
        if self.children is None:
            self.children  = list()
            return False
        else:
            if self.children.__contains__(child):
                return True
            else:
                return False
    def remove_child(self, child):
        if self.contain_child(child):
            self.children.remove(child)
    def insert_child(self, child):
        if not self.contain_child(child):
            self.children.append(child)
    def set_children(self, children):
        if self.children is None:
            self.children = children
