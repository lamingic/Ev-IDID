# -*- coding: utf-8 -*-
# Copyright (C) 2001-2020
# Author: Biyang Ma <biyangma@stu.xmu.edu.cn> <mabiyang001@hotmail.com>
import numpy as np
import tkinter as tk
def iter_except(function, exception):
    """Works like builtin 2-argument `iter()`, but stops on `exception`."""
    try:
        while True:
            yield function()
    except exception:
        return

class Game(tk.Frame):
    def __init__(self, master):
        super(Game, self).__init__(master)
        self.lives = 3
        self.width = 500
        self.height = 500
        self.canvas = tk.Canvas(self, bg='#aaaaff',width=self.width,height=self.height)
        self.canvas.pack()
        self.pack()
        self.game_objects = dict()
    def get_canvas(self):
        return self.canvas
    def get_game_objects(self):
        return self.game_objects
    def add_game_object(self,game_object):
        if not self.game_objects.__contains__(game_object.get_id()):
           #print('no exist')
            self.game_objects[game_object.get_id()] = (game_object)
        else:
            self.game_objects[game_object.get_id()] = (game_object)
    def delete_game_object(self, game_object):
        if self.game_objects.__contains__(game_object.get_id()):
            self.game_objects.pop(game_object.get_id())
            self.canvas.delete(game_object.get_item())
class GameObject(object):
    COLORS = {1: 'white', 2: 'red'}
    STATUS = {1:'inactived',2:'actived'}
    def __init__(self, canvas, item,label_str=None, id=None):
        self.canvas = canvas
        self.item = item
        self.id = id
        self.label_str = label_str
        self.active_status = GameObject.STATUS[1]
        self.show_label()
        self.angle = 0
    def set_label_angle(self,angle):
        self.angle = angle
    def get_item(self):
        return self.item

    def set_label_str(self, label_str):
        self.label_str = label_str

    def get_label_str(self):
        return self.label_str
    def set_id(self,id):
        self.id = id
    def get_id(self):
        return self.id
    def active(self):
        self.canvas.itemconfig(self.item, fill= GameObject.COLORS[2])
        self.active_status =  GameObject.STATUS[2]
    def disactive(self):
        self.canvas.itemconfig(self.item, fill= GameObject.COLORS[1])
        self.active_status =  GameObject.STATUS[1]
    def get_position(self):
        return self.canvas.coords(self.item)
    def move(self, x, y):
        self.canvas.move(self.item, x, y)
        if not self.label_str == None:
            self.canvas.move(self.label, x, y)
    def delete(self):
        self.canvas.delete(self.item)
        if not self.label_str == None:
            self.canvas.delete( self.label)
    def show_label(self):
        if not self.label_str ==None:
            x = (self.get_position()[0] + self.get_position()[2]) / 2
            y = (self.get_position()[1] + self.get_position()[3]) / 2
            self.label = self.canvas.create_text((x, y), text=self.label_str,angle = self.angle)
class Door(GameObject):
    COLORS = {1: 'yellow', 2: 'green'}
    def __init__(self, canvas, x, y,label_str,id=None):
        self.width = 100
        self.height = 20
        self.x = x
        self.y = y
        self.angle = 0
        self.label_str = label_str
        self.id = id
        self.status = 'close'
        self.canvas = canvas
        color = Door.COLORS[1]
        self.item = self.canvas.create_rectangle(x, y,x + self.width, y + self.height,fill=color, tags=id)
        super(Door, self).__init__(self.canvas, self.item,self.label_str, self.id)
    def open(self):
        color = Door.COLORS[2]
        position = self.get_position()
        #print(position)
        x = position[0]
        y = position[1]
        xw = position[0] + (position[3] - position[1])
        yh = position[1]+ (position[2] - position[0])
        super(Door, self).delete()
        self.item = self.canvas.create_rectangle(x, y, xw,yh, fill=color, tags=self.id)
        super(Door, self).set_label_angle(270)
        super(Door, self).__init__(self.canvas, self.item, self.label_str,self.id)
        self.status = 'open'

    def close(self):
        super(Door, self).delete()
        color = Door.COLORS[1]
        self.item = self.canvas.create_rectangle(self.x, self.y, self.x + self.width, self.y + self.height, fill=color, tags=self.id)
        super(Door, self).set_label_angle(00)
        super(Door, self).__init__(self.canvas, self.item, self.label_str,self.id)
        self.status = 'close'
class Brick(GameObject):
    COLORS = {1: 'black', 2: 'grey'}
    def __init__(self, canvas, x, y,width,height):
        self.width = width
        self.height = height
        self.canvas = canvas
        self.angle = 0
        color = Brick.COLORS[1]
        item = self.canvas.create_rectangle(x,y ,x + self.width,y + self.height, fill=color, tags='brick')
        super(Brick, self).__init__(self.canvas, item)
    def hit(self):
        self.canvas.itemconfig(self.item, fill=Brick.COLORS[2])
class ATiger(GameObject):
    COLORS = {1: '#D2A438', 2: 'red'}
    def __init__(self, canvas, x, y, label_str,id,status):
        self.x = x
        self.y = y
        self.id = id
        self.angle = 0
        self.label_str = label_str
        self.color = ATiger.COLORS[1]
        self.canvas = canvas
        self.radius = 30
        self.status = status
        self.direction = [1, -1]
        item = canvas.create_oval(x - self.radius, y - self.radius,
                                  x + self.radius, y + self.radius,
                                  fill=self.color )
        super(ATiger, self).__init__(self.canvas, item,self.label_str ,self.id)
    def growl(self):
        self.canvas.itemconfig(self.item, fill=ATiger.COLORS[2])
    def move_to_right(self):
        super(ATiger, self).move(200,0)
        self.status = 'right'
    def move_to_left(self):
        super(ATiger, self).move(-200,0)
        self.status = 'left'
    def make_swift(self):
        if self.status == 'left':
            self.move_to_right()
        else:
            self.move_to_left()
        #print(self.status)
class Agent(GameObject):
    COLORS = {1: '#43C3DC',2:'pink', 3: 'red',4:'white'}
    def __init__(self, canvas, x, y,label_str,id,color_id,status):
        self.x = x
        self.y = y
        self.id = id
        self.status = status
        self.angle = 0
        self.label_str = label_str
        self.color = Agent.COLORS[color_id]
        self.canvas = canvas
        self.radius = 30
        self.direction = [1, -1]
        self.speed = 10
        item = canvas.create_oval(x - self.radius, y - self.radius,
                                  x + self.radius, y + self.radius,
                                  fill=self.color )
        super(Agent, self).__init__(self.canvas, item,self.label_str,self.id)
    def danger(self):
        self.canvas.itemconfig(self.item, fill=Agent.COLORS[3])
    def move_to_right(self):
        super(Agent, self).move(200,0)
        self.status = 'right'
    def move_to_left(self):
        super(Agent, self).move(-200,0)
        self.status = 'left'
    def make_swift(self):
        if self.status == 'left':
            self.move_to_right()
        else:
            self.move_to_left()
    def listen(self):
        self.canvas.itemconfig(self.item, fill=Agent.COLORS[4])
        #print(self.status)
class Game_Tree(tk.Frame):
    def __init__(self, master,width=None,height=None,ratio = None):
        super(Game_Tree, self).__init__(master,width=width,height=height)
        if width is None:
            self.width = 200
        else:
            self.width = width
        if height is None:
            self.height = 200
        else:
            self.height = height
        if not ratio is None:
            self.height = int(np.floor(self.height * ratio))
            self.width = int(np.floor(self.width * ratio))
        self.canvas = tk.Canvas(self, bg='#aaaaff',width=self.width,height=self.height)
        self.canvas.pack()
        self.pack()
        self.game_objects = dict()
        self.node_ids = dict()
    def set_width(self,width):
        self.width = width
        self.canvas.config(width=self.width, height=self.height)
    def set_height(self, height):
        self.height =  height
        self.canvas.config(width=self.width, height=self.height)
    def get_canvas(self):
        return self.canvas
    def get_game_objects(self):
        return self.game_objects
    def add_game_object(self,game_object):
        if not self.game_objects.__contains__(game_object.get_id()):
            self.game_objects[game_object.get_id()] = (game_object)
        else:
            self.game_objects[game_object.get_id()] = (game_object)
    def delete_game_object(self, game_object):
        if self.game_objects.__contains__(game_object.get_id()):
            self.game_objects.pop(game_object.get_id())
            self.canvas.delete(game_object.get_item())
    def clear_game_object(self):
        self.canvas.delete("all")
        self.game_objects = dict()
class Game_Node(GameObject):
    def __init__(self, canvas,x, y,label_str,id,radius=None,height=None,ratio = None):
        self.canvas = canvas
        self.id = id
        self.angle = 0
        self.label_str = label_str
        if radius  is None:
            self.radius = 30
        else:
            self.radius = radius
        if height is None:
            self.height = 20
        else:
            self.height = height
        if not ratio is None:
            self.height  = int(np.floor(self.height  * ratio))
            self.radius = int(np.floor(self.radius * ratio))
        self.x = x
        self.y = y
        self.direction = [1, -1]
        self.item = canvas.create_oval(x - self.radius, y - self.height,
                                  x + self.radius, y + self.height, fill= 'white')
        super(Game_Node, self).__init__(self.canvas, self.item,self.label_str,self.id)
class Game_Edge(GameObject):
    def __init__(self, canvas,x_start, y_start,x_end, y_end,label_str,id,length = None,ratio = None):
        l_x = np.abs(x_end - x_start)
        l_y = np.abs(y_end - y_start)
        if length  is None:
            self.length = 20
        else:
            self.length = length
        if not ratio is  None:
            self.length = int(np.floor(self.length*ratio))
        delta_x = self.length * l_x / np.sqrt(l_x * l_x + l_y * l_y)
        delta_y = self.length * l_y / np.sqrt(l_x * l_x + l_y * l_y)
        if x_end-x_start<0 and delta_x>0:
            delta_x = - delta_x
        self.x_start = x_start + delta_x
        self.y_start = y_start + delta_y
        self.x_end = x_end - delta_x
        self.y_end = y_end - delta_y
        self.id = id
        self.angle = 0
        self.label_str = label_str
        self.canvas = canvas
        self.item = self.canvas.create_line( self.x_start, self.y_start, self.x_end, self.y_end, arrow=tk.LAST, tags=id)
        super(Game_Edge, self).__init__(self.canvas, self.item,self.label_str,self.id)