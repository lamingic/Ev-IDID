# -*- coding: utf-8 -*-
# Copyright (C) 2001-2020
# Author: Biyang Ma <biyangma@stu.xmu.edu.cn> <mabiyang001@hotmail.com>
# Import the required libraries
# builtins
import math
import os
#
from xlsxwriter.workbook import Workbook
import time
import numpy as np
import datetime
import sys
import tkinter as tk
from tkinter import font,ttk,Menu,filedialog,scrolledtext,messagebox,Frame
from tqdm import tqdm
# from tkinter import *
from PIL import Image, ImageTk
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from widget import ScrolledFrame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
# self package
from Model import Model
from DID import DID
from IDID import IDID
from GameLib import *
from PolicyTree import PolicyTree
from Diversity import Diversity
import namespace as nsp
import popup_window as PW
# class
def iter_except(function, exception):
    """Works like builtin 2-argument `iter()`, but stops on `exception`."""
    try:
        while True:
            yield function()
    except exception:
        return
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
        the text labels.    """

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
class BkgrFrame(tk.Frame):
    def __init__(self, parent, file_path, width, height):
        super(BkgrFrame, self).__init__(parent, borderwidth=0, highlightthickness=0)
        self.canvas = tk.Canvas(self, width=width, height=height)
        self.canvas.pack()
        pil_img = Image.open(file_path)
        self.img = ImageTk.PhotoImage(pil_img.resize((width, height), Image.ANTIALIAS))
        self.bg = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)
    def add(self, widget, x, y):
        canvas_window = self.canvas.create_window(x, y, anchor=tk.NW, window=widget)
        return widget
class main_gui():
    def __init__(self):
        # Pyinstaller -F -w GUI_Main.py
        #Pyinstaller -F -w -i logo.ico GUI_Main.py
        self.winm = tk.Tk()
        self.winm.iconbitmap("./logo.ico")
        self.idid_title = "Enhanced I-DID Solutions through Evolutionary Computation"
        self.winm.title(self.idid_title)# Disable resizing the GUI by passing in False/False
        self.winm.resizable(False, False)# Enable resizing x-dimension, disable y-dimension
        # self.center_window(self.winm,1, 1)
        self.rowid = int(0)
        self.columnid = int(0)
        self.pnames = nsp.Name()
        self.CompareModels = dict()
        self.CompareDomains = dict()
        self.message_CompareModels = dict()
        self.message_CompareDomains = dict()
        self.SimMod = dict()
        self.message_CompareDomains['CB_list'] = self.pnames.CB_list_Domains
        self.message_CompareModels['CB_list'] = self.pnames.CB_list_Models
        self.CSV_data = dict()
        self.Models_Coverage = dict()
        self.Simmodels = None
        self.widgets = dict()
        self.welcome()
    def win_keep(self):
        self.winm.geometry("+330+30")
        self.winm.update()
        self.center_window(self.winm, 880, 900)
        self.winm.wm_attributes('-topmost', 1)
    def center_window(self, root,width,height,movex=None,movey = None):
        if movex is None:
            movex = 0
        if movey is None:
            movey = 0
        # get screen width and height
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        # calculate position x and y coordinates
        x = (screen_width / 2) - (width / 2)-movex
        y = (screen_height / 2) - (height / 2)-movey
        root.geometry('%dx%d+%d+%d' % (width, height, x, y))
    def welcome(self):
        self.win_welcome = tk.Toplevel()
        self.win_welcome.iconbitmap("./logo.ico")
        WIDTH, HEIGTH = 1200, 800
        self.center_window(self.win_welcome, WIDTH, HEIGTH)
        self.win_welcome.wm_attributes('-topmost', 1)
        self.win_welcome.title('Welcome to '+self.idid_title)
        IMAGE_PATH = './idid_platformEG.jpg'
        bkrgframe = BkgrFrame(self.win_welcome, IMAGE_PATH, WIDTH, HEIGTH)
        bkrgframe.pack()
        buttonFont1 = font.Font(family='Helvetica', size=24, weight='bold')
        button = bkrgframe.add(
            tk.Label(self.win_welcome, text=self.idid_title, width=50, font=buttonFont1, bg='#FFFFFF', relief="flat"),
            int(WIDTH * 0.10), int(HEIGTH * 0.10))
        buttonFont = font.Font(family='Helvetica', size=18, weight='bold')
        button = bkrgframe.add(tk.Button(self.win_welcome, text="Start", width=20,font=buttonFont,bg='#cacFFc',command=self.create_tabs), int(WIDTH*0.4),  int(HEIGTH*0.85))
    def about(self):
        messagebox.showinfo('About', 'Author:  Biyang Ma \n Email:  <mabiyang001@hotmail.com>')
    def popup_messeage(self,step = None):
        self.error = False
        if len(self.CompareModels) ==0:
            messagebox.showinfo('Error', 'You need to add models before running the test')
            self.error = True
        if len(self.CompareDomains) == 0:
            messagebox.showinfo('Error', 'You need to add domains before running the test')
            self.error = True
        if  step  == self.pnames.messeage_clues[0]:
            if self.Simmodels is None:
                self.error = True
                messagebox.showinfo('Error', 'You need to run test before doing coverage analysis')
        if step == self.pnames.messeage_clues[1]:
            if len(self.Models_Coverage) == 0:
                messagebox.showinfo('Error', 'You need to run coverage before plotting the coverage matrix')
                self.error = True
        if  step == self.pnames.messeage_clues[2]:
            if self.Simmodels is None:
                self.error = True
                messagebox.showinfo('Error', 'You need to run test before doing analysis')
        if  step == self.pnames.messeage_clues[3] or step == self.pnames.messeage_clues[4]:
            if self.Simmodels is None:
                self.error = True
            if self.error:
                messagebox.showinfo('Error', 'You need to run analysis before plotting the statistical data')
        if step == self.pnames.messeage_clues[5]:
            if self.Simmodels is None or self.dataset is None or len(self.dataset) ==0 :
                self.error = True
            if self.error:
                messagebox.showinfo('Error', 'You need to run analysis before plotting the tables')
        if  step == self.pnames.messeage_clues[6]:
            if self.Simmodels is None:
                self.error = True
                messagebox.showinfo('Error', 'You need to run test before doing diversity analysis')
        if  step == self.pnames.messeage_clues[7]:
            if self.Simmodels is None:
                self.error = True
            if self.error:
                messagebox.showinfo('Error', 'You need to run the models before plotting the policy tree')

        return self.error
    def update_rowid(self,num=None):
        if num is None:
            self.rowid = int(self.rowid + 1)
        else:
            self.rowid = int(self.rowid + num)
        return self.rowid
    def tab_frame(self,key,sufix,scroll = None):
        win_style = False
        if win_style:
            win = tk.Toplevel()
            win.wm_attributes('-topmost', 1)
            win.iconbitmap("./logo.ico")
            win.title(key)
            win.geometry("1000x900")
            win.resizable(False, False)
            tabControl = ttk.Notebook(win)  # Create Tab Control
            self.add_tab_frame(tabControl,key,sufix,scroll)
        else:
            self.add_tab_frame(self.tabControl,key,sufix,scroll)
    def add_tab_frame(self,tabControl,key,sufix,scroll = None):
        if scroll is None:
            self.widgets[key + '_tab' + sufix] = ttk.Frame(tabControl)  # Add a second tab
            self.widgets[key + '_tab' + sufix].grid(column=0, row=self.rowid, padx=8, pady=4)
            tabControl.add(self.widgets[key + '_tab' + sufix], text=key + sufix)  # Make second tab visible
            tabControl.pack(expand=1, fill="both")  # Pack to make visible
        else:
            self.widgets[key + '_tab_sfm' + sufix] = ttk.Frame(tabControl)  # Add a second tab
            self.widgets[key + '_tab_sfm' + sufix].grid(column=0, row=self.rowid, padx=8, pady=4)
            tabControl.add(self.widgets[key + '_tab_sfm' + sufix], text=key + sufix)  # Make second tab visible
            tabControl.pack(expand=1, fill="both")  # Pack to make visible
            root = self.widgets[key + '_tab_sfm' + sufix]
            self.widgets[key + '_tab_sf' + sufix] = ScrolledFrame(root, width=880, height=900)
            self.widgets[key + '_tab_sf' + sufix].pack(side="top", expand=1, fill="both")
            self.widgets[key + '_tab_sf' + sufix].bind_arrow_keys(root)
            self.widgets[key + '_tab_sf' + sufix].bind_scroll_wheel(root)
            self.widgets[key + '_tab' + sufix] = self.widgets[key + '_tab_sf' + sufix].display_widget(Frame)
    def create_tabs(self):
        self.win_welcome.destroy()
        self.winm.geometry("+330+30")
        self.winm.update()
        self.center_window(self.winm, 880, 900)
        self.winm.wm_attributes('-topmost', 1)
        # menu bar -------------------------------------
        menuBar = Menu(self.winm)
        self.winm.config(menu=menuBar)
        quitVar = tk.IntVar()
        fileMenu = Menu(menuBar, tearoff=0)
        fileMenu.add_checkbutton(label="退出", command=self.winm.destroy, variable=quitVar)
        menuBar.add_cascade(label="File", menu=fileMenu)
        helpMenu = Menu(menuBar, tearoff=0)
        aboutVar = tk.IntVar()
        helpMenu.add_checkbutton(label="About(关于)", command=self.about, variable=aboutVar)
        helpMenu.add_separator()
        helpMenu.add_checkbutton(label="Quit(退出)", command=self.winm_destroy, variable=quitVar)
        menuBar.add_cascade(label="Help", menu=helpMenu)
        # Tab Control introduced here --------------------------------------
        self.tabControl = ttk.Notebook(self.winm)  # Create Tab Control
        for i in range(0,len(self.pnames.GUI_Tabs)):
            tab = self.pnames.GUI_Tabs[i]
            sufix = self.pnames.GUI_Tabs_sc[i]
            self.tab_frame(tab, sufix,scroll = True)
        # ~ Tab Control introduced here -----------------------------------------
        self.Main_tab(0)
        self.Plot_tab(1)
        self.Diversity_tab(2)
        self.Diversity_sc_tab(3)
        self.PolicyTree_tab(4)
        self.winm.update()
        self.winm_size = self.winm.winfo_width()
        #-------------------------------------------------------------------------
    def winm_destroy(self):
        self.winm.destroy()
        self.winm = tk.Tk()
        self.winm.iconbitmap("./logo.ico")
        self.idid_title = "Enhanced I-DID Solutions through Evolutionary Computation"
        self.winm.title(self.idid_title)  # Disable resizing the GUI by passing in False/False
        self.winm.resizable(False, False)  # Enable resizing x-dimension, disable y-dimension
        self.welcome()
    # main tab
    def Main_tab(self,index):
        key = self.pnames.GUI_Tabs[index]
        sufix = self.pnames.GUI_Tabs_sc[index]
        tab = self.widgets[key + '_tab' + sufix]
        # domains
        self.create_domain_frame(tab,0)
        # models
        self.create_model_frame(tab,1)
        # sim mode
        self.create_simmulation_frame(tab,2)
        # play
        self.create_play_frame(tab,3)
        # message
        self.create_message_frame(tab,4)
        ## plot the ineraction of agents
        self.create_figure_frame(tab,5)
    # domain frame
    def create_domain_frame(self,tab,id):
        master = self.create_frame(tab,id)
        frame_name = self.pnames.GUI_Frames[id]
        self.message_CompareDomains['chosen'] =tk.StringVar()
        ttk.Label(master, text="Compare Domains:", width=20).grid(column=0, row=self.update_rowid())
        self.message_CompareDomains['CB_'+frame_name] = ttk.Combobox(master, width=18, textvariable=self.message_CompareDomains['chosen'])
        self.message_CompareDomains['CB_'+frame_name]['values'] = list()
        self.message_CompareDomains['CB_'+frame_name].grid(column=1, row=self.rowid, padx=1, pady=2)

        add = ttk.Button(master, text="Add", width=20, command=self.add_compare_domains)
        add.grid(column=0, row=self.update_rowid(), padx=1, pady=2)
        delete = ttk.Button(master, text="Delete", width=20, command=self.delete_compare_domains)
        delete.grid(column=1, row=self.rowid, padx=1,  pady=2)
    def add_compare_domains(self):
        self.clear_test_record()
        self.set_None()
        frame_name = self.pnames.GUI_Frames[0]
        self.domain_parameters = nsp.DomainParameters()
        PW.popup_window(frame_name,self.domain_parameters,self.CompareDomains,self.message_CompareDomains)
    def delete_compare_domains(self):
        self.clear_test_record()
        self.set_None()
        key = self.message_CompareDomains['chosen'].get()
        if self.CompareDomains.__contains__(key):
            self.CompareDomains.pop(key)
        if len(list(self.CompareDomains.keys())) > 0:
            messagebox = list(self.CompareDomains.keys())
        else:
            messagebox = list()
        for ei in self.message_CompareDomains['CB_list']:
            self.message_CompareDomains[ei]['values'] = messagebox
            if len(messagebox)>0:
                self.message_CompareDomains[ei].current(0)
            self.message_CompareDomains[ei].update()
    # model frame
    def create_model_frame(self,tab,id):
        master = self.create_frame(tab, id)
        frame_name = self.pnames.GUI_Frames[id]

        self.Solvers = dict.fromkeys(self.pnames.Solver_types,dict())
        for key in self.pnames.Solver_types.keys():
            self.update_rowid()
            masteri = ttk.LabelFrame(master, text=key, width=20)
            masteri.grid(column=0, row=self.rowid, padx=0, pady=4, columnspan=2, sticky=tk.E)
            self.Solvers[key] = dict()
            for keyi in self.pnames.Solver_types.get(key).keys():
                self.Solvers.get(key)[keyi] = tk.StringVar()
                self.update_rowid()
                ttk.Label(masteri, text=keyi, width=20).grid(column=0, row=self.rowid, padx=0)
                model_solver_Chosen = ttk.Combobox(masteri, width=18, textvariable=self.Solvers.get(key)[keyi])
                if keyi.__contains__('extending policy trees'):
                    model_solver_Chosen['values'] = list(self.pnames.Extension.values())
                else:
                    model_solver_Chosen['values'] = list(self.pnames.Solver.values())
                model_solver_Chosen.grid(column=1, row=self.rowid, padx=0, pady=2)
                model_solver_Chosen.current(0)

        self.update_rowid()
        self.message_CompareModels['chosen'] = tk.StringVar()
        ttk.Label(master, text="Compare Models:", width=18).grid(column=0, row=self.rowid, padx=0, sticky=tk.W)
        self.message_CompareModels['CB_'+frame_name] = ttk.Combobox(master, width=18, textvariable=self.message_CompareModels['chosen'])
        self.message_CompareModels['CB_'+frame_name]['values'] = list()
        self.message_CompareModels['CB_'+frame_name].grid(column=1, row=self.rowid, padx=0, pady=2, sticky=tk.W)

        self.update_rowid()
        add = ttk.Button(master, text="Add", width=18, command=self.add_compare_models)
        add.grid(column=0, row=self.rowid, padx=0, pady=2, sticky=tk.W)
        delete = ttk.Button(master, text="Delete", width=20, command=self.delete_compare_models)
        delete.grid(column=1, row=self.rowid, padx=0,  pady=2, sticky=tk.W)

        self.update_rowid()
        self.Extend_mode = tk.IntVar()
        check = tk.Checkbutton(master, text="Extend_mode", width=18, variable=self.Extend_mode)
        check.deselect()
        check.grid(column=0, row=self.rowid,  sticky=tk.W, padx=0)
        confirm = ttk.Button(master, text="Select", width=20, command=self.popup_window_SBM)
        confirm.grid(column=1, row=self.rowid, padx=0, pady=2, sticky=tk.W)
    def add_compare_models(self):
        self.clear_test_record()
        self.set_None()
        self.ModelParameters = nsp.ModelParameters()
        index = str(len(self.CompareModels) + 1)
        name = 'IDID @' + index + '-'
        for key in self.Solvers.keys():
            for keyi in self.Solvers.get(key).keys():
                self.ModelParameters.parameters[key][keyi]=self.Solvers.get(key).get(keyi).get()
                self.ModelParameters.values[key][keyi]['name'] = self.Solvers.get(key).get(keyi).get()
                name = name + '-' + self.ModelParameters.values[key][keyi]['name']
                if self.Solvers[key][keyi].get().__contains__('GA'):
                    self.popup_window_GA(self.ModelParameters.values.get(key).get(keyi))
                if self.Solvers[key][keyi].get().__contains__('MD'):
                    self.popup_window_MD(self.ModelParameters.values.get(key).get(keyi))
        self.ModelParameters.Name = name
        self.CompareModels[self.ModelParameters.Name] = self.ModelParameters
        # message
        for ei in self.message_CompareModels['CB_list']:
            self.message_CompareModels[ei]['values'] = list(self.CompareModels.keys())
            self.message_CompareModels[ei].current(0)
            self.message_CompareModels[ei].update()
    def delete_compare_models(self):
        self.clear_test_record()
        self.set_None()
        if len(list(self.CompareModels.keys())) > 0:
            key = self.message_CompareModels['chosen'].get()
            print(key)
            if self.CompareModels.__contains__(key):
                self.CompareModels.pop(key)
            if len(list(self.CompareModels.keys())) >0:
                for ei in self.message_CompareModels['CB_list']:
                    self.message_CompareModels[ei]['values'] = list(self.CompareModels.keys())
                    self.message_CompareModels[ei].current(0)
            else:
                for ei in self.message_CompareModels['CB_list']:
                    self.message_CompareModels[ei]['values'] = list()
            for ei in self.message_CompareModels['CB_list']:
                self.message_CompareModels[ei].update()
    def popup_window_MD(self,solver):
        initialdir = os.getcwd()
        solver['parameters'] = filedialog.askopenfilename(initialdir=initialdir, title="Select file",
                                                   filetypes=(("txt files", "*.txt"), ("all files", "*.*")))
    def popup_window_GA(self, solver):
        if solver['name'] == self.pnames.Solver[2]:
            solver['parameters'] = nsp.GA_ps()
        if solver['name']==self.pnames.Solver[3] or solver['name']==self.pnames.Solver[4]:
            solver['parameters'] = nsp.GGA_ps()
        if solver['name'] == self.pnames.Solver[5]:
            solver['parameters'] = nsp.MGA_ps()
        if solver['name'] == self.pnames.Solver[6] or solver['name'] == self.pnames.Solver[7]:
            solver['parameters'] = nsp.MGGA_ps()
        PW.popup_window(solver['type']+'@'+solver['pointer'], solver['parameters'])
    def popup_window_SBM(self):
        if self.Extend_mode:
            if len(list(self.CompareModels.keys())) > 0:
                win = tk.Toplevel()
                win.iconbitmap("./logo.ico")
                win.wm_attributes('-topmost', 1)
                master = ttk.LabelFrame(win, text='select base model')
                rowid = 0
                master.grid(column=0, row=rowid, padx=8, pady=4)

                self.selected_main_model = {'parent':'','child':''}
                ttk.Label(master, text="Base Models:", width=20).grid(column=0, row=rowid)
                self.selected_main_model['parent'] = tk.StringVar()
                pa_Chosen = ttk.Combobox(master, width=13, textvariable=self.selected_main_model['parent'])
                pa_Chosen['values'] = list(self.CompareModels.keys())
                pa_Chosen.grid(column=1, row=rowid, padx=1, pady=2)

                rowid = rowid + 1
                ttk.Label(master, text="Extend Models:", width=20).grid(column=0, row=rowid)
                self.selected_main_model['child'] = tk.StringVar()
                ch_Chosen = ttk.Combobox(master, width=13, textvariable=self.selected_main_model['child'])
                ch_Chosen['values'] = list(self.CompareModels.keys())
                ch_Chosen.grid(column=1, row=rowid, padx=1, pady=2)

                rowid = rowid + 1
                button_confirm = ttk.Button(master, text="Confirm", command=self.select_base_model)
                button_confirm.grid(column=0, row=rowid, padx=1, pady=2)
                button_close = ttk.Button(master, text="Close", command=win.destroy)
                button_close.grid(column=1, row=rowid, padx=1, pady=2)
    def select_base_model(self):
        ch = self.selected_main_model['child'].get()
        pa = self.selected_main_model['parent'].get()
        self.CompareModels.get(ch).Base_model = pa
    # simulation frame
    def create_simmulation_frame(self,tab,id):
        master = self.create_frame(tab, id)
        frame_name = self.pnames.GUI_Frames[id]

        self.SimMod[frame_name] = nsp.SimMod()
        for key in self.SimMod[frame_name].parameters_type.keys():
            self.update_rowid()
            ttk.Label(master, text=key, width=20).grid(column=0, row=self.rowid)
            mod_Chosen = ttk.Combobox(master, width=18, textvariable=self.SimMod[frame_name].parameters_type.get(key))
            mod_Chosen['values'] = list(self.pnames.Sim_mode.values())
            mod_Chosen.grid(column=1, row=self.rowid, padx=1, pady=2)
            mod_Chosen.current(1)
        self.update_rowid()
        ttk.Label(master, text=" Test Style:", width=20).grid(column=0, row=self.rowid)
        self.widgets['test_style_'+frame_name] = tk.StringVar()
        test_style_Chosen = ttk.Combobox(master, width=18, textvariable=self.widgets['test_style_'+frame_name])
        test_style_Chosen['values'] = list(self.pnames.Test_style.values())
        test_style_Chosen.grid(column=1, row=self.rowid, padx=1, pady=2)
        test_style_Chosen.current(1)

        self.update_rowid()
        run = ttk.Button(master, text="Run", width=20, command=self.exp)
        run.grid(column=0, row=self.rowid, padx=1,  pady=2, columnspan=1)
        test = ttk.Button(master, text="Analysis", command=lambda :self.analysis(frame_name))
        test.grid(column=1, row=self.rowid, padx=1, pady=2, sticky='WE')

        self.update_rowid()
        coverage = ttk.Button(master, text="Coverage", width=20, command=self.coverage)
        coverage.grid(column=0, row=self.rowid, padx=1, pady=2, columnspan=1)
        self.widgets['Analysis_all_'+frame_name] = tk.IntVar()
        check3 = tk.Checkbutton(master, text="Analysis_all", width=9, variable=self.widgets['Analysis_all_'+frame_name])
        check3.deselect()
        check3.grid(column=1, row=self.rowid, sticky=tk.W)
    def clear_test_record(self):
        [self.SimMod.get(key).clear() for key in self.SimMod.keys()]
    def exp(self):
        if self.popup_messeage():
            return -1
        if len(self.CompareModels.keys()) > 0:
            self.clear_test_record()
            self.dataset = dict()
            self.print('>  preparing')
            self.Simmodels = nsp.SimModels(self.CompareDomains,self.CompareModels)
            self.print('>> Testing')
            for keyd in self.CompareDomains.keys():
                self.print('>>> run model @ domain: ' + keyd )
                for i in range(0,len(self.Simmodels.base)):
                    keym = self.Simmodels.base.get(i)
                    self.print('>>>> run model @ domain: ' + keyd + ' @ model: ' + keym )
                    self.Simmodels.IDID[keyd][keym]= IDID(self.CompareDomains.get(keyd),self.CompareModels.get(keym),self.scr_message)#parameters
                    if self.CompareModels.get(keym).Base_model != '':
                       self.print('>>>>>load model from base model') # LOAD BASE MODEL
                       keyb = self.CompareModels.get(keym).Base_model
                       self.Simmodels.IDID[keyd][keym].did.base_model = self.Simmodels.IDID[keyd][keyb].did
                    # gen i's policy tree
                    self.Simmodels.IDID[keyd][keym].gen_pathes()
                    self.print('>>>> finish model @ domain: ' + keyd + ' @ model: ' + keym)
        self.print('>>>> finish all >>>>')
        self.win_keep()
    def analysis(self,frame_name):
        if self.popup_messeage('analysis'):
            return -1
        if self.widgets['Analysis_all_'+frame_name].get():
            test_styles = self.pnames.Test_style.values()
        else:
            test_styles = [self.widgets['test_style_'+frame_name].get()]
        self.SimMod[frame_name].update()
        sim_mode_i_set, sim_mode_j_set = self.get_sim_mode(frame_name)
        for test_style in test_styles:
            self.SimMod[frame_name].add(test_style,sim_mode_i_set, sim_mode_j_set)
        # self.popup_window_Table()
        self.write_csv_runtimes()
        for test_style in test_styles:
            self.on_offline_test(test_style,sim_mode_i_set,sim_mode_j_set)
            self.write_csv_results(test_style,sim_mode_i_set, sim_mode_j_set)
    def write_csv_results(self,test_style,sim_mode_i_set, sim_mode_j_set):
        # test_style = self.widgets['test_style_sim'].get()
        # write to excel
        # Create an new Excel file and add a worksheet.
        workbook = Workbook(self.pnames.Save_filepath+test_style+'.xlsx')
        worksheet = workbook.add_worksheet()
        # Add a bold format to use to highlight cells. for headings
        bold = workbook.add_format({'bold': True,'align':'center','valign':'vcenter'})
        # Text with formatting.
        worksheet.merge_range('A1:A2', 'Algorithm', bold)
        worksheet.merge_range('B1:B2', 'Combat Style', bold)
        # Widen the first column to make the text clearer.
        worksheet.set_column('A:A', 30)
        worksheet.set_column('B:B', 30)
        #row headings
        # row headings
        self.rows_dict = dict()
        alg_set = set()
        for key in self.CompareModels.keys():
            alg_set.add(self.CompareModels.get(key).Name)
        # sim_mode_i_set, sim_mode_j_set = self.get_sim_mode()
        len_alg_steps = len(sim_mode_i_set)*len(sim_mode_j_set)
        start = 3
        for key in alg_set:
            index_str = 'A'+str(start) + ':A'+ str(start+len_alg_steps-1)
            self.rows_dict[key] = dict()
            # worksheet.merge_range(index_str, key, bold)
            if len_alg_steps > 1:
                worksheet.merge_range(index_str, key, bold)
            else:
                worksheet.write(index_str, key, bold)
            i = 1
            for si in sim_mode_i_set:
                for sj in sim_mode_j_set:
                    sij = si +' vs '+sj
                    index_str = 'B' + str(start - 1 + i)
                    worksheet.write(index_str, sij)
                    self.rows_dict[key][sij] = str(start - 1 + i)
                    i = i+1
            start = start + len_alg_steps
        # columns headings
        domain_set = set()
        for keyd in self.CompareDomains.keys():
            domain_set.add(self.CompareDomains.get(keyd).values.get(self.pnames.domain_parameters[0]))
        domain_horizon_list = dict()
        len_domain_horizon_list = dict()
        for key in domain_set:
            horizon_list = list()
            for keyd in self.CompareDomains.keys():
                if self.CompareDomains.get(keyd).values.get(self.pnames.domain_parameters[0]) == key:
                    horizon_list.append(self.CompareDomains.get(keyd).values.get(self.pnames.domain_parameters[4]))
            domain_horizon_list[key] = horizon_list
            len_domain_horizon_list[key] = len(horizon_list)
        total_len = sum(len_domain_horizon_list.values())
        Columns = list()
        for s in self.iter_all_strings():
            Columns.append(s.upper())
            if len(Columns) ==  total_len+3:
                break
        start = 2
        self.columns_dict = dict()
        columns_dict = dict()
        for key in domain_set:
            len_d = len_domain_horizon_list.get(key)
            index_str = Columns[start]+ str(1) + ':'+Columns[start+len_d-1] + str(1)
            if len_d>1:
                worksheet.merge_range(index_str, key, bold)
            else:
                worksheet.write(index_str, key, bold)
            for i in range(0,len(domain_horizon_list.get(key))):
                h  = domain_horizon_list.get(key)[i]
                index_str = Columns[start+i] + str(2)
                worksheet.write(index_str, h)
                for keyd in self.CompareDomains.keys():
                    if self.CompareDomains.get(keyd).values.get(self.pnames.domain_parameters[0]) == key and self.CompareDomains.get(keyd).values.get(self.pnames.domain_parameters[4]) == h:
                        self.columns_dict[keyd]  = Columns[start+i]
                        columns_dict[keyd] = start + i
            start = start + len_d
        # results
        for keyd in self.CompareDomains.keys():
            column = self.columns_dict.get(keyd)
            for keym in self.CompareModels.keys():
                result = self.Simmodels.IDID[keyd][keym].result.rewards.get(test_style)
                for si in sim_mode_i_set:
                    for sj in sim_mode_j_set:
                        key = si + ' vs ' + sj
                        c = 'N(' + str(round(np.mean(result.get(key).get('mean')), 2)) +','+str(round(np.std(result.get(key).get('std')), 2))+')'
                        rowt = self.rows_dict[keym].get(key)
                        worksheet.write(column +rowt, c)
        self.create_dataset(domain_set, domain_horizon_list, columns_dict,
                       self.rows_dict,test_style,test_style, sim_mode_i_set, sim_mode_j_set)
        workbook.close()
    def create_dataset(self,domain_set,domain_horizon_list,columns_dict,rows_dict,filename,test_style = None,sim_mode_i_set= None, sim_mode_j_set= None):
        dataset = dict()#
        dataset[0] = ['Algorithm', 'Combat Style']#
        dataset[1] = ['-','-']#
        lined = []#
        lineh = []#
        for key in domain_set:
            for i in range(0,len(domain_horizon_list.get(key))):
                h  = domain_horizon_list.get(key)[i]
                lined.append(key)#
                lineh.append(h)#
        [dataset[0].append(ei) for ei in lined]#
        [dataset[1].append(ei) for ei in lineh]#
        # results
        for keyd in self.CompareDomains.keys():
            column = columns_dict.get(keyd)
            print(keyd,column)
            for keym in self.CompareModels.keys():
                result = self.Simmodels.IDID[keyd][keym].result.rewards.get(test_style)
                for si in sim_mode_i_set:
                    for sj in sim_mode_j_set:
                        key = si + ' vs ' + sj
                        c = 'N(' + str(round(np.mean(result.get(key).get('mean')), 2)) + ',' + str(
                            round(np.std(result.get(key).get('std')), 2)) + ')'
                        rowt = rows_dict[keym].get(key)
                        line = [0 for ei in dataset[0]]
                        line[0] = keym
                        line[1] = key
                        line[column] = c
                        if dataset.__contains__(int(rowt)-1):
                            linec = dataset[int(rowt)-1]
                            linec[column] = c
                            dataset[int(rowt) - 1] = linec
                        else:
                            dataset[int(rowt)-1] = line
                        # print(int(rowt)-1,column,line)
        # for key in dataset:
        #     print(key,dataset[key])
        data = dict()
        data['columns'] = len(dataset[0])
        data['rows'] = len(list(dataset.keys()))
        data['data'] = dataset
        self.dataset[filename] = data
    def write_csv_runtimes(self):
        # write to excel
        # import xlsxwriter
        # Create an new Excel file and add a worksheet.
        workbook = Workbook(self.pnames.Save_filepath+self.pnames.table_files[2]+'.xlsx')
        worksheet = workbook.add_worksheet()
        # Add a bold format to use to highlight cells. for headings
        bold = workbook.add_format({'bold': True,'align':'center','valign':'vcenter'})
        # Text with formatting.
        worksheet.merge_range('A1:A2', 'Algorithm', bold)
        worksheet.merge_range('B1:B2', 'DID/IDID', bold)
        worksheet.merge_range('C1:C2', 'Progress', bold)
        # Widen the first column to make the text clearer.
        worksheet.set_column('A:A', 30)
        worksheet.set_column('C:C', 24)
        # row headings
        self.rows_dict = dict()
        alg_set = set()
        for key in self.CompareModels.keys():
            alg_set.add(self.CompareModels.get(key).Name)
        did_steps = list(self.pnames.Steps.get('DID').values())
        idid_steps = list(self.pnames.Steps.get('IDID').values())
        len_did_steps = len(did_steps)
        len_idid_steps = len(idid_steps)
        len_alg_steps = len_did_steps+len_idid_steps
        start = 3
        for key in alg_set:
            index_str = 'A'+str(start) + ':A'+ str(start+len_alg_steps-1)
            self.rows_dict[key] = {'DID': dict(), 'IDID': dict(), 'Index':str(start)}
            worksheet.merge_range(index_str, key, bold)
            index_str = 'B' + str(start) + ':B' + str(start + len_did_steps - 1)
            worksheet.merge_range(index_str, 'DID', bold)
            index_str = 'B' + str(start+ len_did_steps) + ':B' + str(start ++ len_did_steps+ len_idid_steps - 1)
            worksheet.merge_range(index_str, 'IDID', bold)
            for i in range(1,len(self.pnames.Steps.get('DID').keys())+1):
                index_str = 'C' + str(start-1+i)
                worksheet.write(index_str, self.pnames.Steps.get('DID').get(i))
                self.rows_dict.get(key).get('DID')[self.pnames.Steps.get('DID').get(i)] =   str(start-1+i)
            for i in range(1,len(self.pnames.Steps.get('IDID').keys())+1):
                index_str = 'C' + str(start+ len_did_steps-1+i)
                worksheet.write(index_str, self.pnames.Steps.get('IDID').get(i))
                self.rows_dict.get(key).get('IDID')[self.pnames.Steps.get('IDID').get(i)] = str(start+ len_did_steps-1+i)
            start = start + len_alg_steps
        # columns headings
        domain_set = set()
        for keyd in self.CompareDomains.keys():
            domain_set.add(self.CompareDomains.get(keyd).values.get(self.pnames.domain_parameters[0]))
        domain_horizon_list = dict()
        len_domain_horizon_list = dict()
        for key in domain_set:
            horizon_list = list()
            for keyd in self.CompareDomains.keys():
                if self.CompareDomains.get(keyd).values.get(self.pnames.domain_parameters[0]) == key:
                    horizon_list.append(self.CompareDomains.get(keyd).values.get(self.pnames.domain_parameters[4]))
            domain_horizon_list[key] = horizon_list
            len_domain_horizon_list[key] = len(horizon_list)
        total_len = sum(len_domain_horizon_list.values())
        Columns = list()
        for s in self.iter_all_strings():
            Columns.append(s.upper())
            if len(Columns) ==  total_len+3:
                break
        start = 3
        self.columns_dict = dict()
        columns_dict = dict()
        for key in domain_set:
            len_d = len_domain_horizon_list.get(key)
            index_str = Columns[start]+ str(1) + ':'+Columns[start+len_d-1] + str(1)

            if len_d>1:
                worksheet.merge_range(index_str, key, bold)
            else:
                worksheet.write(index_str, key, bold)
            for i in range(0,len(domain_horizon_list.get(key))):
                h  = domain_horizon_list.get(key)[i]
                index_str = Columns[start+i] + str(2)
                worksheet.write(index_str, h)
                for keyd in self.CompareDomains.keys():
                    if self.CompareDomains.get(keyd).values.get(self.pnames.domain_parameters[0]) == key and self.CompareDomains.get(keyd).values.get(self.pnames.domain_parameters[4]) == h:
                        self.columns_dict[keyd]  = Columns[start+i]
                        columns_dict[keyd] = start + i
            start = start + len_d
        # times
        for keyd in self.CompareDomains.keys():
            column = self.columns_dict.get(keyd)
            for keym in self.CompareModels.keys():
                times = self.Simmodels.IDID[keyd][keym].times
                row = self.rows_dict[keym].get('IDID')
                for t in times:
                    # print(t)
                    rowt = row.get(t)
                    worksheet.write(column +rowt, times.get(t)*24*3600)
                times = self.Simmodels.IDID[keyd][keym].did.times
                row = self.rows_dict[keym].get('DID')
                for t in times:
                    rowt = row.get(t)
                    worksheet.write(column + rowt, times.get(t)*24*3600)
        self.create_dataset_time( domain_set, domain_horizon_list, columns_dict,self.rows_dict, self.pnames.table_files[2])
        workbook.close()
    def create_dataset_time(self,domain_set,domain_horizon_list,columns_dict,rows_dict,filename):
        dataset = dict()#
        dataset[0] = ['Algorithm', 'DID/IDID','Progress']#
        dataset[1] = ['-','-','-']#
        lined = []#
        lineh = []#
        for key in domain_set:
            for i in range(0,len(domain_horizon_list.get(key))):
                h  = domain_horizon_list.get(key)[i]
                lined.append(key)#
                lineh.append(h)#
        [dataset[0].append(ei) for ei in lined]#
        [dataset[1].append(ei) for ei in lineh]#
        # time
        for keyd in self.CompareDomains.keys():
            column = columns_dict.get(keyd)
            for keym in self.CompareModels.keys():
                times = self.Simmodels.IDID[keyd][keym].did.times
                row = self.rows_dict[keym].get('DID')
                for t in times:
                    rowt = row.get(t)
                    line = [0 for ei in dataset[0]]
                    line[0] = keym
                    line[1] = 'DID'
                    line[2] = t
                    line[column] = times.get(t)
                    if dataset.__contains__(int(rowt) - 1):
                        linec = dataset[int(rowt) - 1]
                        linec[column] = times.get(t)
                        dataset[int(rowt) - 1] = linec
                    else:
                        dataset[int(rowt) - 1] = line
                    # dataset[int(rowt) - 1] = line
                times = self.Simmodels.IDID[keyd][keym].times
                row = self.rows_dict[keym].get('IDID')
                for t in times:
                    rowt = row.get(t)
                    line = [0 for ei in dataset[0]]
                    line[0] = keym
                    line[1] = 'IDID'
                    line[2] = t
                    line[column] = times.get(t)
                    if dataset.__contains__(int(rowt) - 1):
                        linec = dataset[int(rowt) - 1]
                        linec[column] = times.get(t)
                        dataset[int(rowt) - 1] = linec
                    else:
                        dataset[int(rowt) - 1] = line
                    # dataset[int(rowt) - 1] = line
        # for key in dataset:
        #     print(key,dataset[key])
        data = dict()
        data['columns'] = len(dataset[0])
        data['rows'] = len(list(dataset.keys()))
        data['data'] = dataset
        self.dataset[filename] = data
    def iter_all_strings(self):
        from string import ascii_lowercase
        import itertools
        for size in itertools.count(1):
            for s in itertools.product(ascii_lowercase, repeat=size):
                yield "".join(s)
    def get_sim_mode(self,frame_name):
        if self.widgets['Analysis_all_'+frame_name].get():
            sim_mode_i = 'all'
            sim_mode_j = 'all'
        else:
            sim_mode_i = self.SimMod[frame_name].values.get(self.SimMod[frame_name].parameters[0])
            sim_mode_j = self.SimMod[frame_name].values.get(self.SimMod[frame_name].parameters[1])
        if sim_mode_i == 'all':
            sim_mode_i_set = [s for s in self.pnames.Sim_mode.values() if s !='all']
        else:
            sim_mode_i_set = [sim_mode_i]
        if sim_mode_j == 'all':
            sim_mode_j_set = [s for s in self.pnames.Sim_mode.values() if s != 'all']
        else:
            sim_mode_j_set = [sim_mode_j]
        return sim_mode_i_set,sim_mode_j_set
    def on_offline_test(self,test_style,sim_mode_i_set,sim_mode_j_set):
        self.print('-------------- analysis now --------------------------')
        self.print('>>> test @:' +test_style)
        for keyd in self.CompareDomains.keys():
            self.print('>>> run test @ domain: ' + keyd)
            domain = self.CompareDomains.get(keyd)
            if test_style == self.pnames.Test_style.get(2):
                mp = nsp.ModelParameters()
                mp.default('IDID @ '+domain.Tostring() +test_style)
                self.Simmodels.DID[keyd] = IDID(domain,mp,self.scr_message)  # parameters
                self.Simmodels.DID[keyd].gen_pathes_online()
            for keym in self.CompareModels.keys():
                self.print('>>> run test @ model: ' + keym)
                idid_test = self.Simmodels.IDID[keyd][keym]
                if test_style == self.pnames.Test_style.get(1):
                    idid = self.Simmodels.IDID[keyd][keym]
                if test_style == self.pnames.Test_style.get(2):
                    idid = self.Simmodels.DID[keyd]
                    idid.dbn.result = self.Simmodels.IDID[keyd][keym].dbn.result
                    if sim_mode_i_set.__contains__(self.pnames.Sim_mode[3]) or  sim_mode_j_set.__contains__(self.pnames.Sim_mode[3]):
                        idid.expansion(expansion_flag=False, policy_tree=idid.dbn.result.get('policy_tree'))
                self.test_sub(domain, idid,idid_test, sim_mode_i_set, sim_mode_j_set,test_style)
        self.print('-------------- analysis done --------------------------')
    def test_sub(self,domain,idid,idid_test,sim_mode_i_set,sim_mode_j_set,test_style):
        policy_dict_i = idid.dbn.result.get('policy_dict')
        policy_dict_j = idid.did.dbn.result.get('policy_dict')
        policy_path_weight_i = idid.dbn.result.get('policy_path_weight')
        policy_path_weight_j = idid.did.dbn.result.get('policy_path_weight')
        bar = tqdm(total=int(len(sim_mode_i_set)*len(sim_mode_j_set)*len(policy_dict_i.keys()) * len(policy_dict_j.keys())))
        for si in sim_mode_i_set:
            for sj in sim_mode_j_set:
                self.print('agent i simulate @: ' +  si)
                self.print('agent j simulate @: ' +  sj)
                rewards = {'mean':'','std':'','var':''}
                r_mean = np.zeros([len(policy_dict_i.keys()), len(policy_dict_j.keys())])
                r_var = np.zeros([len(policy_dict_i.keys()), len(policy_dict_j.keys())])
                r_std = np.zeros([len(policy_dict_i.keys()), len(policy_dict_j.keys())])
                for modi in policy_dict_i.keys():
                    pathes_i = policy_dict_i.get(modi)
                    weights_i = policy_path_weight_i.get(modi)
                    pathes_i, weights_i = self.gen_pathes_weights(pathes_i, weights_i,  si)
                    for modj in policy_dict_j.keys():
                        pathes_j = policy_dict_j.get(modj)
                        weights_j = policy_path_weight_j.get(modj)
                        pathes_j, weights_j = self.gen_pathes_weights(pathes_j, weights_j,  sj)
                        if si == self.pnames.Sim_mode[3] or sj == self.pnames.Sim_mode[3]:
                            reward = self.get_rewards_gui_tree(idid, domain, pathes_i, pathes_j, weights_i, weights_j,si,sj,modi,modj)
                        else:
                            reward = self.get_rewards_gui(idid,domain,pathes_i, pathes_j, weights_i, weights_j)
                        r_mean[modi, modj] = np.mean(reward)
                        r_var[modi, modj] = np.var(reward)
                        r_std[modi, modj] = np.std(reward)
                        bar.update(1)
                rewards['mean'] = r_mean
                rewards['var'] = r_mean
                rewards['std'] = r_mean
                key = si+' vs '+ sj
                idid_test.result.rewards.get(test_style)[key]= rewards
        bar.close()
    def get_rewards_gui_tree(self,idid,domain, pathes_i, pathes_j, weights_i, weights_j,si,sj,modi,modj):
        if si == self.pnames.Sim_mode[3]:
            evid_value = idid.dbn.result.get('policy_tree').roots_map.get(str(modi))
            self.enter_tree_evds(idid,idid.dbn_sim.MODPrefix,evid_value)
        else:
            self.enter_evidences_gui_sim(idid, idid.dbn_sim.evidences, pathes_i[0])
        if sj == self.pnames.Sim_mode[3]:
            evid_value = idid.did.dbn.result.get('policy_tree').roots_map.get(str(modj))
            self.enter_tree_evds(idid, idid.dbn.MODPrefix, evid_value)
        else:
            self.enter_evidences_gui_sim(idid, ['O' + ei for ei in idid.did.dbn.evidences], pathes_j[0])
        reward = list()
        num_test = domain.values.get(self.pnames.domain_parameters[3])
        belief = domain.beliefs.get('test')
        for i in range(0, num_test):
            ei = "S" + str(idid.dbn_sim.horizon)
            # idid.dbn_sim.net.set_node_definition(ei, belief[i])
            idid.dbn_sim.net.set_virtual_evidence(ei, belief[i])
            idid.dbn_sim.net.update_beliefs()
            reward_path = list()
            for hi in range(0, idid.dbn_sim.horizon):
                reward_path.append(np.mean(np.array(idid.dbn_sim.net.get_node_value("U" + str(hi + 1)))))
            prob_path = 1.0
            reward.append(np.sum(reward_path) * prob_path)
        idid.dbn_sim.net.clear_all_evidence()
        return reward
    def enter_tree_evds(self,idid,MODPrefix,evid_value):
        node_id_mod = MODPrefix + str(idid.dbn_sim.horizon)
        node_states = idid.dbn_sim.net.get_outcome_ids(node_id_mod)
        index = node_states.index(evid_value)
        idid.dbn_sim.net.set_evidence(node_id_mod, int(index))
    def get_rewards_gui(self,idid,domain, pathes_i, pathes_j, weights_i, weights_j):
        reward = list()
        num_test = domain.values.get(self.pnames.domain_parameters[3])
        belief = domain.beliefs.get('test')
        for pi in range(0, len(pathes_i)):
            path_i = pathes_i[pi]
            weight_i = weights_i[pi]
            self.enter_evidences_gui(idid,idid.dbn.evidences, path_i)
            for pj in range(0, len(pathes_j)):
                path_j = pathes_j[pj]
                weight_j = weights_j[pj]
                self.enter_evidences_gui(idid,['O' + ei for ei in idid.did.dbn.evidences], path_j)
                for i in range(0, num_test):
                    ei = "S" + str(idid.dbn.horizon)
                    # idid.dbn.net.set_node_definition(ei, belief[i])
                    idid.dbn.net.set_virtual_evidence(ei, belief[i])
                    idid.dbn.net.update_beliefs()
                    reward_path = list()
                    for hi in range(0, idid.dbn.horizon):
                        reward_path.append(np.mean(np.array(idid.dbn.net.get_node_value("U" + str(hi + 1)))))
                    prob_path = 1.0
                    reward.append(np.sum(reward_path) * prob_path)
        idid.dbn.net.clear_all_evidence()
        return reward
    def enter_evidences_gui_sim(self, idid, evidences, path):
        [idid.dbn_sim.net.clear_evidence(ei) for ei in evidences]
        for ei_index in range(0, len(evidences)):
            ei = evidences[ei_index]
            idid.dbn_sim.net.set_evidence(ei, int(path[ei_index]))
        idid.dbn_sim.net.update_beliefs()
    def enter_evidences_gui(self,idid,evidences,path):
        [idid.dbn.net.clear_evidence(ei) for ei in evidences]
        for ei_index in range(0, len(evidences)):
            ei = evidences[ei_index]
            idid.dbn.net.set_evidence(ei,int(path[ei_index]))
        idid.dbn.net.update_beliefs()
    def gen_pathes_weights(self,pathes,weights,mode):
        if mode == self.pnames.Sim_mode[3]:
            return pathes,weights
        if mode ==  self.pnames.Sim_mode[2]:
            choose = np.argmax(np.array(weights))
            pathes = [pathes[choose]]# compared one
            weights = [weights[choose]]# compared one
            return pathes, weights
        if mode ==  self.pnames.Sim_mode[1]:
            # choose = np.random.randint(low=0,high=len(weights), size=1)[0]
            choose = self.russian_roulette(weights)
            pathes = [pathes[choose]]# compared one
            weights = [weights[choose]]# compared one
            return pathes, weights
    def russian_roulette(self, weights):
        import random
        sum_w = np.sum(weights)
        sum_ = 0
        u = random.random() * sum_w
        for pos in range(0, len(weights)):
            sum_ += weights[pos]
            if sum_ > u:
                break
        return pos
    # play frame
    def create_play_frame(self,tab,id):
        master = self.create_frame(tab, id)
        frame_name = self.pnames.GUI_Frames[id]

        self.update_rowid()
        self.widgets['model_'+frame_name] = tk.StringVar()
        ttk.Label(master, text="Play Model:", width=20).grid(column=0, row=self.rowid)
        self.message_CompareModels['CB_'+frame_name] = ttk.Combobox(master, width=18, textvariable=self.widgets['model_'+frame_name])
        self.message_CompareModels['CB_'+frame_name] ['values'] = list()
        self.message_CompareModels['CB_'+frame_name] .grid(column=1, row=self.rowid, padx=1, pady=2)

        self.update_rowid()
        self.widgets['domain_'+frame_name] = tk.StringVar()
        ttk.Label(master, text="Play Domain:", width=20).grid(column=0, row=self.rowid)
        self.message_CompareDomains['CB_'+frame_name] = ttk.Combobox(master, width=18, textvariable=self.widgets['domain_'+frame_name] )
        self.message_CompareDomains['CB_'+frame_name] ['values'] = list()
        self.message_CompareDomains['CB_'+frame_name] .grid(column=1, row=self.rowid, padx=1, pady=2)

        self.update_rowid()
        ttk.Label(master, text=" Test Style:", width=20).grid(column=0, row=self.rowid)
        self.widgets['test_style_'+frame_name]  = tk.StringVar()
        test_style_Chosen = ttk.Combobox(master, width=18, textvariable=self.widgets['test_style_'+frame_name])
        test_style_Chosen['values'] = list(self.pnames.Test_style.values())
        test_style_Chosen.grid(column=1, row=self.rowid, padx=1,  pady=2)
        test_style_Chosen.current(1)

        self.update_rowid()
        # action = ttk.Button(master, text="Run", width=20, command=self.simulate)
        action = ttk.Button(master, text="Confirm", width=20, command=self.confirm_play_model)
        action.grid(column=0, row=self.rowid, padx=1,  pady=2)
        # action_test = ttk.Button(master, text="Close/Open", width=20, command=self.simulate1)
        action_test = ttk.Button(master, text="Close/Open", width=20, command=self.simulate1)
        action_test.grid(column=1, row=self.rowid, padx=1,  pady=2)

        self.update_rowid()
        # action_step = ttk.Button(master, text="Step", width=20, command=self.simulate1)
        action_step = ttk.Button(master, text="Step", width=20, command=self.step_play_game)
        action_step.grid(column=0, row= self.rowid, padx=1,  pady=2)
        # action_play = ttk.Button(master, text="Play", width=20, command=self.simulate1)
        action_play = ttk.Button(master, text="Play", width=20, command=lambda :self.play_game(frame_name))
        action_play.grid(column=1, row= self.rowid, padx=1,  pady=2)

        self.update_rowid()
        self.widgets['V3D_play'] = tk.IntVar()
        check3 = tk.Checkbutton(master, text="3D display", width=9, variable=  self.widgets['V3D_play'])
        check3.deselect()
        check3.grid(column=1, row= self.rowid, sticky=tk.W)
    def create_policytree_frame_i(self):
        self.policytree_i_win = tk.Toplevel()
        self.policytree_i = ttk.LabelFrame(self.policytree_i_win, text=' Agent i\'s policy trees', width=40, height=20)
        self.policytree_i.grid(column=0, row=  0, padx=1,  pady=2)
        self.policytree_i_plot = Game_Tree(self.policytree_i)
    def create_policytree_frame_j(self):
        self.policytree_j_win = tk.Toplevel()
        self.policytree_j = ttk.LabelFrame(self.policytree_j_win, text=' Agent j\'s policy trees', width=40, height=20)
        self.policytree_j.grid(column=0, row= 0, padx=1,  pady=2)
        self.policytree_j_plot = Game_Tree(self.policytree_j)
    def step_play_game(self):
        # simulate
        if self.Tiger.game_objects.get('door_left').status == 'close':
            self.Tiger.game_objects.get('door_left').open()
            # for key in self.policytree_i_plot.game_objects.keys():
            #     self.policytree_i_plot.game_objects.get(key).active()
            # for key in self.policytree_i_plot.node_ids.keys():
            # self.policytree_i_plot.game_objects.get(self.policytree_i_plot.node_ids.get(key)).active()
        else:
            self.Tiger.game_objects.get('door_left').close()

        if self.Tiger.game_objects.get('door_right').status == 'close':
            self.Tiger.game_objects.get('door_right').open()
            # for key in self.policytree_i_plot.game_objects.keys():
            #     self.policytree_i_plot.game_objects.get(key).active()
            # for key in self.policytree_i_plot.node_ids.keys():
            # self.policytree_i_plot.game_objects.get(self.policytree_i_plot.node_ids.get(key)).active()
        else:
            self.Tiger.game_objects.get('door_right').close()
            # for key in self.policytree_i_plot.game_objects.keys():
            #     self.policytree_i_plot.game_objects.get(key).disactive()

        self.Tiger.game_objects.get('tiger').make_swift()
        self.Tiger.game_objects.get('Agent i').make_swift()
        self.Tiger.game_objects.get('Agent j').make_swift()
    def play_game(self,frame_name):
        # self.create_policytree_frame_i()
        # ## agent j's policy tree
        # self.create_policytree_frame_j()
        if self.widgets['test_style_'+frame_name].get() == 'Offline test':
            # read model
            # read policy tree
            # simulate
            if self.Tiger.game_objects.get('door_left').status == 'close':
                self.Tiger.game_objects.get('door_left').open()
                # for key in self.policytree_i_plot.game_objects.keys():
                #     self.policytree_i_plot.game_objects.get(key).active()
                # for key in self.policytree_i_plot.node_ids.keys():
                # self.policytree_i_plot.game_objects.get(self.policytree_i_plot.node_ids.get(key)).active()
            else:
                self.Tiger.game_objects.get('door_left').close()

            if self.Tiger.game_objects.get('door_right').status == 'close':
                self.Tiger.game_objects.get('door_right').open()
                # for key in self.policytree_i_plot.game_objects.keys():
                #     self.policytree_i_plot.game_objects.get(key).active()
                # for key in self.policytree_i_plot.node_ids.keys():
                # self.policytree_i_plot.game_objects.get(self.policytree_i_plot.node_ids.get(key)).active()
            else:
                self.Tiger.game_objects.get('door_right').close()
                # for key in self.policytree_i_plot.game_objects.keys():
                #     self.policytree_i_plot.game_objects.get(key).disactive()

            self.Tiger.game_objects.get('tiger').make_swift()
            self.Tiger.game_objects.get('Agent i').make_swift()
            self.Tiger.game_objects.get('Agent j').make_swift()
    def confirm_play_model(self):
        # quit old windows
        if  not self.policytree_i_win is None:
            self.policytree_i_win.destroy()
        if not self.policytree_j_win is None:
            self.policytree_j_win.destroy()
        # read model

        # read policy tree
        # agent i's policy tree
        self.create_policytree_frame_i()
        ## agent j's policy tree
        self.create_policytree_frame_j()

        # policytree_i = self.sim.idids.get(self.play_model_name_value).policy_tree
        # self.create_policy_tree(self.policytree_i_plot,policytree_i)#
        # if self.widgets['test_style_play'].get() == 'Online test':
        #     # online test
        #     self.create_policy_tree(self.policytree_j_plot, self.sim.did.policy_tree)
        # else:
        #     # offline test
        #     policytree_j = self.sim.idids.get(self.play_model_name_value).did.policy_tree
        #     self.create_policy_tree(self.policytree_j_plot, policytree_j)  #
    def simulate1(self):
        if self.Tiger.game_objects.get('door_left').status == 'close':
            self.Tiger.game_objects.get('door_left').open()
            # for key in self.policytree_i_plot.game_objects.keys():
            #     self.policytree_i_plot.game_objects.get(key).active()
            # for key in self.policytree_i_plot.node_ids.keys():
            # self.policytree_i_plot.game_objects.get(self.policytree_i_plot.node_ids.get(key)).active()
        else:
            self.Tiger.game_objects.get('door_left').close()

        if self.Tiger.game_objects.get('door_right').status == 'close':
            self.Tiger.game_objects.get('door_right').open()
            # for key in self.policytree_i_plot.game_objects.keys():
            #     self.policytree_i_plot.game_objects.get(key).active()
            # for key in self.policytree_i_plot.node_ids.keys():
            # self.policytree_i_plot.game_objects.get(self.policytree_i_plot.node_ids.get(key)).active()
        else:
            self.Tiger.game_objects.get('door_right').close()
            # for key in self.policytree_i_plot.game_objects.keys():
            #     self.policytree_i_plot.game_objects.get(key).disactive()
        self.Tiger.game_objects.get('tiger').make_swift()
        self.Tiger.game_objects.get('Agent i').make_swift()
        self.Tiger.game_objects.get('Agent j').make_swift()
    def create_policy_tree(self,master,policy_tree):
        master.clear_game_object()
        node_list = policy_tree.get_nodelist()
        edgelist = policy_tree.get_edgelist()
        node_labels = policy_tree.get_node_labels()
        edge_labels = policy_tree.get_edge_labels()
        node_dict = policy_tree.get_node_dict()
        num_max = 1
        for level in node_dict.keys():
            if len(node_dict.get(level))>num_max:
                num_max = len(node_dict.get(level))
        c = 30
        w = 80
        height  = int(math.floor(c*self.horizon_max_value+2*c*(self.horizon_max_value-1))+2*c)
        width = int(math.floor(num_max*w)+w)

        master.set_width(width)
        master.set_height(height)
        node_pos = dict()
        for level in node_dict.keys():
            ns = node_dict.get(level)
            y = c*(self.horizon_max_value - level)+2*c*(self.horizon_max_value-level)+c
            w = int(math.floor(width/len(ns)))
            X = range(int(w/2),width,w)
            for ni in range(0,len(ns)):
                node_pos[ns[ni]] = [X[ni],y]
        for ei in edgelist:
            x_start = node_pos.get(ei[0])[0]
            y_start = node_pos.get(ei[0])[1]
            x_end = node_pos.get(ei[1])[0]
            y_end = node_pos.get(ei[1])[1]
            edge = Game_Edge(master.get_canvas(),
                             x_start=x_start,y_start= y_start,x_end=x_end,y_end=y_end,
                             label_str = edge_labels.get(ei),id= ei)
            master.add_game_object(edge)
            #edge.active()
        for ni in node_list:
            x_start = node_pos.get(ni)[0]
            y_start = node_pos.get(ni)[1]
            node = Game_Node(master.get_canvas(),
                             x=x_start,y= y_start,
                             label_str = node_labels.get(ni),id= ni)
            master.add_game_object(node)
            master.node_ids[ni] = node
    # message frame
    def create_message_frame(self,tab,id):
        scrolW = 120
        scrolH = 5
        master = self.create_frame(tab, id,clspan =3)
        frame_name = self.pnames.GUI_Frames[id]
        self.scr_message = scrolledtext.ScrolledText(master, width=scrolW, height=scrolH, wrap=tk.WORD)
        self.scr_message.grid(column=0, row=self.rowid , sticky='WE', columnspan=3)
    def print(self, string):
        self.scr_message.insert(tk.END,string + '\n')
        self.scr_message.see(tk.END)
        self.scr_message.update()
    def set_None(self):
        self.print('restarting the test now!')
        self.Simmodels = None
    def create_frame(self,tab,id,clspan = None):
        clspan = 1 if clspan is None else clspan
        self.update_rowid()
        frame_name = self.pnames.GUI_Frames[id]
        self.widgets['LF_' + frame_name] = ttk.LabelFrame(tab, text=frame_name)
        self.widgets['LF_' + frame_name].grid(column=0, row=self.rowid, padx=8, pady=4, columnspan=clspan)
        return self.widgets['LF_' + frame_name]
    # figure frame
    def create_figure_frame(self, tab, id):
        rows = int(np.floor((self.rowid) / 2))
        self.rowid = int(0)
        self.columnid = int(self.columnid + 1)
        master = self.create_frame(tab, id)
        frame_name = self.pnames.GUI_Frames[id]
        master.grid(column=self.columnid, row=self.rowid, padx=1, pady=2, rowspan=rows * 2, columnspan=1)
        self.Tiger_plot(master)
    def Tiger_plot(self,master):
        self.Tiger = Game(master)
        wall1 = Brick(self.Tiger.get_canvas(), x=00, y=200,width =100 ,height = 50)
        wall2 = Brick(self.Tiger.get_canvas(), x=200, y=200,width =100 ,height = 50)
        wall3 = Brick(self.Tiger.get_canvas(), x=400, y=200, width=100, height=50)

        wall4 = Brick(self.Tiger.get_canvas(), x=225, y=150, width=50, height=50)
        wall5 = Brick(self.Tiger.get_canvas(), x=225, y=00, width=50, height=50)
        door_left = Door(self.Tiger.get_canvas(), x=100, y=215, label_str ='door_left',id ='door_left')
        door_right = Door(self.Tiger.get_canvas(), x=300, y=215, label_str ='door_right',id='door_right')

        tiger  = ATiger(self.Tiger.get_canvas(), x=150, y=100, label_str ='tiger',  id='tiger',status='left')
        Agent_i = Agent(self.Tiger.get_canvas(), x=120, y=400, label_str ='Agent i',id='Agent i',color_id=1,status='left')
        Agent_j = Agent(self.Tiger.get_canvas(), x=380, y=400, label_str ='Agent j',id='Agent j',color_id=2,status='right')

        self.Tiger.add_game_object(door_left)
        self.Tiger.add_game_object(door_right)
        self.Tiger.add_game_object(tiger)
        self.Tiger.add_game_object(Agent_i)
        self.Tiger.add_game_object(Agent_j)
    #plot tab
    def Plot_tab(self,index):
        key = self.pnames.GUI_Tabs[index]
        sufix = self.pnames.GUI_Tabs_sc[index]
        tab = self.widgets[key + '_tab' + sufix]
        self.rowid = 0
        self.rowid_fix = 10
        self.create_plot_frame(tab,key,6)
        self.create_plot_alg_frame(tab,key,7)
        self.create_table_frame(tab,key,8)
    # plot frame
    def create_plot_frame(self,tab,key_tab,id):
        master = self.create_frame(tab, id)
        frame_name = self.pnames.GUI_Frames[id]
        # master = ttk.LabelFrame(tab, text=frame_name)
        # master.grid(column=0, row= self.rowid , padx=8, pady=4, columnspan=3)
        self.widgets['figures_' + frame_name] = dict()
        self.update_rowid()
        i = 0
        self.widgets['domain_' + frame_name] = tk.StringVar()
        ttk.Label(master, text=" Domain:", width=10).grid(column=i, row = self.rowid)
        self.message_CompareDomains['CB_'+ frame_name] = ttk.Combobox(master, width=20,
                                                                    textvariable=self.widgets['domain_' + frame_name])
        self.message_CompareDomains['CB_'+ frame_name]['values'] = list()
        self.message_CompareDomains['CB_'+ frame_name].grid(column=i+1, row = self.rowid, padx=1, pady=2)

        self.update_rowid()
        ttk.Label(master, text=" Test:", width=10).grid(column=i, row = self.rowid)
        self.widgets['test_style_'+frame_name] = tk.StringVar()
        test_style_Chosen = ttk.Combobox(master, width=20, textvariable=self.widgets['test_style_'+frame_name])
        test_style_Chosen['values'] = list(self.pnames.Test_style.values())
        test_style_Chosen.grid(column=i+1, row = self.rowid, padx=1, pady=2)
        test_style_Chosen.current(1)

        self.SimMod[frame_name] = nsp.SimMod()
        i=2
        for key in self.SimMod[frame_name].parameters_type.keys():
            ttk.Label(master, text=key, width=20).grid(column=i, row=self.rowid)
            mod_Chosen = ttk.Combobox(master, width=20, textvariable=self.SimMod[frame_name].parameters_type.get(key))
            mod_Chosen['values'] = list(self.pnames.Sim_mode.values())
            mod_Chosen.grid(column=i+1, row=self.rowid, padx=1, pady=2)
            mod_Chosen.current(1)
            self.rowid = self.rowid - 1
        i+=2
        self.update_rowid()
        self.widgets['Analysis_all_'+frame_name] = tk.IntVar()
        check3 = tk.Checkbutton(master, text="Plot_all", width=9, variable=self.widgets['Analysis_all_'+frame_name])
        check3.deselect()
        check3.grid(column=i, row=self.rowid, sticky=tk.W)
        self.widgets['C_' + frame_name] = tk.IntVar()
        check4 = tk.Checkbutton(master, text="Popup", width=9, variable=self.widgets['C_' + frame_name])
        check4.deselect()
        check4.grid(column=i+1, row=self.rowid, sticky=tk.W)

        self.update_rowid()
        test = ttk.Button(master, text="Plot", command=lambda: self.plot_data(tab,frame_name,key_tab))
        test.grid(column=i, row=self.rowid, padx=1, pady=2, sticky='WE')
        coverage = ttk.Button(master, text="Coverage", width=10, command=lambda :self.coverage(tab,frame_name,key_tab))
        coverage.grid(column=i+1, row=self.rowid, padx=1, pady=2, columnspan=1)
        close = ttk.Button(master, text="Close all", width=10,command=lambda :self.close_all_figures(frame_name,key_tab))
        close.grid(column=i+2, row=self.rowid, padx=1, pady=2, sticky='WE')

        self.update_rowid()
        self.widgets['rowid_' + key_tab] = self.rowid_fix
    def close_all_figures(self,frame_name,key_tab):
        for ei in  self.widgets['figures_' + frame_name].keys():
            self.widgets['figures_' + frame_name][ei].destroy()
            self.widgets['rowid_' + key_tab] = self.widgets['rowid_' + key_tab] - 1
        self.widgets['figures_' + frame_name] = dict()
        if self.widgets['rowid_' + key_tab] <self.rowid_fix:
            self.widgets['rowid_' + key_tab] = self.rowid_fix
    def plot_data(self,tab,frame_name,key_tab):
        if self.popup_messeage(frame_name):
            return -1
        if self.widgets['Analysis_all_' + frame_name].get():
            test_styles = self.pnames.Test_style.values()
        else:
            test_styles = [self.widgets['test_style_' + frame_name].get()]
        self.SimMod[frame_name].update()
        sim_mode_i_set, sim_mode_j_set = self.get_sim_mode(frame_name)
        for test_style in test_styles:
            self.SimMod[frame_name].add(test_style, sim_mode_i_set, sim_mode_j_set)
        domain = self.widgets['domain_'+frame_name].get()
        for test_style in test_styles:
            self.on_offline_test(test_style, sim_mode_i_set, sim_mode_j_set)
            self.write_csv_results(test_style, sim_mode_i_set, sim_mode_j_set)
            wins = self.plot_data_sub(sim_mode_i_set,sim_mode_j_set,domain,test_style,tab,frame_name,key_tab)
            for ei in wins:
                self.widgets['figures_' + frame_name][len(self.widgets['figures_' + frame_name])] = ei
    def plot_data_sub(self,sim_mode_i_set,sim_mode_j_set,domain,test_style,tab,frame_name,key_tab):
        wins = list()
        for si in sim_mode_i_set:
            for sj in sim_mode_j_set:
                key = si + ' vs ' + sj
                key_str = self.pnames.Sim_mode_abrv.get(si)+ ' vs ' +self.pnames.Sim_mode_abrv.get(sj)
                num_mod_did_list = list()
                keys = list(self.CompareModels.keys())
                for keym in self.CompareModels.keys():
                    if test_style == self.pnames.Test_style.get(1):
                        idid = self.Simmodels.IDID[domain][keym]
                    if test_style == self.pnames.Test_style.get(2):
                        idid = self.Simmodels.DID[domain]
                    num_mod_did_list.append(len(idid.did.dbn.result.get('policy_dict').keys()))#
                num_mod_did_max = np.max(num_mod_did_list)
                index = np.argmax(num_mod_did_list)
                indexs = [keys[index]]
                for i in range(0, len(keys)):
                    if i != index:
                        indexs.append(keys[i])
                bar_width = 1 / (len(indexs) + 1)
                modi_keys = idid.dbn.result.get('policy_dict').keys()
                win = self.popup_window_plot_fig(key_str,modi_keys, indexs, domain, key, test_style, bar_width, num_mod_did_max,tab,frame_name,key_tab)
                wins.append(win)
        return wins
    def popup_window_plot_fig(self,key_str,modi_keys,indexs,domain,key,test_style,bar_width,num_mod_did_max,tab,frame_name,key_tab):
        if  self.widgets['C_' + frame_name].get():
            win = tk.Toplevel()
            win.iconbitmap("./logo.ico")
            win.wm_attributes('-topmost', 1)
            win.title('plot data @ domain: ' + domain)
        else:
            self.widgets['rowid_' + key_tab] = self.widgets['rowid_' + key_tab] + 1
            rowid = self.widgets['rowid_' + key_tab]
            win = ttk.LabelFrame(tab, text=frame_name)
            win.grid(column=0, row= rowid , padx=8, pady=4, columnspan=3)
        # canvas = tk.Canvas(win)
        # canvas.grid(row=0, column=0)
        # fig = plt.figure(dpi=100, figsize=(8, 6))
        # canvas1 = FigureCanvasTkAgg(fig, canvas)
        # canvas1.get_tk_widget().pack()
        # canvas1._tkcanvas.pack()
        # # canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        fig = plt.figure(dpi=100,figsize=(8,6))
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        cwidg = canvas.get_tk_widget()
        # Get the width and height of the *figure* in pixels
        w = fig.get_figwidth() * fig.get_dpi()
        h = fig.get_figheight() * fig.get_dpi()
        # Generate a blank tkinter Event object
        evnt = tk.Event()
        # Set the "width" and "height" values of the event
        evnt.width = w * 1.2
        evnt.height = h * 1.2
        cwidg.configure(width=w, height=h)
        canvas.resize(evnt)
        for modi in modi_keys:
            axis = plt.subplot(len(modi_keys), 1, modi+1)
            count = 0
            for keym in indexs:
                idid = self.Simmodels.IDID[domain][keym]
                result = idid.result.rewards.get(test_style)
                # print(key)
                rewards_mean = result.get(key).get('mean')
                rewards_std = result.get(key).get('std')
                values = rewards_mean[modi, :]
                SD = rewards_std[modi, :]
                num_mod_did = len(values)
                index = np.arange(num_mod_did)
                color_set = mcolors.CSS4_COLORS
                ckeys = list(color_set.keys())
                color = []
                for i in range(0,len(values)):
                    color.append(color_set[ckeys[0 + 3 * count + 10]])
                    # if values[i] < 0:
                    #     color.append(color_set[ckeys[0+3*count+10]])
                    # if values[i] == 0:
                    #     color.append(color_set[ckeys[1+3*count+10]])
                    # if values[i] > 0:
                    #     color.append(color_set[ckeys[2+3*count+10]])
                color = tuple( color)
                plt.bar(index + count*bar_width, values, width= bar_width,yerr=SD,
                        color =color,error_kw={'ecolor': '0.2', 'capsize': 6},
                        alpha=0.7, label='First')
                count =  count +1
            title = test_style + '- Reward @ '+ domain + ' @ ' +'i\'s mod ' + str(modi) + '-'+ key_str
            plt.title(title)
            if  modi == len(modi_keys)-1 :
                ticks = list()
                index = np.arange(num_mod_did_max)
                for modj in range(1, num_mod_did_max  + 1):
                    ticks.append('m' + str(modj))
                plt.xticks(index + 0.5, ticks)
                plt.setp(axis.get_xticklabels(), rotation=30, horizontalalignment='right')
                axis.set_xlabel('Agent j\'s policy trees')
            else:
                plt.setp(axis.get_xticklabels(), visible=False)
                plt.tick_params(
                    axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)  # labels along the bottom edge are off
            plt.legend(indexs, loc='best')
            plt.subplots_adjust(top=10, bottom=9.9)
            axis.set_ylabel('Reward')
            # axis.set_xlabel('Agent j\'s policy trees')
            # plt.show()
        canvas.draw()
        fig.tight_layout()
        fig.savefig(self.pnames.Save_filepath+ title + '.pdf')
        self.win_keep()
        self.winm.resizable(True, True)
        return win
        # win.quit()
    # coverage
    def coverage(self,tab=None,frame_name=None, key_tab = None):
        if self.popup_messeage('coverage'):
            return -1
        # rebuild a public path dictionary for opperation
        # calculate the similarity of different policy tree pathes
        # compute the join of policy tree, i.e join/union
        # return the coverage of policy tree space
        test_style = 'coverage test'
        self.print('-------------- analysis now --------------------------')
        self.print('>>> test @:' + test_style)
        for keyd in self.CompareDomains.keys():
            self.print('>>> run test @ domain: ' + keyd)
            models_coverage= dict()
            for keym_A in self.CompareModels.keys():
                self.print('>>> run test @ model A: ' + keym_A)
                for keym_B in self.CompareModels.keys():
                    if keym_A == keym_B:
                        continue
                    self.print('>>> run test @ model B: ' + keym_B)
                    strm = keym_A+keym_B
                    if models_coverage.__contains__(strm):
                        continue
                    strm1 = keym_B+keym_A
                    if models_coverage.__contains__(strm1):
                        continue
                    idid_A = self.Simmodels.IDID[keyd][keym_A]
                    idid_B = self.Simmodels.IDID[keyd][keym_B]
                    policy_dict_A = idid_A.did.dbn.result.get('policy_dict')
                    policy_dict_B = idid_B.did.dbn.result.get('policy_dict')
                    R_A = idid_A.did.dbn.result.get('reward')
                    R_B = idid_B.did.dbn.result.get('reward')
                    mat_cof = np.ones((len(policy_dict_A.keys()), len(policy_dict_B.keys())))
                    mat_reward = np.zeros((len(policy_dict_A.keys()), len(policy_dict_B.keys())))
                    for modi_a in  policy_dict_A.keys():
                        reward_a =R_A.get(modi_a)
                        pathes_a = policy_dict_A.get(modi_a)
                        for modi_b in policy_dict_B.keys():
                            reward_b =R_B.get(modi_b)
                            pathes_b = policy_dict_B.get(modi_b)
                            num = len(pathes_a) * len(pathes_a[0])
                            if pathes_a[0][0]!= pathes_b[0][0]:
                                cof = 0
                            else:
                                length = len(pathes_a[0])
                                cof = 0
                                for i in range(0,len(pathes_a)):
                                    for j in range(0, length, 2):
                                        pa_b = pathes_a[i][j]-pathes_b[i][j]
                                        if pa_b == 0 :
                                            cof = cof +1
                                        else:
                                            continue
                            mat_cof[modi_a,modi_b] = cof/num
                            print(reward_a, reward_b)
                            mat_reward[modi_a, modi_b] = reward_a - reward_b
                    models_coverage[strm] = mat_cof
                    print(mat_cof)
                    print(mat_reward)
                    self.plot_cof_mat(mat_cof,keym_A,keym_B,keyd,tab,frame_name,key_tab)
                self.Models_Coverage[keyd] = models_coverage
        self.print('-------------- analysis done --------------------------')
    def plot_cof_matbk(self,mat_cof,keym_A,keym_B,keyd,tab,frame_name):
        if tab is None and frame_name is None:
            win = tk.Toplevel()
            win.iconbitmap("./logo.ico")
            win.wm_attributes('-topmost', 1)
            win.title('Coverage @ domain: ' + keyd)
        else:
            if self.widgets['C_' + frame_name].get():
                win = tk.Toplevel()
                win.iconbitmap("./logo.ico")
                win.wm_attributes('-topmost', 1)
                win.title('Coverage @ domain: ' + keyd)
            else:
                win = ttk.LabelFrame(tab, text=frame_name)
                self.update_rowid()
                win.grid(column=0, row=self.rowid, padx=8, pady=4, columnspan=3)
        fig = plt.figure(dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        cwidg = canvas.get_tk_widget()
        # Get the width and height of the *figure* in pixels
        w = fig.get_figwidth() * fig.get_dpi()
        h = fig.get_figheight() * fig.get_dpi()
        # Generate a blank tkinter Event object
        evnt = tk.Event()
        # Set the "width" and "height" values of the event
        evnt.width = w * 1.2
        evnt.height = h * 1.2

        # Set the width and height of the canvas widget
        cwidg.configure(width=w, height=h)
        # Pass the generated event object to the FigureCanvasTk.resize() function
        canvas.resize(evnt)
        # fig, ax = plt.subplots()
        ax = plt.gca()
        xticks = list()
        for modj in range(0,mat_cof.shape[0]):
            xticks.append('m' + str(modj))
        yticks = list()
        for modi in range(0, mat_cof.shape[1]):
            yticks.append('m' + str(modi))
        ax.set_ylabel('ALG:' +keym_A)
        ax.set_xlabel('ALG:' +keym_B)
        title = 'Coverage @'+ keym_A+' Vs '+keym_B +'\n @ ' + keyd
        ax.set_title(title)
        im, cbar = heatmap(mat_cof, yticks, xticks, ax=ax,
                           cmap="magma_r", cbarlabel="coverage")  # "YlGn""Wistia""magma_r""PuOr"
        texts = annotate_heatmap(im, valfmt="{x:.1f}")
        # plt.show()
        title = 'Coverage @' + keym_A + ' Vs ' + keym_B + ' @ ' + keyd
        canvas.draw()
        fig.tight_layout()
        fig.savefig(self.pnames.Save_filepath + title + '_heatmap.pdf')
    def plot_cof_mat(self,mat_cof,keym_A,keym_B,keyd,tab,frame_name, key_tab):
        if tab is None and frame_name is None and key_tab is None:
            win = tk.Toplevel()
            win.iconbitmap("./logo.ico")
            win.wm_attributes('-topmost', 1)
            win.title('Coverage @ domain: ' + keyd)
        else:
            if self.widgets['C_' + frame_name].get():
                win = tk.Toplevel()
                win.iconbitmap("./logo.ico")
                win.wm_attributes('-topmost', 1)
                win.title('Coverage @ domain: ' + keyd)
            else:
                win = ttk.LabelFrame(tab, text=frame_name)
                self.widgets['rowid_' + key_tab] = self.widgets['rowid_' + key_tab] + 1
                rowid = self.widgets['rowid_' + key_tab]
                win.grid(column=0, row=rowid, padx=8, pady=4, columnspan=3)
        # canvas = tk.Canvas(win)
        # canvas.grid(row=0, column=0)
        fig = plt.figure(dpi=100,figsize=(8,6),)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        cwidg = canvas.get_tk_widget()
        # Get the width and height of the *figure* in pixels
        w = fig.get_figwidth() * fig.get_dpi()
        h = fig.get_figheight() * fig.get_dpi()
        # Generate a blank tkinter Event object
        evnt = tk.Event()
        # Set the "width" and "height" values of the event
        evnt.width = w * 1.2
        evnt.height = h * 1.2

        # Set the width and height of the canvas widget
        cwidg.configure(width=w, height=h)
        # Pass the generated event object to the FigureCanvasTk.resize() function
        canvas.resize(evnt)
        # fig, ax = plt.subplots()
        ax = plt.gca()
        xticks = list()
        for modj in range(0,mat_cof.shape[1]):
            xticks.append('m' + str(modj))
        yticks = list()
        for modi in range(0, mat_cof.shape[0]):
            yticks.append('m' + str(modi))
        ax.set_ylabel('ALG:' +keym_A)
        ax.set_xlabel('ALG:' +keym_B)
        title = 'Coverage @'+ keym_A+' Vs '+keym_B +'\n @ ' + keyd
        ax.set_title(title)
        im, cbar = heatmap(mat_cof, yticks, xticks, ax=ax,
                           cmap="magma_r", cbarlabel="coverage")  # "YlGn""Wistia""magma_r""PuOr"
        texts = annotate_heatmap(im, valfmt="{x:.1f}")
        # plt.show()
        title = 'Coverage @' + keym_A + ' Vs ' + keym_B + ' @ ' + keyd
        canvas.draw()
        fig.tight_layout()
        fig.savefig(self.pnames.Save_filepath + title + '_heatmap.pdf')
        if not(tab is None and frame_name is None and key_tab is None):
            self.widgets['figures_' + frame_name][len(self.widgets['figures_' + frame_name])]= win
        self.winm.resizable(True, True)
    # Alg
    def create_plot_alg_frame(self, tab, key_tab, id):
        master = self.create_frame(tab, id)
        frame_name = self.pnames.GUI_Frames[id]
        # master = ttk.LabelFrame(tab, text=frame_name)
        # master.grid(column=0, row= self.rowid , padx=8, pady=4, columnspan=3)
        self.widgets['figures_' + frame_name] = dict()
        self.update_rowid()
        i = 0
        self.widgets['Algs_' + frame_name] = tk.StringVar()
        ttk.Label(master, text='Alg data', width=9).grid(column=i, row=self.rowid)
        self.widgets['CB_' + frame_name] = ttk.Combobox(master, width=44,
                                                        textvariable=self.widgets['Algs_' + frame_name])
        self.widgets['CB_' + frame_name]['values'] = list()
        self.widgets['CB_' + frame_name].grid(column=i+1 , row=self.rowid, padx=1, pady=2)
        self.widgets['Plot_all_' + frame_name] = tk.IntVar()
        check3 = tk.Checkbutton(master, text="Plot_all", width=9, variable=self.widgets['Plot_all_' + frame_name])
        check3.deselect()
        check3.grid(column=i+2, row=self.rowid, sticky=tk.W)
        self.widgets['C_' + frame_name] = tk.IntVar()
        check4 = tk.Checkbutton(master, text="Popup", width=9, variable=self.widgets['C_' + frame_name])
        check4.deselect()
        check4.grid(column=i + 3, row=self.rowid, sticky=tk.W)
        coverage = ttk.Button(master, text="Update", width=9,
                              command=lambda: self.update_alg(tab, frame_name, key_tab))
        coverage.grid(column=i +4, row=self.rowid, padx=1, pady=2, columnspan=1)

        test = ttk.Button(master, text="Plot", command=lambda: self.plot_data_alg(tab, frame_name, key_tab))
        test.grid(column=i+5, row=self.rowid, padx=1, pady=2, sticky='WE')

        close = ttk.Button(master, text="Close all", width=9,
                           command=lambda: self.close_all_figures(frame_name, key_tab))
        close.grid(column=i +6, row=self.rowid, padx=1, pady=2, sticky='WE')
        self.update_rowid()
        self.widgets['rowid_' + key_tab] = self.rowid_fix
    def update_alg(self,tab, frame_name, key_tab):
        if self.popup_messeage(frame_name):
            return -1
        self.widgets['CB_' + frame_name]['values'] = list()
        self.widgets['Plots_' + frame_name] = dict()
        items = list()
        for keyd in self.CompareDomains.keys():
            for keym in self.CompareModels.keys():
                head = list()
                head.append(keyd)
                head.append(keym)
                idid_rs = self.Simmodels.IDID[keyd][keym].dbn.result['Plot']
                did_rs = self.Simmodels.IDID[keyd][keym].did.dbn.result['Plot']
                self.update_alg_sub(idid_rs, head, items, frame_name,'idid')
                self.update_alg_sub(did_rs, head, items, frame_name,'did')
        if len(items)>0:
            self.widgets['CB_' + frame_name]['values'] = items
            self.widgets['CB_' + frame_name].update()
    def update_alg_sub(self,idid_rs,head,items,frame_name,type):
        if len(idid_rs)!=0:
            lines = [ [key_st, key_ft] for key_st in idid_rs.keys() for key_ft in idid_rs.get(key_st).keys()]
            for line in lines:
                item = list()
                [item.append(ei) for ei in head]
                item.append(type)
                for ei in line:
                    item.append(ei)
                item_str = '|'.join(item)
                items.append(item_str)
                self.widgets['Plots_' + frame_name][item_str] = item
    def plot_data_alg(self,tab, frame_name, key_tab):
        if self.popup_messeage(frame_name):
            return -1
        # print( len(self.widgets['CB_' + frame_name]['values']))
        if len(self.widgets['CB_' + frame_name]['values'])>0:
            if self.widgets['Plot_all_'+ frame_name].get():
                alg_figures = self.widgets['Plots_' + frame_name].keys()
            else:
                alg_figures = [self.widgets['Algs_' + frame_name].get()]
            # print(alg_figures)
            for alg_figure in alg_figures:
                item = self.widgets['Plots_' + frame_name].get(alg_figure)
                keyd = item[0]
                keym = item[1]
                if item[2] == 'idid':
                    fig_data = self.Simmodels.IDID[keyd][keym].dbn.result['Plot'][item[3]][item[4]]
                else:
                    fig_data = self.Simmodels.IDID[keyd][keym].did.dbn.result['Plot'][item[3]][item[4]]
                self.plot_alg(fig_data,tab,frame_name, key_tab,alg_figure)
    def plot_alg(self,fig_data,tab,frame_name, key_tab,alg_figure):
        if tab is None and frame_name is None and key_tab is None:
            win = tk.Toplevel()
            win.iconbitmap("./logo.ico")
            win.wm_attributes('-topmost', 1)
            win.title('Plot @ Alg')
        else:
            if self.widgets['C_' + frame_name].get():
                win = tk.Toplevel()
                win.iconbitmap("./logo.ico")
                win.wm_attributes('-topmost', 1)
                win.title('Plot @ Alg')
            else:
                win = ttk.LabelFrame(tab, text=alg_figure)
                self.widgets['rowid_' + key_tab] = self.widgets['rowid_' + key_tab] + 1
                rowid = self.widgets['rowid_' + key_tab]
                win.grid(column=0, row=rowid, padx=8, pady=4, columnspan=3)
        fig = plt.figure(dpi=100,figsize=(8,6),)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        cwidg = canvas.get_tk_widget()
        w = fig.get_figwidth() * fig.get_dpi()
        h = fig.get_figheight() * fig.get_dpi()
        evnt = tk.Event()
        evnt.width = w * 1.2
        evnt.height = h * 1.2
        cwidg.configure(width=w, height=h)
        canvas.resize(evnt)
        ax = plt.gca()
        xValues=fig_data['xValues']
        data = fig_data['yValues']
        xlim = fig_data['xlim']
        (minvalue, maxvalue) = fig_data['ylim']
        if maxvalue == 'NaN' or maxvalue == 'Inf' or maxvalue is None:
            minvalue = np.min([np.min(data.get(key)) for key in data.keys()])
            maxvalue = np.max([np.max(data.get(key)) for key in data.keys()])
        if  maxvalue != 'NaN' and  maxvalue != 'Inf' and not (maxvalue is None):
            ax.set_ylim(minvalue, maxvalue)
        ylabel = fig_data['ylabel']
        xlabel = fig_data['xlabel']
        title  = fig_data['title']
        t_legend = fig_data['legend']
        filename = fig_data['filename']
        import matplotlib.colors as mcolors
        ax.set_xlim(xlim[0], xlim[1])
        t = list()
        count = 0
        color_set = mcolors.CSS4_COLORS
        ckeys = list(color_set.keys())
        for key in data.keys():
            t_i, = plt.plot(xValues, data.get(key), color=color_set[ckeys[0 + 3 * count + 10]])
            t.append(t_i)
            count = count + 1
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.grid()
        plt.legend(t, t_legend, loc='best')
        plt.title(title)
        # plt.show()
        canvas.draw()
        fig.tight_layout()
        fig.savefig(filename)
        if not(tab is None and frame_name is None and key_tab is None):
            self.widgets['figures_' + frame_name][len(self.widgets['figures_' + frame_name])]=win
        self.winm.resizable(True, True)
    # table
    def create_table_frame(self,tab,key_tab,id):
        master = self.create_frame(tab, id)
        frame_name = self.pnames.GUI_Frames[id]
        self.widgets['tablewins_' + frame_name] = dict()
        self.update_rowid()
        i = 0
        ttk.Label(master, text=" Table:", width=10).grid(column=i, row = self.rowid)
        self.widgets['table_'+frame_name] = tk.StringVar()
        test_style_Chosen = ttk.Combobox(master, width=53, textvariable=self.widgets['table_'+frame_name])
        test_style_Chosen['values'] = self.pnames.table_files
        test_style_Chosen.grid(column=i+1, row = self.rowid, padx=1, pady=2)
        test_style_Chosen.current(1)

        self.widgets['open_all_'+frame_name] = tk.IntVar()
        check3 = tk.Checkbutton(master, text="open_all", width=9, variable=self.widgets['open_all_'+frame_name])
        check3.deselect()
        check3.grid(column=i+2, row=self.rowid, sticky=tk.W)

        self.widgets['C_' + frame_name] = tk.IntVar()
        check4 = tk.Checkbutton(master, text="Popup", width=9, variable=self.widgets['C_' + frame_name])
        check4.deselect()
        check4.grid(column=i+3, row=self.rowid, sticky=tk.W)

        table = ttk.Button(master, text="Table", width=10, command=lambda: self.popup_window_Table(tab,frame_name,key_tab))
        table.grid(column=i +4, row=self.rowid, padx=1, pady=2, sticky='WE')

        close = ttk.Button(master, text="Close all", width=10,command=lambda :self.close_all_tables(frame_name,key_tab))
        close.grid(column=i+5, row=self.rowid, padx=1, pady=2, sticky='WE')
        self.update_rowid()
        self.widgets['rowid_' + key_tab] = self.rowid_fix
    def close_all_tables(self,frame_name,key_tab):
        for ei in  self.widgets['tablewins_' + frame_name].keys():
            self.widgets['tablewins_'+ frame_name][ei].destroy()
            self.widgets['rowid_' + key_tab] = self.widgets['rowid_' + key_tab] -1
        self.widgets['tablewins_' + frame_name] = dict()
        if self.widgets['rowid_' + key_tab] < self.rowid_fix:
            self.widgets['rowid_' + key_tab] = self.rowid_fix
    def popup_window_Table(self,tab,frame_name,key_tab):
        if self.popup_messeage(frame_name):
            return -1
        if self.widgets['open_all_' + frame_name].get():
            table_files =self.pnames.table_files
        else:
            table_files = [self.widgets['table_' + frame_name].get()]
        for table in table_files:
            if self.widgets['C_' + frame_name].get():
                win = tk.Toplevel()
                win.iconbitmap("./logo.ico")
                win.wm_attributes('-topmost', 1)
                win.title(table)
                rowid = 0
            else:
                self.widgets['rowid_' + key_tab] = self.widgets['rowid_' + key_tab]+1
                rowid = self.widgets['rowid_' + key_tab]
                win = ttk.LabelFrame(tab, text=frame_name + '@' + table)
                win.grid(column=0, row = rowid, padx=8, pady=4, columnspan=3)
            # Set the size of the tkinter window
            # excel_dir = self.pnames.Save_filepath+table+'.xlsx'
            # api_info_list,dataset = self.read_data_from_excel(excel_dir)
            if not self.dataset.__contains__(table):
                return -1
            dataset = self.dataset.get(table)
            # Add a Treeview widget
            cls = tuple(["c"+str(i) for i in range(1,dataset['columns']+1)])
            tree = ttk.Treeview(win, column=cls, show='headings', height=5)
            tree.grid(column=0, row=rowid+1, padx=8, pady=4, columnspan=3)
            #heading
            for i in range(1, dataset['columns']+1):
                text =  dataset['data'][0][i-1]
                tree.column("# "+str(i), anchor=tk.CENTER)
                tree.heading("# "+str(i), text=text)
            # Insert the data in Treeview widget
            for r in range(1, dataset['rows']):  # 从第2行开始取数据
                text = tuple(dataset['data'][r])
                tree.insert('', 'end', text="1", values=text)
            tree.pack()
            self.widgets['tablewins_' + frame_name][len(self.widgets['tablewins_' + frame_name])]=win
    # diversity tab
    def Diversity_tab(self,index):
        key = self.pnames.GUI_Tabs[index]
        sufix = self.pnames.GUI_Tabs_sc[index]
        tab = self.widgets[key + '_tab' + sufix]
        self.rowid = 0
        self.create_diversity_frame(tab,'',key)
    # diversity_scroll tab
    def Diversity_sc_tab(self,index):
        key = self.pnames.GUI_Tabs[index]
        sufix = self.pnames.GUI_Tabs_sc[index]
        tab = self.widgets[key + '_tab' + sufix]
        self.rowid = 0
        self.create_diversity_frame(tab,self.pnames.sufix_sc,key)
    # diversity frame
    def create_diversity_frame(self,tab,sufix,key):
        self.rowid = 0

        self.widgets['width'] = 120
        self.widgets['height'] = 150
        frame_name = key
        self.widgets['master_ps'+sufix] = ttk.LabelFrame(tab, text= frame_name, height=10, width=self.widgets['width'])
        self.widgets['master_ps'+sufix].grid(column=0, row=self.rowid, padx=4, pady=4, columnspan=2)
        self.update_rowid()
        # domain select
        self.widgets['domain_div'+sufix] = tk.StringVar()
        ttk.Label(self.widgets['master_ps'+sufix], text=" Domain:", width=10).grid(column=0, row = self.rowid)
        self.message_CompareDomains['CB_'+frame_name+sufix] = ttk.Combobox(self.widgets['master_ps'+sufix], width=20,
                                                                     textvariable=self.widgets['domain_div'+sufix])
        self.message_CompareDomains['CB_'+frame_name+sufix]['values'] = list()
        self.message_CompareDomains['CB_'+frame_name+sufix].grid(column=1, row = self.rowid, padx=4, pady=2)
        #model select
        self.widgets['model_div'+sufix] = tk.StringVar()
        ttk.Label(self.widgets['master_ps'+sufix], text=" Model:", width=10).grid(column=2, row = self.rowid)
        self.message_CompareModels['CB_'+frame_name+sufix] = ttk.Combobox(self.widgets['master_ps'+sufix], width=20,
                                                                   textvariable=self.widgets['model_div'+sufix])
        self.message_CompareModels['CB_'+frame_name+sufix]['values'] = list()
        self.message_CompareModels['CB_'+frame_name+sufix].grid(column=3, row = self.rowid, padx=4, pady=2)

        self.widgets['idid_div' + sufix] = tk.StringVar()
        ttk.Label(self.widgets['master_ps' + sufix], text=" IDID/DID:", width=10).grid(column=4, row=self.rowid)
        self.widgets['CB_' + frame_name + sufix] = ttk.Combobox(self.widgets['master_ps' + sufix],width=20,
                                                                              textvariable=self.widgets['idid_div' + sufix])
        self.widgets['CB_' + frame_name + sufix]['values'] = self.pnames.simmod_pt
        self.widgets['CB_' + frame_name + sufix].grid(column=5, row=self.rowid, padx=4, pady=2)

        Confirm = ttk.Button(self.widgets['master_ps' + sufix], text="Confirm", width=10,
                             command=lambda: self.diversity_confirm(sufix,frame_name))
        Confirm.grid(column=6, row=self.rowid, padx=4, pady=2)
        self.update_rowid()
    def diversity_confirm(self,sufix,frame_name):
        if self.popup_messeage(frame_name):
            return -1
        self.destroy_frame(sufix)
        model = self.widgets['model_div'+sufix].get()
        domain = self.widgets['domain_div'+sufix].get()
        idid_did = self.widgets['idid_div' + sufix].get()
        if idid_did == self.pnames.simmod_pt[1]:
            did = self.Simmodels.IDID[domain][model].did
        else:
            did = self.Simmodels.IDID[domain][model]
        self.widgets[frame_name+sufix] = Diversity(did.dbn.result.get('policy_dict'))

        self.create_frame_div(frame_name,'Policy Trees','pt',sufix,scroll = False if sufix =='' else True)
        pt_dict = did.dbn.result.get('policy_dict')
        pt_keys = pt_dict.keys()
        height = self.widgets['height']
        width = int(np.ceil(self.widgets['width']))
        action_list = did.dbn.action_list
        observation_list = did.dbn.observation_list
        pt_horizon_dict = dict([(i,did.dbn.horizon)for i in pt_keys])
        textlist = dict([(i, 'H_%s' % i) for i in pt_keys])
        master = self.widgets['master_inner_frame_'+'pt'+sufix] if not sufix =='' else self.widgets['master_div_' + 'pt' + sufix]
        self.create_subframe('pt',sufix, width, height, pt_dict, pt_horizon_dict, action_list, observation_list,textlist,master)

        self.update_rowid()
        self.create_frame_div(frame_name,'Set of unique Triangles','st',sufix,scroll = False if sufix =='' else True)
        pt_dict = self.widgets[frame_name+sufix].triangles
        height = self.widgets['height']
        width = int(np.ceil(self.widgets['width']))
        width, height = self.normilize_WH(self.widgets[frame_name+sufix], width, height)
        pt_horizon_dict = self.widgets[frame_name+sufix].triangles_horizon
        textlist = self.widgets[frame_name+sufix].Columns
        master = self.widgets['master_inner_frame_' + 'st' + sufix]if not sufix =='' else self.widgets['master_div_' + 'st' + sufix]
        self.create_subframe('st',sufix, width, height, pt_dict, pt_horizon_dict, action_list, observation_list,textlist,master)

        self.create_message_div_frame(self.widgets[frame_name+sufix],sufix,pt_keys,frame_name)
        action = ttk.Button(self.widgets['master_div_psm'+sufix], text="Calculate", width=20, command=lambda:self.calculate_diversity(frame_name,sufix,self.widgets[frame_name+sufix]))
        action.grid(column=2, row=self.rowid, padx=1, pady=2)
        self.update_rowid()
    def destroy_frame(self,sufix):
        if self.widgets.__contains__('master_div_psm'+ sufix):
            self.widgets['master_div_psm' + sufix].destroy()
        if self.widgets.__contains__('master_div_ptm'+ sufix):
            self.widgets['master_div_ptm' + sufix].destroy()
        if self.widgets.__contains__('master_div_stm'+ sufix):
            self.widgets['master_div_stm' + sufix].destroy()
        self.winm.update()
    def create_message_div_frame(self,diversity,sufix,pt_keys,frame_name):
        # len_keys = len(list(diversity.triangles.keys()))
        scrolW = int(np.ceil(self.winm.winfo_width() * 0.130))
        scrolH = 4
        self.update_rowid()
        self.create_frame_div(frame_name, 'Dictionary of Set of Triangles','dst',sufix,scroll = False)
        self.widgets['scr_message_div'+sufix] = scrolledtext.ScrolledText(self.widgets['master_div_'+'dst'+sufix], width=scrolW,
                                                                      height=scrolH, wrap=tk.WORD)
        self.widgets['scr_message_div'+sufix].grid(column=0, row=self.rowid, sticky='WE', columnspan=2)
        for modi in pt_keys:
            string = 'DST[H_' + str(modi) + ']=[' + ','.join(diversity.sub_genomes_abc[modi]) + ']'
            self.widgets['scr_message_div'+sufix].insert(tk.END, string + '\n')
            self.widgets['scr_message_div'+sufix].see(tk.END)
            self.widgets['scr_message_div'+sufix].update()
        str1 = ','.join(['H_' + str(modi) for modi in pt_keys])
        self.widgets['scr_message_div'+sufix].insert(tk.END,
                                                 'Diversity(' + str1 + ')=' + str(diversity.diversity_pop) + '\n')
        self.widgets['scr_message_div'+sufix].see(tk.END)
        self.widgets['scr_message_div'+sufix].update()
        self.update_rowid()

        # self.widgets['master_div_psm'+sufix] = ttk.LabelFrame(self.widgets[key+'_tab'+sufix], text='parameters', height=10,
        #                                               width=self.widgets['width'])
        # self.widgets['master_div_psm'+sufix].grid(column=0, row=self.rowid, padx=2, pady=4, columnspan=2)
        self.create_frame_div(frame_name, 'parameters', 'psm', sufix, scroll=False)

        self.update_rowid()
        self.widgets['Policy_Tree_chosen'+sufix] = tk.StringVar()
        ttk.Label(self.widgets['master_div_psm'+sufix], text="Policy Tree:", width=20).grid(column=0, row=self.rowid)
        self.widgets['Policy Trees'+sufix] = ttk.Combobox(self.widgets['master_div_psm'+sufix], width=20,
                                                      textvariable=self.widgets['Policy_Tree_chosen'+sufix])
        self.widgets['Policy Trees'+sufix]['values'] = list()
        self.widgets['Policy Trees'+sufix].grid(column=1, row=self.rowid, padx=1, pady=2)
        self.widgets['Policy Trees'+sufix]['values'] = list(pt_keys)
        self.widgets['Policy Trees'+sufix].current(0)
        self.widgets['Policy Trees'+sufix].update()
    def create_frame_div(self,key,text,framestr,sufix,scroll,width = None,cl= None,clspan = None):
        if clspan is None:
            clspan = 2
        if cl is None:
            cl=0
        if width is None:
            width = 800
        if self.widgets.__contains__('master_div_'+framestr+sufix):
            self.widgets['master_div_'+framestr+sufix].destroy()
        self.widgets['master_div_'+framestr+sufix] = ttk.LabelFrame(self.widgets[key+'_tab'+sufix], text=text,
                                                     height=self.widgets['height'], width=self.widgets['width'])
        self.widgets['master_div_'+framestr+sufix].grid(column=cl, row=self.rowid, padx=8, pady=4, columnspan=clspan)
        if scroll:
            self.widgets['master_sf'+framestr+sufix] = ScrolledFrame(self.widgets['master_div_'+framestr+sufix], width=width, height=self.widgets['height'])
            self.widgets['master_sf'+framestr+sufix].pack(side="top", expand=1, fill="both")
            # Bind the arrow keys and scroll wheel
            self.widgets['master_sf'+framestr+sufix].bind_arrow_keys(self.widgets['master_div_'+framestr+sufix])
            self.widgets['master_sf'+framestr+sufix].bind_scroll_wheel(self.widgets['master_div_'+framestr+sufix])
            # Create a frame within the ScrolledFrame
            self.widgets['master_inner_frame_'+framestr+sufix] = self.widgets['master_sf'+framestr+sufix].display_widget(Frame)
            # Add a bunch of widgets to fill some space
    def create_subframe(self,framestr,sufix,width, height,pt_dict,pt_horizon_dict,action_list,observation_list,textlist,master,col=None):
        self.widgets['policytrees_' +framestr+ sufix] = dict()
        if col is None:
            col=0
        count = -1
        row = 0
        move = 0
        for ei in pt_dict.keys():
            if count == 5:
                count = 0
                row = row + 1
                move = move + 6
            else:
                count = count + 1
            horizon = pt_horizon_dict.get(ei)
            pt_st = ttk.LabelFrame(master, text=textlist[ei], width=width,height=height)
            pt_st.grid(column=int(ei) - move+col, row=row, padx=1, pady=2)
            self.widgets['policytrees_' +framestr+ sufix][ei] = Game_Tree(pt_st,width=width,height=height)
            policy_dict = dict()
            policy_dict[0] = pt_dict.get(ei)
            policy_tree = PolicyTree('Plot'+sufix, action_list=action_list,observation_list=observation_list)
            policy_tree.set_policy_dict(policy_dict)
            policy_tree.gen_policy_trees_memorysaved()
            self.plot_policy_tree(self.widgets['policytrees_' +framestr+ sufix][ei], policy_tree, horizon,width=width,height=height)  #
    def normilize_WH(self,diversity,width,height):
        c = 60
        w = 80
        num_os = diversity.get_num_os()
        height_list = list()
        width_list = list()
        for ei in diversity.triangles.keys():
            horizon_max_value = diversity.triangles_horizon.get(ei)
            num_max = np.power(num_os,horizon_max_value)
            height1 = int(math.floor(c * horizon_max_value + 2 * c * (horizon_max_value - 1)) + 2 * c)
            width1 = int(math.floor(num_max * w) + w)
            # print(height, width, height1, width1)
            ratioh = height / height1
            ratiow = width / width1
            ratio = np.mean([ratioh, ratiow])
            height_list.append(int(np.floor(height1 * ratioh)))
            width_list.append(int(np.floor(width1 * ratiow)))
        width = np.max(width_list)
        height = np.max(height_list)
        return width,height
    def plot_policy_tree(self,master,policy_tree,horizon_max_value,width=None,height=None):
        master.clear_game_object()
        node_list = policy_tree.get_nodelist()
        edgelist = policy_tree.get_edgelist()
        node_labels = policy_tree.get_node_labels()
        edge_labels = policy_tree.get_edge_labels()
        node_dict = policy_tree.get_node_dict()
        num_max = 1
        for level in node_dict.keys():
            if len(node_dict.get(level))>num_max:
                num_max = len(node_dict.get(level))
        if width  is None :
            width = 100
        if height is None:
            height = 200
        c = 60
        w = 80
        height1 = int(math.floor(c * horizon_max_value + 2 * c * (horizon_max_value - 1)) + 2 * c)
        width1 = int(math.floor(num_max * w) + w)
        # print(height, width,height1, width1)
        ratioh = height/height1
        ratiow =width/width1
        ratio = np.mean([ratioh,ratiow])
        height =int(np.floor( height1*ratioh))
        width = int(np.floor(width1 *ratiow))
        w = int(math.floor(width/(num_max  + 1)))
        c = int(math.floor(height/(horizon_max_value + 2  * (horizon_max_value - 1)+ 2 )))
        master.set_width(width)
        master.set_height(height)
        node_pos = dict()
        for level in node_dict.keys():
            ns = node_dict.get(level)
            y = c*(horizon_max_value - level)+2*c*(horizon_max_value-level)+c
            w = int(math.floor(width/len(ns)))
            X = range(int(w/2),width,w)
            for ni in range(0,len(ns)):
                node_pos[ns[ni]] = [X[ni],y]
        for ei in edgelist:
            x_start = node_pos.get(ei[0])[0]
            y_start = node_pos.get(ei[0])[1]
            x_end = node_pos.get(ei[1])[0]
            y_end = node_pos.get(ei[1])[1]
            edge = Game_Edge(master.get_canvas(),
                             x_start=x_start,y_start= y_start,x_end=x_end,y_end=y_end,
                             label_str = edge_labels.get(ei),id= ei,ratio=ratio)
            master.add_game_object(edge)
            #edge.active()
        for ni in node_list:
            x_start = node_pos.get(ni)[0]
            y_start = node_pos.get(ni)[1]
            node = Game_Node(master.get_canvas(),
                             x=x_start,y= y_start,
                             label_str = node_labels.get(ni),id= ni,ratio =ratio)
            master.add_game_object(node)
            master.node_ids[ni] = node
    def calculate_diversity(self,frame_name,sufix,diversity):
        scroll = False if sufix == '' else True
        # diversity = self.widgets['Diversity'+sufix]
        model = self.widgets['model_div'+sufix].get()
        domain = self.widgets['domain_div'+sufix].get()
        idid_did = self.widgets['idid_div' + sufix].get()
        if idid_did == self.pnames.simmod_pt[1]:
            did = self.Simmodels.IDID[domain][model].did
        else:
            did = self.Simmodels.IDID[domain][model]
        pt_keys = did.dbn.result.get('policy_dict').keys()
        modi = int(self.widgets['Policy_Tree_chosen'+sufix].get())

        self.create_frame_div(frame_name, 'Policy Tree@H_'+str(modi), 'ptm', sufix, scroll=False,cl=0,clspan =1 )
        pt_dict = dict()
        pt_dict[modi] =did.dbn.result.get('policy_dict')[modi]
        pt_keys = pt_dict.keys()
        height = self.widgets['height']
        width = int(np.ceil(self.widgets['width']))
        action_list = did.dbn.action_list
        observation_list = did.dbn.observation_list
        pt_horizon_dict = dict([(i, did.dbn.horizon) for i in pt_keys])
        textlist = dict([(i, 'H_%s' % i) for i in pt_keys])
        master = self.widgets['master_div_' + 'ptm' + sufix]
        self.create_subframe('ptm', '', width, height, pt_dict, pt_horizon_dict, action_list, observation_list,
                             textlist, master,col = -modi)

        self.create_frame_div(frame_name, 'Set of unique Triangles @H_'+str(modi), 'stm', sufix, scroll=scroll,width = 600,cl=1,clspan =1)
        pt_dict = dict([(ei,diversity.triangles.get(ei)) for ei in diversity.sub_genomes[modi]])
        pt_keys = pt_dict.keys()
        height = self.widgets['height']
        width = int(np.ceil(self.widgets['width']))
        width, height = self.normilize_WH(diversity, width, height)
        pt_horizon_dict = diversity.triangles_horizon
        textlist = diversity.Columns
        master = self.widgets['master_div_' + 'stm' + sufix] if sufix == '' else self.widgets['master_inner_frame_' + 'stm' + sufix]
        self.create_subframe('stm', '', width, height, pt_dict, pt_horizon_dict, action_list, observation_list,
                             textlist, master,col=1)
    # PolicyTree_tab
    def PolicyTree_tab(self,index):
        key = self.pnames.GUI_Tabs[index]
        sufix = self.pnames.GUI_Tabs_sc[index]
        tab = self.widgets[key + '_tab' + sufix]
        self.rowid = 0
        id = 9
        self.create_policytree_frame(tab, 9,key)
    def create_policytree_frame(self,tab,id, key_tab):
        master = self.create_frame(tab, id)
        frame_name = self.pnames.GUI_Frames[id]
        self.update_rowid()
        i = 0
        self.widgets['policytrees_' + frame_name] = dict()
        self.widgets['model_'+frame_name] = tk.StringVar()
        ttk.Label(master, text="Model:", width=9).grid(column=i, row=self.rowid)
        self.message_CompareModels['CB_' + frame_name] = ttk.Combobox(master, width=18,
                                                                      textvariable=self.widgets['model_'+frame_name])
        self.message_CompareModels['CB_' + frame_name]['values'] = list()
        self.message_CompareModels['CB_' + frame_name].grid(column=i+1, row=self.rowid, padx=1, pady=2)

        self.widgets['domain_'+frame_name] = tk.StringVar()
        ttk.Label(master, text="Domain:", width=9).grid(column=i+2, row=self.rowid)
        self.message_CompareDomains['CB_' + frame_name] = ttk.Combobox(master, width=18,
                                                                       textvariable=self.widgets['domain_'+frame_name])
        self.message_CompareDomains['CB_' + frame_name]['values'] = list()
        self.message_CompareDomains['CB_' + frame_name].grid(column=i+3, row=self.rowid, padx=1, pady=2)

        self.widgets['idid_' + frame_name] = tk.StringVar()
        ttk.Label(master, text=" IDID/DID:", width=9).grid(column=i+4, row=self.rowid)
        self.widgets['CB_' + frame_name ] = ttk.Combobox(master, width=18, textvariable=self.widgets['idid_' + frame_name])
        self.widgets['CB_' + frame_name ]['values'] = self.pnames.simmod_pt
        self.widgets['CB_' + frame_name].grid(column=i+5, row=self.rowid, padx=4, pady=2)

        plot_policytree = ttk.Button(master, text="Plot", width=10,
                             command=lambda: self.plot_policytree(frame_name,master,key_tab,tab))
        plot_policytree.grid(column=i+6, row=self.rowid, padx=4, pady=2)
        close = ttk.Button(master, text="Close all", width=10,
                           command=lambda: self.close_all_pts(frame_name, key_tab))
        close.grid(column=i +7, row=self.rowid, padx=1, pady=2, sticky='WE')
        self.update_rowid()
        self.widgets['rowid_' + key_tab] = self.rowid
    def close_all_pts(self,frame_name, key_tab):
        for ei in  self.widgets['policytrees_' + frame_name].keys():
            self.widgets['policytrees_' + frame_name][ei].destroy()
            self.widgets['rowid_' + key_tab] = self.widgets['rowid_' + key_tab] - 1
        self.widgets['policytrees_' + frame_name] = dict()
        if self.widgets['rowid_' + key_tab] <=0:
            self.widgets['rowid_' + key_tab] = 2
    def plot_policytree(self,frame_name,master,key_tab,tab):
        if self.popup_messeage(frame_name):
            return -1
        keym = self.widgets['model_'+frame_name].get()
        keyd = self.widgets['domain_'+frame_name].get()
        type = self.widgets['idid_' + frame_name].get()
        if type == self.pnames.simmod_pt[0]:
            # idid
            pt = self.Simmodels.IDID[keyd][keym].dbn.result['policy_tree']
        else:
            pt = self.Simmodels.IDID[keyd][keym].did.dbn.result['policy_tree']
        self.widgets['rowid_' + key_tab] = self.widgets['rowid_' + key_tab] + 1
        id = len(self.widgets['policytrees_' + frame_name])
        width = 300
        height = 400
        text = keyd +'|'+keym +'|'+ type
        self.widgets['policytrees_' + frame_name][id] = ttk.LabelFrame(tab, text=text,width=width,height=height)
        self.widgets['policytrees_' + frame_name][id].grid(column=0, row=self.widgets['rowid_' + key_tab], padx=1, pady=2)
        gt= Game_Tree(self.widgets['policytrees_' + frame_name][id],width=width,height=height)
        horizon = pt.get_horizon()
        self.plot_policy_tree(gt,pt, horizon,width=width,height=height)

oop = main_gui()
oop.winm.mainloop()
# oop.win_wel.mainloop()
