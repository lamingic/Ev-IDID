# import time
# import os
from itertools import islice
from textwrap import dedent
from threading import Thread
# import tkinter.font as font
import random
import math
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# import os
# import sys
import pysmile
import pysmile_license
import numpy as np
import re
#import datetime
#import math
from graphviz import Source
import graphviz as G
from graphviz import Digraph
import pydotplus
import collections
import numpy as np
import pysmile
import re
from tqdm import tqdm
import random
import math
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import tkinter as tk
import numpy as np
import math
import os
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
from tkscrolledframe import ScrolledFrame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# def welcome(self):
#     import matplotlib.image as mpimg
#     self.win_wel = tk.Tk()
#     self.win_wel.wm_attributes('-topmost', 1)
#     self.win_wel.title('Welcome to '+ self.idid_title)  # Disable resizing the GUI by passing in False/False
#     self.win_wel.resizable(False, False)  # Enable resizing x-dimension, disable y-dimension
#     self.master_wel = ttk.LabelFrame(self.win_wel, text='Welcome')
#     self.master_wel.grid(column=0, row=0, padx=8, pady=4)
#     menuBar = Menu(self.win_wel)
#     self.win_wel.config(menu=menuBar)
#     openVar = tk.IntVar()
#     beginVar = tk.IntVar()
#     quitVar = tk.IntVar()
#     fileMenu = Menu(menuBar, tearoff=0)
#     fileMenu.add_checkbutton(label="Begin(开始)", command=self.createWidgets, variable=beginVar)
#     menuBar.add_cascade(label="File", menu=fileMenu)
#     helpMenu = Menu(menuBar, tearoff=0)
#     aboutVar = tk.IntVar()
#     helpMenu.add_checkbutton(label="About(关于)", command=self.about, variable=aboutVar)
#     helpMenu.add_separator()
#     helpMenu.add_checkbutton(label="Quit(退出)", command=self.win_wel.destroy, variable=quitVar)
#     menuBar.add_cascade(label="Help", menu=helpMenu)
#     # ~ Tab Control introduced here -----------------------------------------
#
#     photo = './idid_platform.png'
#     self.fig = plt.figure(dpi=150)
#     frame = plt.gca()
#     frame.axes.get_yaxis().set_visible(False)
#     frame.axes.get_xaxis().set_visible(False)
#     plt.axis('off')
#
#     canvas = FigureCanvasTkAgg(self.fig, master=self.master_wel)
#     canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
#     cwidg = canvas.get_tk_widget()
#     # Get the width and height of the *figure* in pixels
#     w = self.fig.get_figwidth() * self.fig.get_dpi()
#     h = self.fig.get_figheight() * self.fig.get_dpi()
#     # Generate a blank tkinter Event object
#     evnt = tk.Event()
#     # Set the "width" and "height" values of the event
#     evnt.width = w * 2
#     evnt.height = h * 2
#     # Set the width and height of the canvas widget
#     cwidg.configure(width=w, height=h)
#     # Pass the generated event object to the FigureCanvasTk.resize() function
#     canvas.resize(evnt)
#     self.img = mpimg.imread(photo)
#     plt.imshow(self.img)

# def test_scroll_frame(self):
#     from tkscrolledframe import ScrolledFrame
#     self.test_scroll = ttk.Frame(self.tabControl)  # Add a second tab
#     self.tabControl.add(self.test_scroll, text='test_scroll frame')  # Make second tab visible
#     self.tabControl.pack(expand=1, fill="both")  # Pack to make visible
#     master = ttk.LabelFrame(self.test_scroll, text='Simulation')
#     master.grid(column=0, row=self.rowid, padx=8, pady=4, columnspan=3)
#             # Create a ScrolledFrame widget
#     root = master
#     sf = ScrolledFrame(root, width=640, height=480)
#     sf.pack(side="top", expand=1, fill="both")
#     # Bind the arrow keys and scroll wheel
#     sf.bind_arrow_keys(root)
#     sf.bind_scroll_wheel(root)
#     # Create a frame within the ScrolledFrame
#     inner_frame = sf.display_widget(Frame)
#     # Add a bunch of widgets to fill some space
#     num_rows = 16
#     num_cols = 16
#     for row in range(num_rows):
#         for column in range(num_cols):
#             w = Label(inner_frame,
#                       width=15,
#                       height=5,
#                       borderwidth=2,
#                       relief="groove",
#                       anchor="center",
#                       justify="center",
#                       text=str(row * num_cols + column))
#
#             w.grid(row=row,
#                    column=column,
#                    padx=4,
#                    pady=4)
def add_compare_domains(self):
    self.set_None()
    self.domain_parameters = nsp.DomainParameters()
    popup_window = PW.popup_window('Domain', self.domain_parameters, self.CompareDomains, self.message_CompareDomains)
    # self.popup_window_DP(self.domain_parameters)
    def popup_window_DP(self,domain):
        win = tk.Toplevel()
        win.wm_attributes('-topmost',1)
        win.title('Domain parameters')
        master = tk.LabelFrame(win, text=' Domain parameters')
        rowid = 0
        master.grid(column=0, row=rowid, padx=8, pady=4)

        for i in range(0,len(domain.parameters)):
            rowid = rowid + 1
            key = domain.parameters[i]
            ttk.Label(master, text=key, width=20).grid(column=0, row=rowid)
            if key.__contains__('domain_name'):
                domain_Chosen = ttk.Combobox(master, width=18, textvariable=domain.parameters_type[key])
                domain_Chosen['values'] = domain.parameters_setting[key]
                domain_Chosen.grid(column=1, row=rowid, padx=1,  pady=2)
                domain_Chosen.current(0)
            else:
                Entered = ttk.Entry(master, width=20, textvariable=domain.parameters_type[key])
                Entered.grid(column=1, row=rowid, padx=1, pady=2)
                Entered.delete(0)
                Entered.insert(0, domain.parameters_setting[key])
            rowid = rowid + 1

        button_confirm = ttk.Button(master, text="Confirm", command=lambda: self.update_domain_parameters(domain))
        button_confirm.grid(column=0, row=rowid, padx=1, pady=2)
        button_close = ttk.Button(master, text="Close", command=win.destroy)
        button_close.grid(column=1, row=rowid, padx=1, pady=2)
    def update_domain_parameters(self,domain):
        domain.update()
        index = str(len(self.CompareDomains) + 1)
        self.domain_parameters.Name = 'Domain @' + index + '-' + self.domain_parameters.values['domain_name']
        self.CompareDomains[self.domain_parameters.Name] = self.domain_parameters
        # message
        for ei in self.message_CompareDomains['combobox_list']:
            self.message_CompareDomains[ei]['values'] = list(self.CompareDomains.keys())
            self.message_CompareDomains[ei].current(0)
            self.message_CompareDomains[ei].update()
    def add_compare_models(self):
        self.set_None()
        self.ModelParameters = nsp.ModelParameters()
        index = str(len(self.CompareModels) + 1)
        name = 'IDID @'+ index+ '-'
        for key in self.Solvers.keys():
            for keyi in self.Solvers.get(key).keys():
                self.ModelParameters.values[key][keyi]['name'] = self.Solvers.get(key).get(keyi).get()
                name = name+'-'+self.ModelParameters.values[key][keyi]['name']
                if self.Solvers[key][keyi].get().__contains__('GA'):
                    self.popup_window_GA(self.ModelParameters.values.get(key).get(keyi))
                if self.Solvers[key][keyi].get().__contains__('MD'):
                    self.popup_window_MD(self.ModelParameters.values.get(key).get(keyi))
        self.ModelParameters.Name = name
        self.CompareModels[self.ModelParameters.Name] = self.ModelParameters
        # message
        for ei in self.message_CompareModels['combobox_list']:
            self.message_CompareModels[ei]['values'] = list(self.CompareModels.keys())
            self.message_CompareModels[ei].current(0)
            self.message_CompareModels[ei].update()
    def popup_window_GA1(self,solver):
        win = tk.Toplevel()
        win.wm_attributes('-topmost', 1)
        win.title( solver['type']+'@'+solver['pointer'] )
        if solver['name']=='GGA' or solver['name']=='PGA':
            solver['parameters'] = nsp.GGA_ps()
        if solver['name'] == 'GA':
            solver['parameters'] = nsp.GA_ps()
        if solver['name'] == 'MGGA' or solver['name'] == 'MPGA':
            solver['parameters'] = nsp.MGGA_ps()
        if solver['name'] == 'MGA':
            solver['parameters'] = nsp.MGA_ps()
        master = ttk.LabelFrame(win, text=solver['name'] +' parameters')
        rowid = 0
        master.grid(column=0, row=rowid, padx=8, pady=4)
        count = 0
        for i in range(0,len( solver['parameters'].parameters)):
            key =  solver['parameters'].parameters[i]
            if key.__contains__('_method'):
                rowid = rowid + 1
                ttk.Label(master, text=key).grid(column=0, row=rowid)
                Chosen = ttk.Combobox(master, width=18, textvariable= solver['parameters'].parameters_type[key])
                Chosen['values'] =  solver['parameters'].parameters_setting[key]
                Chosen.grid(column=1, row=rowid, padx=1, pady=2)
                Chosen.current(1)

            if key.__contains__('_size') or key.__contains__('_rate'):
                rowid = rowid + 1
                ttk.Label(master, text=key).grid(column=0, row=rowid)
                Entry = ttk.Entry(master, width=20, textvariable= solver['parameters'].parameters_type[key])
                Entry.grid(column=1, row=rowid, padx=1, pady=2)
                Entry.delete(0)
                Entry.insert(0,  solver['parameters'].parameters_setting[key])

            if key.__contains__('_mode') :
                count = count + 1
                if count == 2:
                    count = 0
                else:
                    rowid = rowid + 1
                Check = tk.Checkbutton(master, text=key, variable= solver['parameters'].parameters_type[key])
                if  solver['parameters'].parameters_setting[key]:
                    Check.select()
                else:
                    Check.deselect()
                Check.grid(column=count, row=rowid, sticky=tk.W)


        rowid = rowid + 1
        button_confirm = ttk.Button( master, text="Confirm", command=lambda :self.update_ga_parameters(solver))
        button_confirm.grid(column=0, row=rowid, padx=1, pady=2)
        button_close = ttk.Button( master, text="Close", command= win.destroy)
        button_close.grid(column=1, row=rowid, padx=1, pady=2)
    def update_ga_parameters(self,solver):
        solver['parameters'].update()
        # solver['values'] = solver['parameters'].values
        self.ModelParameters.display()
    def popup_window_Table(self):
        # Create an instance of tkinter frame
        win = tk.Toplevel()
        win.wm_attributes('-topmost', 1)
        win.title('Testing results')
        rowid = 0
        scrolW = 100;
        scrolH = 10
        master = ttk.LabelFrame(win, text='Time comsumption of algorithims')
        master.grid(column=0, row=rowid, padx=8, pady=4)

        # Set the size of the tkinter window
        win.geometry("700x350")

        # Add a Treeview widget
        tree = ttk.Treeview(master, column=("c1", "c2", "c3"), show='headings', height=5)
        tree.column("# 1", anchor=tk.CENTER)
        tree.heading("# 1", text="ID")
        tree.column("# 2", anchor=tk.CENTER)
        tree.heading("# 2", text="FName")
        tree.column("# 3", anchor=tk.CENTER)
        tree.heading("# 3", text="LName")

        # Insert the data in Treeview widget
        tree.insert('', 'end', text="1", values=('1', 'Joe', 'Nash'))
        tree.insert('', 'end', text="1", values=('2', 'Joe', 'Nash'))
        tree.insert('', 'end', text="2", values=('2', 'Emily', 'Mackmohan'))
        tree.insert('', 'end', text="3", values=('3', 'Estilla', 'Roffe'))
        tree.insert('', 'end', text="4", values=('4', 'Percy', 'Andrews'))
        tree.insert('', 'end', text="5", values=('5', 'Stephan', 'Heyward'))
        # tree = ttk.Treeview(master, selectmode='browse' )
        # tree.insert('', 0, iid=1, text='Parent1')
        # tree.insert('', 0, iid=2, text='Parent2')
        #
        # tree.insert(1, 0, text='Child 1')
        # tree.insert(2, 0, text='Child 2')
        # child3 = tree.insert(2, 0, text='Child 3')
        # tree.move(child3, 1, 'end')
        # tree.delete(2)
        tree.pack()
    def popup_table(self):
        # Create an instance of tkinter frame
        win = tk.Toplevel()
        win.wm_attributes('-topmost', 1)
        win.title('Testing results')
        rowid = 0
        scrolW = 100;
        scrolH = 10
        self.df = None
        master = ttk.LabelFrame(win, text='Time comsumption of algorithims')
        master.grid(column=0, row=rowid, padx=8, pady=4)
        # Set the size of the tkinter window
        win.geometry("700x350")
        self.text = tk.Text(master)
        self.text.pack()
        from tkinter.filedialog import askopenfilename
        import pandas as pd
        name = askopenfilename(filetypes=[('CSV', '*.csv',), ('Excel', ('*.xls', '*.xlsx'))])

        if name:
            if name.endswith('.csv'):
                self.df = pd.read_csv(name)
            else:
                self.df = pd.read_excel(name)

            self.filenamex = name
        if self.df is not None:
            self.text.insert('end', self.filenamex + '\n')
            self.text.insert('end', str(self.df.head()) + '\n')

# def get_data_from_excel(self, excel_dir):  # 读取excel，取出所有sheet要执行的接口信息，返回列表
#     from openpyxl import load_workbook
#     work_book = load_workbook(excel_dir)
#     all_sheets = work_book.sheetnames
#     api_info_list = []
#     dataset = dict()
#     for i in range(0, len(all_sheets)):
#         work_sheet = all_sheets[i]
#         sheet = work_book[work_sheet]
#         rows = sheet.max_row
#         dataset['rows'] = rows
#         dataset['columns'] = 0
#         data = list()
#         for r in range(0, rows):  # 从第2行开始取数据
#             api_data = {}
#             temp_list = []
#             dataset['columns'] = np.max([len(sheet[str(r + 1)]),dataset['columns']])
#             for n in range(0, len(sheet[str(r + 1)])):
#                 # if sheet[str(r + 1)][0].value == 1:  # 把标识为1的行，此行的每个单元格数据加入到临时list
#                 if sheet[str(r + 1)][n].value is None:
#                     temp_list.append('-')
#                 else:
#                     temp_list.append(sheet[str(r + 1)][n].value)
#             # print(temp_list)
#             data.append(temp_list)
#             for param in temp_list:  # 把临时表list中有'='符号的元素分割开
#                 if '=' in str(param):
#                     p = param.split('=')
#                     api_data[p[0]] = p[1]
#             if api_data:
#                 api_info_list.append(api_data)
#         dataset['data'] = data
#     return api_info_list,dataset
# def read_data_from_excel(self,excel_dir):
    #     all_sheets = pd.read_excel(excel_dir,sheet_name='Sheet1',header = None,keep_default_na=False)
    #     api_info_list = []
    #     dataset = dict()
    #     # print(all_sheets)
    #     sheet = all_sheets.values
    #     rows = sheet.shape[0]
    #     columns = sheet.shape[1]
    #     dataset['rows'] = rows
    #     dataset['columns'] = sheet.shape[1]
    #     data = list()
    #
    #     for r in range(0, rows):  # 从第2行开始取数据
    #         api_data = {}
    #         temp_list = []
    #         for n in range(0, columns):
    #             if sheet[r,n] == '':
    #                 temp_list.append('-')
    #             else:
    #                 temp_list.append(sheet[r,n])
    #         data.append(temp_list)
    #         # print(temp_list)
    #     dataset['data'] = data
    #     return api_info_list, dataset