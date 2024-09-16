import tkinter as tk
from tkinter import ttk
class popup_window(object):
    def __init__(self,title,parameters,CompareDMs = None,message_CompareDMs = None):
        self.win = tk.Toplevel()
        self.win.wm_attributes('-topmost', 1)
        self.win.iconbitmap("./logo.ico")
        self.title = title
        self.win.title(self.title)
        self.master = ttk.LabelFrame(self.win, text= self.title + '\'s parameters')
        self.rowid = 0
        self.master.grid(column=0, row=self.rowid, padx=8, pady=4)
        self.parameters = parameters
        if not CompareDMs is None:
            self.CompareDMs = CompareDMs
        if not message_CompareDMs is None:
            self.message_CompareDMs = message_CompareDMs
        self.parameters_type = self.parameters.parameters_type
        self.parameters_setting = self.parameters.parameters_setting
        self.build_window()
    def update_rowid(self,num=None):
        if num is None:
            self.rowid = int(self.rowid + 1)
        else:
            self.rowid = int(self.rowid + num)
    def build_window(self):
        count = 0
        for key in self.parameters.parameters:
            if key.__contains__('_method') or key.__contains__('_name'):
                self.update_rowid()
                tk.Label(self.master, text=key).grid(column=0, row = self.rowid)
                Chosen = ttk.Combobox(self.master, width=18, textvariable = self.parameters_type[key])
                Chosen['values'] = self.parameters_setting[key]
                Chosen.grid(column=1, row=self.rowid, padx=1, pady=2)
                Chosen.current(1)

            if key.__contains__('_size') or key.__contains__('_rate')or key.__contains__('num_'):
                self.update_rowid()
                ttk.Label(self.master, text=key).grid(column=0, row = self.rowid)
                Entry = ttk.Entry(self.master, width=20, textvariable=self.parameters_type[key])
                Entry.grid(column=1, row=self.rowid, padx=1, pady=2)
                Entry.delete(0)
                Entry.insert(0, self.parameters_setting[key])

            if key.__contains__('_mode'):
                count = count + 1
                if count == 2:
                    count = 0
                else:
                    self.update_rowid()
                Check = tk.Checkbutton(self.master, text=key, variable=self.parameters_type[key])
                if self.parameters_setting[key]:
                    Check.select()
                else:
                    Check.deselect()
                Check.grid(column=count, row=self.rowid, sticky=tk.W)
        self.update_rowid()
        button_confirm = ttk.Button(self.master, text="Confirm", command = self.parameters.update)
        button_confirm.grid(column=0, row=self.rowid, padx=1, pady=2)
        button_close = ttk.Button(self.master, text="Update", command = self.update_parameters)
        button_close.grid(column=1, row=self.rowid, padx=1, pady=2)
    def update_parameters(self):
        if self.title == 'Domain':
            index = str(len(self.CompareDMs) + 1)
            self.parameters.Name = 'Domain @' + index + '-' + self.parameters.values['domain_name']
            self.CompareDMs[self.parameters.Name] = self.parameters
            # message
            for ei in self.message_CompareDMs['CB_list']:
                self.message_CompareDMs[ei]['values'] = list(self.CompareDMs.keys())
                self.message_CompareDMs[ei].current(len(self.CompareDMs)-1)
                self.message_CompareDMs[ei].update()
        self.win.destroy()