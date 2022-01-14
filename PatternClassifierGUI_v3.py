### GUI
import csv
from os import remove
from time import sleep
from tkinter import *
from tkinter import messagebox, ttk
from tkinter.filedialog import askdirectory
from re import match, split
from matplotlib.backends.backend_pdf import PdfPages
from glob import glob
from traceback import format_exc
from pandas import concat, read_csv
from gc import collect

from multiprocessing import Pool, cpu_count, current_process

from pattern_classifier_mainloop import run_classification

class OnlyNumbersEntry(Entry):
    """
    Restrict to number entries on text entries
    """
    def __init__(self, master=None, value=None, positive_num=False, **kwargs):
        self.var = StringVar(master, value=value)
        Entry.__init__(self, master, textvariable=self.var, **kwargs)

        if positive_num:
            self.var.trace('w', self.validate_natural)
        else:
            self.var.trace('w', self.validate_real)

    def isfloat(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def validate_real(self, *args):
        if match(r"(\+|\-)?[0-9,.]*$", self.var.get()) == None:
            corrected = ''.join(filter(lambda v: match(r"(\+|\-)?[0-9,.]*$", v), self.var.get()))
            self.var.set(corrected)

            if not self.isfloat(self.var.get()):
                corrected = ''.join(filter(self.isfloat, self.var.get()))
                self.var.set(corrected)

    def validate_natural(self, *args):
        if not self.isfloat(self.var.get()):
            corrected = ''.join(filter(self.isfloat, self.var.get()))
            self.var.set(corrected)


class Checkbar(Frame):
    """
    Checkbar class
    Not implemented
    """
    def __init__(self, parent=None, picks=[], side=LEFT, anchor=W):
        Frame.__init__(self, parent)
        self.vars = []
        for pick in picks:
            var = IntVar()
            var.set(1)
            chk = Checkbutton(self, text=pick, variable=var)
            chk.pack(side=side, anchor=anchor, expand=YES)
            self.vars.append(var)

    def state(self):
        return [var.get() for var in self.vars]


class GUI(Tk):
    """
    The main GUI class
    """
    def __init__(self):
        # GUI
        Tk.__init__(self)
        self.title("Spike pattern classifier")
        # self.iconbitmap(r'G:\My Drive\Documents\PycharmProjects\PatternClassifierDist\icon.ico')
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Organized in two frames. Up is for parameters; bottom is only the run buttons
        self.top_frame = Frame(self)
        self.bottom_frame = Frame(self)
        self.top_frame.pack(side=TOP)
        self.bottom_frame.pack(side=BOTTOM)

        ##### Initialize globals #####
        self.input_path = StringVar(self)
        self.output_path = StringVar(self)
        self.master_sheet_name = StringVar(self)
        self.brain_area = StringVar(self)
        self.status = StringVar(self)

        # This variable is to detect whether output has been set;
        # Program will also check for memory_key_pairing_list integrity (therefore also input validity)
        self.output_path_set = BooleanVar(self)
        self.pool = None

        self.number_of_simulations = IntVar(self)
        self.pre_stimulus_time = DoubleVar(self)
        self.post_stimulus_time = DoubleVar(self)
        self.pre_stimulus_raster = DoubleVar(self)
        self.post_stimulus_raster = DoubleVar(self)
        self.number_of_cores = IntVar(self)

        # Progress bar is not implemented yet
        self.progressbar_progress = DoubleVar(self)
        self.progressbar_increment = DoubleVar(self)

        self.classifier_methods = ['rcorr', 'count', 'cross_corr', 'dtw']
        self.sigma_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        self.bin_set = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 'Stim duration']
        self.relevant_memory_name_pieces = list()
        self.memory_key_pairing_list = list()

        self.menubar = Menu(self)
        self.menubar.add_command(label="About", command=self.show_popup)

        # display the menu
        self.config(menu=self.menubar)

        ##### Initialize widgets #####
        left_row_counter = 0
        right_row_counter = 0

        self.classifier_section_label = Label(self.top_frame, text="Classifier parameters")
        self.classifier_section_label.grid(row=left_row_counter, column=0, sticky=E)
        left_row_counter += 1
        self.classifier_section_label.config(font=('Helvetica', 20))

        self.input_path_button = Button(self.top_frame, text='Choose input folder', command=self.choose_input_path)
        self.input_path_button.grid(row=left_row_counter, column=0, sticky=E)
        left_row_counter += 1

        self.input_label = Label(self.top_frame, textvariable=self.input_path)
        self.input_label.grid(row=left_row_counter, column=0, sticky=E)
        left_row_counter += 1

        self.output_path_button = Button(self.top_frame, text='Choose output folder', command=self.choose_output_path)
        self.output_path_button.grid(row=left_row_counter, column=0, sticky=E)
        left_row_counter += 1

        self.input_label = Label(self.top_frame, textvariable=self.output_path)
        self.input_label.grid(row=left_row_counter, column=0, sticky=E)
        left_row_counter += 1

        self.number_of_simulations_label = Label(self.top_frame, text="Number of Simulations:")
        self.number_of_simulations_label.grid(row=left_row_counter, column=0, sticky=E)

        self.number_of_simulations_entry = OnlyNumbersEntry(self.top_frame, width=8, value=1000, positive_num=True)
        self.number_of_simulations_entry.grid(row=left_row_counter, column=1, sticky=E)
        left_row_counter += 1

        self.pre_stimulus_time_label = Label(self.top_frame, text="Time before stimulus (s); (negative number for offset):")
        self.pre_stimulus_time_label.grid(row=left_row_counter, column=0, sticky=E)

        self.pre_stimulus_time_entry = OnlyNumbersEntry(self.top_frame, width=8, value=0)
        self.pre_stimulus_time_entry.grid(row=left_row_counter, column=1, sticky=E)
        left_row_counter += 1

        self.post_stimulus_time_label = Label(self.top_frame, text="Time after stimulus (s):")
        self.post_stimulus_time_label.grid(row=left_row_counter, column=0, sticky=E)

        self.post_stimulus_time_entry = OnlyNumbersEntry(self.top_frame, width=8, value=2, positive_num=True)
        self.post_stimulus_time_entry.grid(row=left_row_counter, column=1, sticky=E)
        left_row_counter += 1

        self.brain_area_label = Label(self.top_frame, text="Pick brain area: "
                                                 "NCM: all stimuli matter;\n HVC: only BOS matters (first stimulus)")
        self.brain_area_label.grid(row=left_row_counter, column=0, sticky=E)

        self.brain_area_choices = {'NCM', 'HVC'}
        self.brain_area.set('NCM')
        self.brain_area_popup = OptionMenu(self.top_frame, self.brain_area, *self.brain_area_choices)
        self.brain_area_popup.grid(row=left_row_counter, column=1, sticky=E)
        left_row_counter += 1

        # Checkbar.state() to output the list; not implemented yet

        self.brain_area_info_label = Label(self.top_frame, text="Classifier algorithms to run:")
        self.brain_area_info_label.grid(row=left_row_counter, column=0, sticky=E)
        left_row_counter += 1
        self.classifier_methods_checkbar = Checkbar(self.top_frame, self.classifier_methods)
        self.classifier_methods_checkbar.grid(row=left_row_counter, column=0, sticky=E)
        left_row_counter += 1

        self.brain_area_info_label = Label(self.top_frame, text="Sigma set (for rcorr and cross_corr):")
        self.brain_area_info_label.grid(row=left_row_counter, column=0, sticky=E)
        left_row_counter += 1
        self.sigma_set_checkbar = Checkbar(self.top_frame, self.sigma_set)
        self.sigma_set_checkbar.grid(row=left_row_counter, column=0, sticky=E)
        left_row_counter += 1

        self.brain_area_info_label = Label(self.top_frame, text="Bin size set (for count and dtw):")
        self.brain_area_info_label.grid(row=left_row_counter, column=0, sticky=E)
        left_row_counter += 1
        self.bin_set_checkbar = Checkbar(self.top_frame, self.bin_set)
        self.bin_set_checkbar.grid(row=left_row_counter, column=0, sticky=E)
        left_row_counter += 1

        self.all_cores = IntVar()
        self.all_cores.set(0)
        self.number_of_cores_box = Checkbutton(self.top_frame, variable=self.all_cores, onvalue=1, offvalue=0,
                                          text="Use all processors? (might make PC slow while running)")
        self.number_of_cores_box.grid(row=left_row_counter, column=0, sticky=E)
        left_row_counter += 1

        self.plotting_section_label = Label(self.top_frame, text="Plotting parameters")
        self.plotting_section_label.grid(row=right_row_counter, column=2, sticky=N)
        right_row_counter += 1
        self.plotting_section_label.config(font=('Helvetica', 20))

        self.pre_stimulus_raster_label = Label(self.top_frame, text="Time before stimulus in rasterplots (s):")
        self.pre_stimulus_raster_label.grid(row=right_row_counter, column=2, sticky=N)

        self.pre_stimulus_raster_entry = OnlyNumbersEntry(self.top_frame, width=8, value=2)
        self.pre_stimulus_raster_entry.grid(row=right_row_counter, column=3)
        right_row_counter += 1

        self.post_stimulus_raster_label = Label(self.top_frame, text="Time after stimulus in rasterplots (s):")
        self.post_stimulus_raster_label.grid(row=right_row_counter, column=2)

        self.post_stimulus_raster_entry = OnlyNumbersEntry(self.top_frame, width=8, value=4)
        self.post_stimulus_raster_entry.grid(row=right_row_counter, column=3)
        right_row_counter += 1

        self.master_sheet_name_label = Label(self.top_frame, text="Master sheet prefix (e.g. MML_AHSKF_171009):")
        self.master_sheet_name_label.grid(row=right_row_counter, column=2)

        self.master_sheet_name_entry = Entry(self.top_frame, width=16)
        self.master_sheet_name_entry.grid(row=right_row_counter, column=3)
        right_row_counter += 1

        self.memory_key_pairing_button = Button(self.bottom_frame, text="Re-check memory-key pairing",
                                           command=self.check_memory_key_pairing)
        self.memory_key_pairing_button.pack()

        self.run_button = Button(self.bottom_frame, text="Run pattern classifier",
                            command=self.load_classifier_variables)
        self.run_button.config(font=('helvetica', 30, 'underline italic'))
        self.run_button.pack()

        # status_label = Label(right_frame, textvariable=status)
        # status_label.pack()
        # progressbar = ttk.Progressbar(right_frame, variable=progressbar_progress,
        #                               orient=HORIZONTAL, length=100, mode='determinate')
        # progressbar.pack()
        # Menu bar

    def show_popup(self):
        self.withdraw()
        popup = Toplevel()
        popup.title("About")
        msg = Message(popup, text="This program runs the intertrial Pattern Classifier"
                                  " for neuronal recording data \n"
                                  "Based on Caras et al., 2016 \n"
                                  "Developer: Matheus Macedo-Lima\n"
                                  "Questions, problems, compliments: math.biomatter@gmail.com")
        # popup.iconbitmap(r'c:\Users\Matheus\PycharmProjects\polar_hist\Gouache-arabidopsis-thaliana.ico')
        button = Button(popup, text="Close", command=lambda: self.kill_popup(popup))
        msg.pack()
        button.pack()

    def kill_popup(self, popup):
        if type(popup) == list:
            for item in popup:
                item.destroy()
                self.deiconify()
        else:
            popup.destroy()
            self.deiconify()


    def check_memory_key_pairing(self):
        """
        This funcion checks for memory vs key file pairings.
        More than one memory per key is allowed.
        More than one key per memory will flag an error.
        """
        def show_problem_file(problem_memory_path, problem_key_path_or_list=None):
            self.show_memory_checkbar_window(self.input_path.get(), run_check_memory_key_pairings=True)
            memory_key_table_window = Toplevel(self)
            memory_key_table_window.title("Problem file (might not be all of them. Fix it and rerun)")

            if not isinstance(problem_key_path_or_list, list):
                cur_memory_name = "_".join(split("_*_", split("\\\\", problem_memory_path)[-1][:-4]))
                cur_key_name = "_".join(split("_*_", split("\\\\", problem_key_path_or_list)[-1][:-4]))
                Label(memory_key_table_window, text=cur_memory_name).grid(row=0,
                                                                          column=0,
                                                                          sticky=E)
                Label(memory_key_table_window, text=cur_key_name).grid(row=0,
                                                                       column=1,
                                                                       sticky=W)

                self.show_memory_checkbar_window(self.input_path.get(), run_check_memory_key_pairings=True)
            elif len(problem_key_path_or_list) > 0:
                memory_key_table_window_row_counter = 0
                cur_memory_name = "_".join(split("_*_", split("\\\\", problem_memory_path)[-1][:-4]))
                for problem_key in problem_key_path_or_list:
                    cur_key_name = "_".join(split("_*_", split("\\\\", problem_key)[-1][:-4]))
                    Label(memory_key_table_window, text=cur_memory_name).grid(row=memory_key_table_window_row_counter,
                                                                              column=0,
                                                                              sticky=E)
                    Label(memory_key_table_window, text=cur_key_name).grid(row=memory_key_table_window_row_counter,
                                                                           column=1,
                                                                           sticky=W)
                    memory_key_table_window_row_counter += 1
            else:
                cur_memory_name = "_".join(split("_*_", split("\\\\", problem_memory_path)[-1][:-4]))
                Label(memory_key_table_window, text=cur_memory_name).grid(row=0,
                                                                          column=0,
                                                                          sticky=E)
        memory_paths = glob(self.input_path.get() + '\\*txt')

        memory_list = list()
        key_list = list()

        for memory_path in memory_paths:
            split_memory_path = split("_*_", split("\\\\", memory_path)[-1][:-4])
            cur_keyf = [item for idx, item in enumerate(split_memory_path) if
                        self.relevant_memory_name_pieces[idx] == 1]

            cur_key = glob(self.input_path.get() + '\\*' + cur_keyf[0] + "*KEY*csv")

            for item in cur_keyf[1:]:
                cur_key = [keys for keys in cur_key if "_" + item + "_" in keys]

            # if len(cur_key) > 1:
            #     messagebox.showerror("Warning",
            #                          "More than one key file was matched. Check your relevant memory name pieces!")
            #
            #     show_problem_file(memory_path, cur_key)
            #     return
            if len(cur_key) > 0:
                key_list.extend(cur_key)
            else:
                cur_key = glob(self.input_path.get() + '\\*' + "_*".join(cur_keyf) + "_*key*csv")  # try lower case
                if len(cur_key) > 1:
                    messagebox.showerror("Warning",
                                         "More than one key file was matched. Check your relevant memory name pieces!")
                    show_problem_file(memory_path, cur_key)

                    return
                if len(cur_key) > 0:
                    key_list.extend(cur_key)
                else:
                    show_problem_file(memory_path, cur_key)
                return

            # Append the appropriate number of memory path repetitions in case one memory has more than one key
            memory_list.extend([memory_path for _ in range(len(cur_key))])

        memory_key_table_window = Toplevel(self)
        memory_key_table_window.title("Looking good!")

        self.memory_key_pairing_list = list(zip(memory_list, key_list))
        memory_key_table_window_row_counter = 0
        for memory, key in zip(memory_list, key_list):
            cur_memory_name = "_".join(split("_*_", split("\\\\", memory)[-1][:-4]))
            cur_key_name = "_".join(split("_*_", split("\\\\", key)[-1][:-4]))
            Label(memory_key_table_window, text=cur_memory_name).grid(row=memory_key_table_window_row_counter,
                                                                      column=0,
                                                                      sticky=E)
            Label(memory_key_table_window, text=cur_key_name).grid(row=memory_key_table_window_row_counter,
                                                                   column=1,
                                                                   sticky=W)
            memory_key_table_window_row_counter += 1


    @staticmethod
    def update_StrVar(var, text):
        var.set(text)

    @staticmethod
    def update_Checkbar(bar, text):
        # Not implemented
        bar.set(text)

    def show_memory_checkbar_window(self, temp_input_path, run_check_memory_key_pairings=False):
        """
        Checkbar
        Not implemented yet
        """
        def ok_button_run(memory_checkbar):
            self.relevant_memory_name_pieces = memory_checkbar.state()
            self.kill_popup(memory_checkbar_window)

            if run_check_memory_key_pairings:
                self.check_memory_key_pairing()

        split_name_options = glob(temp_input_path + "\\*txt")[0]
        split_name_options = split("_*_", split('\\\\', split_name_options)[-1][:-4])
        memory_checkbar_window = Toplevel(self)
        memory_checkbar_window.title("Choose relevant parts of memory name for auto-matching with key files")
        memory_checkbar_window.geometry('600x50')
        memory_checkbar = Checkbar(memory_checkbar_window, split_name_options)
        memory_checkbar.pack()
        ok_button = Button(memory_checkbar_window, text="Match key files!",
                           command=lambda: ok_button_run(memory_checkbar))
        ok_button.pack()


    def choose_input_path(self):
        self.input_path.set(askdirectory(initialdir=self.input_path.get(), title="Choose input folder"))
        self.show_memory_checkbar_window(self.input_path.get(), run_check_memory_key_pairings=True)
        # Just to make picking a path faster potentially, make output the same as input
        if self.output_path.get() == '':
            self.output_path.set(self.input_path.get())


    def choose_output_path(self):
        self.output_path.set(askdirectory(initialdir=self.output_path.get(), title="Choose output folder"))
        if len(self.output_path.get()) > 0:
            self.output_path_set.set(True)
        else:
            self.output_path_set.set(False)

        # Just to make picking a path faster
        if self.input_path.get() == '':
            self.input_path.set(self.output_path.get())

    def load_classifier_variables(self):
        if len(self.memory_key_pairing_list) > 0:
            pass
        else:
            messagebox.showerror("Error!", "Incomplete or invalid parameters")
            return

        master_sheet_name = self.master_sheet_name_entry.get()
        accuracies_master_sheet_fullpath = self.output_path.get() + "\\" + master_sheet_name + '_accuracies.csv'
        statistics_master_sheet_fullpath = self.output_path.get() + "\\" + master_sheet_name + '_statistics.csv'

        number_of_simulations = int(self.number_of_simulations_entry.get())
        pre_stimulus_time = float(self.pre_stimulus_time_entry.get())
        post_stimulus_time = float(self.post_stimulus_time_entry.get())
        pre_stimulus_raster = float(self.pre_stimulus_raster_entry.get())
        post_stimulus_raster = float(self.post_stimulus_raster_entry.get())

        if self.all_cores.get() == 1:
            number_of_cores = cpu_count()
            # number_of_cores = 1  # only run in one core for now, until I figure out how to queue results correctly
        else:
            number_of_cores = cpu_count() // 2  # use half the number of cores if not all
            # number_of_cores = 1  # only run in one core for now, until I figure out how to queue results correctly

        classifier_methods = [item for idx, item in enumerate(self.classifier_methods) if
                              self.classifier_methods_checkbar.state()[idx] == 1]

        sigma_set = [item for idx, item in enumerate(self.sigma_set) if
                     self.sigma_set_checkbar.state()[idx] == 1]

        bin_set = [item for idx, item in enumerate(self.bin_set) if
                   self.bin_set_checkbar.state()[idx] == 1]

        # make input list
        input_list = list()
        for memory_name, key_name in self.memory_key_pairing_list:
            input_list.append([memory_name, key_name, self.output_path.get(),
                               accuracies_master_sheet_fullpath, statistics_master_sheet_fullpath,
                               number_of_simulations, pre_stimulus_time, post_stimulus_time,
                               pre_stimulus_raster, post_stimulus_raster, self.brain_area.get(), classifier_methods,
                               sigma_set, bin_set])

        self.run_cancel_popup = Toplevel(self)

        self.do_run_button = Button(self.run_cancel_popup, text="Click here to run",
                               command=lambda: self.activate_mp(input_list, number_of_cores))
        self.do_run_button.config(font=('helvetica', 10, 'bold'))
        self.do_run_button.pack()

        self.cancel_button = Button(self.run_cancel_popup, text="Abort", state=DISABLED,
                               command=self.cancel_button_clicked)
        self.cancel_button.config(font=('helvetica', 10, 'bold'))
        self.cancel_button.pack()


    def cancel_button_clicked(self):
        self.do_run_button.configure(state=NORMAL)
        self.cancel_button.configure(state=DISABLED)
        self.kill_pool()


    def kill_pool(self):
        self.pool.terminate()
        self.pool.join()


    def close_pool(self):
        self.pool.join()

    def compile_result_csv(self):
        master_sheet_name = self.master_sheet_name_entry.get()
        accuracies_master_sheet_fullpath = self.output_path.get() + "\\" + master_sheet_name + '_accuracies.csv'
        statistics_master_sheet_fullpath = self.output_path.get() + "\\" + master_sheet_name + '_statistics.csv'
        with open(accuracies_master_sheet_fullpath, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(['Memory'] + ['Key'] + ['Method'] + ['Stimulus'] + ['Accuracy'])
        with open(statistics_master_sheet_fullpath, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(['Memory'] + ['Key'] + ['Method'] + ['T.statistic'] + ['P.value'] + ['Cohen.d'] +
                            ['CI.significance'] + ['Sigma'] + ['Bin.size'])

        for cur_master_sheet_path in [accuracies_master_sheet_fullpath, statistics_master_sheet_fullpath]:
            process_files = glob(cur_master_sheet_path[:-4] + "_*.csv")
            df_merged = (read_csv(f, sep=',', header=None) for f in process_files)
            df_merged = concat(df_merged, ignore_index=True)
            df_merged.to_csv(cur_master_sheet_path, mode='a', header=False, index=False)

            [remove(f) for f in process_files]

    def check_pool_completion(self):
        if self.pool_map_result.ready():
            self.done_popup = Toplevel(self)
            self.done_button = Button(self.done_popup, text="Classifier completed!",
                                      command=lambda : self.kill_popup([self.done_popup, self.run_cancel_popup]))
            self.done_button.config(font=('helvetica', 20, 'bold'))
            self.done_button.pack()
            self.close_pool()
            self.compile_result_csv()
        else:
            self.after(500, self.check_pool_completion)


    def activate_mp(self, input_list, number_of_cores):
        self.do_run_button.configure(state=DISABLED)
        self.cancel_button.configure(state=NORMAL)

        self.pool = Pool(number_of_cores)

        self.pool_map_result = self.pool.map_async(self.run_pattern_classifier, input_list)
        self.pool.close()
        self.check_pool_completion()

    @staticmethod
    def run_pattern_classifier(input_list):
        """
        Main driver
        Grab inputs from list and run on separate cores
        """
        cur_memory_name, cur_key_name, cur_output_path, \
        accuracies_master_sheet_fullpath, statistics_master_sheet_fullpath, \
        number_of_simulations, pre_stimulus_time, post_stimulus_time, \
        pre_stimulus_raster, post_stimulus_raster, brain_area, classifier_methods, \
        sigma_set, bin_set = input_list

        output_name = "_".join(split("_*_", split("\\\\", cur_memory_name)[-1][:-4])) + '_with_' + \
                      "_".join(split("_*_", split("\\\\", cur_key_name)[-1][:-4]))

        with PdfPages(cur_output_path + "\\" + output_name + '.pdf') as pdf:
            for method in classifier_methods:
                try:
                    run_classification(memory_name=cur_memory_name,
                                       key_name=cur_key_name,
                                       method=method,
                                       output_path=cur_output_path,

                                       # fixed refractory period for rasterization purposes;
                                       # will skip spikes faster than this, but making it smaller will increase
                                       # computation time
                                       refractory_period=0.001,

                                       pre_stimulus_time=pre_stimulus_time,
                                       post_stimulus_time=post_stimulus_time,
                                       pre_stimulus_raster=pre_stimulus_raster,
                                       post_stimulus_raster=post_stimulus_raster,
                                       brain_area=brain_area,

                                       # Make these unique per process, then combine all after Pool is ready
                                       accuracies_master_sheet_fullpath=accuracies_master_sheet_fullpath[:-4] + "_" + current_process().name + ".csv",
                                       statistics_master_sheet_fullpath=statistics_master_sheet_fullpath[:-4] + current_process().name + ".csv",

                                       unit_name=output_name,
                                       number_of_simulations=number_of_simulations,
                                       count_bin_set=bin_set,
                                       sigma_set=sigma_set,
                                       file_name=output_name, pdf_handle=pdf)
                    # Force Python to collect garbage
                    collect()

                except Exception:
                    print("Error in running: " + cur_memory_name)
                    print("The following error occurred: ")
                    print(format_exc())
                    print("\n\n")

                    # Force Python to collect garbage
                    collect()

                    continue



    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            if self.pool is not None:
                self.kill_pool()
            self.destroy()


if __name__ == "__main__":
    root = GUI()
    root.mainloop()