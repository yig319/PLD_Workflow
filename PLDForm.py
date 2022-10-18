import shutil
import sys
import json
import datetime
import os, glob, h5py
import numpy as np
import matplotlib.pyplot as plt
from datafed.CommandLib import API

from PyQt5.QtGui import QFont 
from PyQt5.QtCore import * 
from PyQt5.QtWidgets import * 
from ManagePlume import remove_desktop_ini, pack_to_hdf5_and_upload, pack_to_hdf5, upload_to_datafed

class message_window(QWidget):
    def __init__(self, message):
        super().__init__()
        self.message = '\n    '+message
        self.initUI()
    def initUI(self):
        lblName = QLabel(self.message, self)


class GenerateForm(QWidget):
    '''
    This is the digital form to record PLD experiment parameters mainly, 
    with additional feature features to manage, pack, and upload recorded plumes along with recorded parameters. 

    :param version: input variable used to determine user want to open which version of digital form, defaults to "parameter" 
    :type version: str

    '''
    def __init__(self, version='parameter'):
        super().__init__()
        self.version = version
        
        # setting the minimum size
        width = 700
        height = 800
        self.setMinimumSize(width, height)
        self.window_height = 22
        self.button_height = 28
        self.current_page = 0
        
        # pre-define inputs
        # header part
        self.growth_id_input = QLineEdit()
        self.name_input = QLineEdit()
        self.date_input = QLineEdit(datetime.datetime.today().strftime("%m/%d/%Y"))
        self.time_input = QLineEdit(datetime.datetime.now().strftime("%H:%M:%S"))
        self.save_path_input = QLineEdit(os.getcwd()) 
#         self.save_path_input = QLineEdit('C:/Image/') 

        self.custom_key = QLineEdit()
        self.custom_key.setFixedSize(80, self.window_height)
        self.custom_value = QLineEdit()

        # chamber part
        self.chamber_ComboBox = QComboBox()  # creating combo box to select degree
        chamber_list = ["Laser 1A", "Laser 1C"]
        self.chamber_ComboBox.addItems(chamber_list)
        
        substrate_list = ["", "SrTiO3", "None"]
        self.substrate_1_ComboBox = QComboBox()
        self.substrate_1_ComboBox.addItems(substrate_list)
        self.substrate_2_ComboBox = QComboBox()
        self.substrate_2_ComboBox.addItems(substrate_list)
        self.substrate_3_ComboBox = QComboBox()
        self.substrate_3_ComboBox.addItems(substrate_list)
        self.substrate_4_ComboBox = QComboBox()
        self.substrate_4_ComboBox.addItems(substrate_list)
        
        
        self.base_pressure_input = QLineEdit()

        gas_list = ["Vacuum", "Oxygen", "Argon"]
        self.cool_down_gas = QComboBox()
        self.cool_down_gas.addItems(gas_list)

        # notes part
        self.notes_input = QPlainTextEdit()        
        self.notes_input.setFixedSize(600, self.window_height*3)
        
        
        # target
        self.target_input = [QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), 
                             QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()]
        
        # lens part
        self.aperture_input = [QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), 
                             QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()]
        self.focus_input = [QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), 
                             QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()]
        self.attenuator_input = [QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), 
                             QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()]
        self.target_height_input = [QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), 
                             QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()]
        
        # laser part
        self.laser_voltage_input = [QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), 
                             QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()]
        self.laser_energy_input = [QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), 
                             QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()]
        self.energy_mean_input = [QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), 
                             QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()]

        self.energy_std_input = [QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), 
                             QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()]

        
        # pre-ablation and ablation
        self.pre_temperature_input = [QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), 
                             QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()]
        self.pre_pressure_input = [QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), 
                             QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()]
        self.pre_gas_input = [QComboBox(), QComboBox(), QComboBox(), QComboBox(), QComboBox(), 
                              QComboBox(), QComboBox(), QComboBox(), QComboBox(), QComboBox()]   
        self.pre_frequency_input = [QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), 
                             QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()]
        self.pre_number_pulses_input = [QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), 
                             QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()]

        self.temperature_input = [QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), 
                             QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()]
        self.pressure_input = [QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), 
                             QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()]
        self.gas_input = [QComboBox(), QComboBox(), QComboBox(), QComboBox(), QComboBox(), 
                          QComboBox(), QComboBox(), QComboBox(), QComboBox(), QComboBox()]     
        self.frequency_input = [QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), 
                             QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()]
        self.number_pulses_input = [QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), 
                             QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()]
        
        gas_list = ["", "Vacuum", "Oxygen", "Argon"]
        for combobox in self.pre_gas_input:
            combobox.addItems(gas_list)
        for combobox in self.gas_input:
            combobox.addItems(gas_list)
        
        # define the layout
        self.setWindowTitle("PLD Growth Record")
        
        self.toplayout = QVBoxLayout()
        self.setLayout(self.toplayout)

        ## Create the grid layout - second-level
        self.layout = QGridLayout()
        self.toplayout.addLayout(self.layout)

        self.header_form = QGroupBox("Header")
        self.header_layout = self.create_header()
        self.header_form.setLayout(self.header_layout)
        self.layout.addWidget(self.header_form, 0, 0)
        
        self.chamber_form = QGroupBox("Chamber Parameters")
        self.chamber_layout = self.create_chamber()
        self.chamber_form.setLayout(self.chamber_layout)
        self.layout.addWidget(self.chamber_form, 0, 1)
        
        
        ## create QVForm - second level
        self.button_layout = QGridLayout()
        self.toplayout.addLayout(self.button_layout)

            ### Create and connect the combo box to switch between pages
        self.pageCombo = QComboBox()
        self.pageCombo.addItems(["Target_1", "Target_2", "Target_3", "Target_4", "Target_5", 
                                 "Target_6", "Target_7", "Target_8", "Target_9", "Target_10"]) 
        self.pageCombo.activated.connect(self.switchPage)
            ### Add the combo box and the stacked layout to the top-level layout
        self.button_layout.addWidget(self.pageCombo, 0, 0)

#         if self.version == 'plume':
#             self.button_create = QPushButton(self)
#             self.button_create.setText("Create Directory")
#             self.button_create.clicked.connect(lambda: self.create_folder())
#             self.button_layout.addWidget(self.button_create, 0, 1) 

        ## create QGridLayout - second level
        self.multiPages = QFormLayout()
        self.toplayout.addLayout(self.multiPages)
        
            ### Create the forms
        self.Stack = QStackedWidget (self)
        self.multiPages.addWidget(self.Stack)
        for i in range(10):
            self.stack = QWidget()
            self.Stack.addWidget (self.stack)
            stack_layout = self.stackUI(i)
            self.stack.setLayout(stack_layout)
        
        #  save button - second level
        self.button_save = QPushButton(self)
        self.button_save.setText("Save Parameters")
        self.button_save.clicked.connect(lambda: self.save())
        self.toplayout.addWidget(self.button_save)
        
        if self.version == 'plume':
        
            self.button_image = QPushButton(self)
            self.button_image.setText("Convert Video to Images (Not Functional)")
            self.button_image.clicked.connect(lambda: self.convert_to_image())
            self.toplayout.addWidget(self.button_image)  
            
            self.button_pack = QPushButton(self)
            self.button_pack.setText("Save to HDF5 and Upload")
            self.button_pack.clicked.connect(lambda: self.pack_to_hdf5_and_upload_with_popwindow(self.path, self.file_name, self.info_dict))
            self.toplayout.addWidget(self.button_pack)
        
        #  notes - second level
        self.form_notes = QGroupBox()
        self.notes_layout = QFormLayout()
        self.form_notes.setLayout(self.notes_layout)
        self.notes_layout.addRow(QLabel("Notes"), self.notes_input)
        self.toplayout.addWidget(self.form_notes)
 

    def create_header(self):

        '''
        This is a function to create header part of the form.

        '''

        header_layout = QFormLayout()
        header_layout.addRow(QLabel("Growth ID"), self.growth_id_input)
        header_layout.addRow(QLabel("User Name"), self.name_input)
        header_layout.addRow(QLabel("Date"), self.date_input)
        header_layout.addRow(QLabel("Time"), self.time_input)
        header_layout.addRow(QLabel("Directory"), self.save_path_input)
        header_layout.addRow(self.custom_key, self.custom_value)
        return header_layout

    def create_chamber(self):
        '''
        This is a function to create chamber part of the form.
        '''
        
        chamber_layout = QFormLayout()
        chamber_layout.addRow(QLabel("Chamber"), self.chamber_ComboBox)
        chamber_layout.addRow(QLabel("Substrates"), self.substrate_1_ComboBox)
        chamber_layout.addRow(QLabel("          "), self.substrate_2_ComboBox)
        chamber_layout.addRow(QLabel("          "), self.substrate_3_ComboBox)
        chamber_layout.addRow(QLabel("          "), self.substrate_4_ComboBox)
        chamber_layout.addRow(QLabel("Base Pressure (Torr)"), self.base_pressure_input)

        chamber_layout.addRow(QLabel("Cool Down Atmosphere"), self.cool_down_gas)
        return chamber_layout    
    
    
    def stackUI(self, create_index):
        '''
        This is a function to create stacking pages of the form.
        '''
        
        layout = QGridLayout()
        
        form_target = QGroupBox()
        layout_target = QFormLayout()
        form_target.setLayout(layout_target)
        layout.addWidget(form_target, 0, 0)
        layout_target.addRow(QLabel("Target:"), self.target_input[create_index])
        
#         form_button = QGroupBox()
#         layout_button = QFormLayout()
#         form_button.setLayout(layout_button)
#         layout.addWidget(form_button, 1, 0)
 
        form_lens = QGroupBox("Lens Parameters")
        layout_lens = QFormLayout()
        form_lens.setLayout(layout_lens)
        layout.addWidget(form_lens, 1, 0)
        layout_lens.addRow(QLabel("Aperture (mm)"), self.aperture_input[create_index])
        layout_lens.addRow(QLabel("Focus (mm)"), self.focus_input[create_index])
        layout_lens.addRow(QLabel("Attenuator (mm)"), self.attenuator_input[create_index])
        layout_lens.addRow(QLabel("Target Height (mm)"), self.target_height_input[create_index])
        
        form_laser = QGroupBox("Laser Parameters")
        layout_laser = QFormLayout()
        form_laser.setLayout(layout_laser)
        layout.addWidget(form_laser, 1, 1)
        layout_laser.addRow(QLabel("Laser Voltage (kV)"), self.laser_voltage_input[create_index])
        layout_laser.addRow(QLabel("Laser Energy (mJ)"), self.laser_energy_input[create_index])
        layout_laser.addRow(QLabel("Measured Energy Mean(mJ)"), self.energy_mean_input[create_index])
        layout_laser.addRow(QLabel("Measured Energy Std"), self.energy_std_input[create_index])


        form_pre = QGroupBox("Pre-ablation")
        layout_pre = QFormLayout()
        form_pre.setLayout(layout_pre)
        layout.addWidget(form_pre, 1, 2)
        layout_pre.addRow(QLabel("Temperature (\N{DEGREE SIGN}C)"), self.pre_temperature_input[create_index])
        layout_pre.addRow(QLabel("Pressure (mTorr)"), self.pre_pressure_input[create_index])
        layout_pre.addRow(QLabel("Atmosphere Gas"), self.pre_gas_input[create_index])
        layout_pre.addRow(QLabel("Frequency (Hz)"), self.pre_frequency_input[create_index])
        layout_pre.addRow(QLabel("Pulses"), self.pre_number_pulses_input[create_index])

        
        form_ablation = QGroupBox("Ablation")
        layout_ablation = QFormLayout()
        form_ablation.setLayout(layout_ablation)
        layout.addWidget(form_ablation, 1, 3)
        layout_ablation.addRow(QLabel("Temperature (\N{DEGREE SIGN}C)"), self.temperature_input[create_index])
        layout_ablation.addRow(QLabel("Pressure (mTorr)"), self.pressure_input[create_index])
        layout_ablation.addRow(QLabel("Atmosphere Gas"), self.gas_input[create_index])
        layout_ablation.addRow(QLabel("Frequency (Hz)"), self.frequency_input[create_index])
        layout_ablation.addRow(QLabel("Pulses"), self.number_pulses_input[create_index])
        
        if self.version == 'plume':
            self.button_move_pre = QPushButton(self)
    #         self.button_folder.setFixedSize(300, self.window_height)
            self.button_move_pre.setText("Move Videos To Pre-ablation Folder")
            self.button_move_pre.clicked.connect(lambda: self.move_to_folder(pre=True))
            layout.addWidget(self.button_move_pre, 2, 2)  
        
        
            self.button_move = QPushButton(self)
    #         self.button_folder.setFixedSize(300, self.window_height)
            self.button_move.setText("Move Videos To Ablation Folder")
            self.button_move.clicked.connect(lambda: self.move_to_folder(pre=False))
            layout.addWidget(self.button_move, 2, 3)  
        return layout

    def pack_to_hdf5_and_upload_with_popwindow(self, path, file_name, info_dict):
        self.show_message_window('Hdf5 file packed and uploaded to datafed!')
        pack_to_hdf5_and_upload(path, file_name, info_dict)
        
    def show_message_window(self, message):
        self.exPopup = message_window(message)
        self.exPopup.setGeometry(500, 500, 400, 100)
        self.exPopup.show()
    
    
    def convert_to_image(self):

        '''
        This is a function to convert raw plume file to readable images.
        '''
        print('This is button is not functional yet, please convert manually with README instruction.')
        self.show_message_window('This is button is not functional yet, please convert manually with README instruction.')
        

    def move_to_folder(self, pre):

        '''
        This is a function to move plumes in default folder of acquisition software to target folder.

        :param pre: if True, plume images will be moved to pre-ablation folder
        :type pre: bool

        '''

        self.path = self.save_path_input.text() + '/'
        
        id = self.growth_id_input.text()
        name = self.name_input.text()
        date = ''.join(self.date_input.text().split('/'))
        self.file_name = id + '_' + name + '_' + date

        if not os.path.isdir(self.path+self.file_name):
            os.mkdir(self.path+self.file_name)
        
        date_list = self.date_input.text().split('/')
        date_m = date_list[2]+'_'+date_list[0]+'_'+date_list[1]
        remove_desktop_ini(self.path)

        file_list = glob.glob(self.path + date_m + '/*')
        i = self.current_page
        if pre:
            dst = self.path+self.file_name+'/'+str(i+1)+'-'+self.target_input[i].text()+'_'+'Pre'
            print('Saving videos to pre-ablation folder...')
            
        else:
            dst = self.path+self.file_name+'/'+str(i+1)+'-'+self.target_input[i].text()
            print('Saving videos to ablation folder...')
            
        if not os.path.isdir(dst):
            os.mkdir(dst)
            remove_desktop_ini(dst) # remove desktop.ini file from all sub-directory

        for file in file_list:
            shutil.move(file, dst)
            
        print('Done!')
        self.show_message_window('Recorded videos moved to target folder!')


    def switchPage(self, i):
        '''
        This is a function to switch page index of stacking pages in the form.
        '''
        self.Stack.setCurrentIndex(i)
        self.current_page = i
    
    def get_info(self):
        '''
        This is a function to fetch parameters from form and make a dictionary.
        '''
        info_dict = { 'header':{            
                                "Growth ID": self.growth_id_input.text(),
                                "User Name": self.name_input.text(),
                                "Date": self.date_input.text(),
                                "time": self.time_input.text(),
                                "Path": self.save_path_input.text(),
                                self.custom_key.text(): self.custom_value.text(),

                                "Chamber": self.chamber_ComboBox.currentText(),
                                "Substrate_1": self.substrate_1_ComboBox.currentText(),
                                "Substrate_2": self.substrate_2_ComboBox.currentText(),
                                "Substrate_3": self.substrate_3_ComboBox.currentText(),
                                "Substrate_4": self.substrate_3_ComboBox.currentText(),
                                "Base Pressure (Torr)": self.base_pressure_input.text(),
                                "Cool Down Atmosphere": self.cool_down_gas.currentText(),
                                "Notes": self.notes_input.toPlainText()}
                                }
        
        for i in range(10):
            info_dict['target_'+str(i+1)] = {
                        "Target Material": self.target_input[i].text(),

                        "Aperture": self.aperture_input[i].text(),
                        "Focus(mm)": self.focus_input[i].text(),
                        "Attenuator(mm)": self.attenuator_input[i].text(),
                        "Target Height(mm)": self.target_height_input[i].text(),           

                        "Laser Voltage(kV)": self.laser_voltage_input[i].text(),
                        "Laser Energy(mJ)": self.laser_energy_input[i].text(),
                        "Measured Energy Mean(mJ)": self.energy_mean_input[i].text(),
                        "Measured Energy Std(mJ)": self.energy_std_input[i].text(),

                        "Pre-Ablation-Temperature(\N{DEGREE SIGN}C)": self.pre_temperature_input[i].text(),
                        "Pre-Ablation-Pressure(mTorr)": self.pre_pressure_input[i].text(),
                        "Pre-Ablation-Gas Atmosphere": self.pre_gas_input[i].currentText(),
                        "Pre-Ablation-Frequency(Hz)": self.pre_frequency_input[i].text(),
                        "Pre-Ablation-Pulses": self.pre_number_pulses_input[i].text(),           

                        "Ablation-Temperature(\N{DEGREE SIGN}C)": self.temperature_input[i].text(),
                        "Ablation-Pressure(mTorr)": self.pressure_input[i].text(),
                        "Ablation-Atmosphere Gas": self.gas_input[i].currentText(),
                        "Ablation-Frequency(Hz)": self.frequency_input[i].text(),
                        "Ablation-Pulses": self.number_pulses_input[i].text(),            
                                            }
        # clean the empty pairs
        for name in info_dict.keys():
            info_dict[name] = {k: v for k, v in info_dict[name].items() if v}
        info_dict = {k: v for k, v in info_dict.items() if v}
        
        return info_dict

    
    def save(self):
        '''
        This is a function to save parameter dictionary to json file.
        '''
        
        print('Saving dictionary...')
        self.path = self.save_path_input.text() + '/'
        
        id = self.growth_id_input.text()
        name = self.name_input.text()
        date = ''.join(self.date_input.text().split('/'))
        self.file_name = id + '_' + name + '_' + date
        
        self.info_dict = self.get_info()  
        
        # convert to float if possible
        for k in self.info_dict.keys():
            for kk in self.info_dict[k].keys():
                try: self.info_dict[k][kk] = float(self.info_dict[k][kk])
                except ValueError: pass
            
        with open(self.path + '/' + self.file_name + '.json', 'w') as file:
            json.dump(self.info_dict, file)     
        print('Done!')
        
        self.show_message_window('Parameters saved!')
        