from flask import Flask, render_template
import jyserver.Flask as jsf
import sys
import os
import time
import subprocess
import glob

sys.path.insert(1, '../Model-Training')
sys.path.insert(1, '../Update-Data')
sys.path.insert(1, '../Prediction')

app = Flask(__name__)


def start_stop_training(flag, process=None):
    if flag == 1:
        # Clear out the file
        open("./status.txt", "w").close()

        # Update ETF/PUTCALL/CBOE data
        return subprocess.Popen([sys.executable, "../Update-Data/YFinanceScraper.py"])

    if flag == 0:
        with open('./status.txt', 'a') as f:
            f.write("Training Canceled")
        process.terminate()  # Kill the process




@jsf.use(app)  # This is a module named jyserver that allows manipulation of DOM
class Jyserverapp:
    def __init__(self):  # Create an __init__ method
        # Training Variables
        self.m_name_input = ""
        self.models_to_test_select = ""
        self.num_epoch_select = ""
        self.learning_rate_select = ""
        self.lead_up_day_select = ""
        self.limit1_values_select = ""
        self.limit2_values_select = ""
        self.limit3_values_select = ""
        self.momentum_consideration_day_select = ""
        self.checkbox_putt_call = ""
        self.checkbox_junk_bond_demand = ""
        self.checkbox_mcclellan_summation_index = ""
        self.training_box_status = ""
        self.training_status_len = 0
        self.on_training_page_reload = ""
        self.training_loop = True
        self.training_status = False
        self.process = ''
    # -------- The methods below are for the Training page ---------
    # Updates text boxes with status and re-enables user input when 'Done' seen
    def training_status_box_update(self, t_status,
                                   m_name_input, train_button,
                                   models_to_test_select, num_epoch_select,
                                   learning_rate_select, lead_up_day_select,
                                   limit1_values_select, limit2_values_select,
                                   limit3_values_select,
                                   momentum_consideration_day_select,
                                   checkbox_putt_call,
                                   checkbox_junk_bond_demand,
                                   checkbox_mcclellan_summation_index):

        self.on_training_page_reload = ""
        self.training_box_status = ""
        self.training_status_len = 0
        self.training_loop = True
        status_filepath = "./status.txt"

        # Check to see if training is already in progress and populate the
        # status box (in-case the user refreshed the browser) and disable
        # user input, if not clear the status box
        if self.training_status:
            # Start Update, Data Creation
            self.process = start_stop_training(1)

            f = open(status_filepath, "r")
            lines = f.readlines()
            for i in range(len(lines)):
                self.on_training_page_reload += lines[i]
            # Re-Update the specified status through DOM manipulation
            self.js.document.getElementById(t_status).innerHTML \
                = self.on_training_page_reload
            f.close()

            # Disable user input on the training page, if a refreshed happened
            # during training
            self.disable_training_user_input(t_status,
                                             m_name_input, train_button,
                                             models_to_test_select, num_epoch_select,
                                             learning_rate_select, lead_up_day_select,
                                             limit1_values_select, limit2_values_select,
                                             limit3_values_select,
                                             momentum_consideration_day_select,
                                             checkbox_putt_call,
                                             checkbox_junk_bond_demand,
                                             checkbox_mcclellan_summation_index)

            # Repopulate Dataset Selection on page reload and options
            self.js.document.getElementById(models_to_test_select).value = self.models_to_test_select

            self.js.document.getElementById(num_epoch_select).value = self.num_epoch_select

            self.js.document.getElementById(learning_rate_select).value = self.learning_rate_select

            self.js.document.getElementById(lead_up_day_select).value = self.lead_up_day_select

            self.js.document.getElementById(limit1_values_select).value = self.limit1_values_select

            self.js.document.getElementById(limit2_values_select).value = self.limit2_values_select

            self.js.document.getElementById(limit3_values_select).value = self.limit3_values_select

            self.js.document.getElementById(momentum_consideration_day_select).value \
                = self.momentum_consideration_day_select

            self.js.document.getElementById(checkbox_putt_call).value = self.checkbox_putt_call

            self.js.document.getElementById(checkbox_junk_bond_demand).value = self.checkbox_junk_bond_demand

            self.js.document.getElementById(checkbox_mcclellan_summation_index).value \
                = self.checkbox_mcclellan_summation_index

            # Repopulate user defined Model Name on page reload
            self.js.document.getElementById(m_name_input).value \
                = self.m_name_input
        else:
            return None

        # This loop checks a given file for updates and puts contents to a
        # specific status text box on the webpage

        while self.training_loop:
            time.sleep(.00001)
            f = open(status_filepath, "r")
            lines = f.readlines()
            f.close()
            # If it's new information then append it to the status
            if len(lines) > self.training_status_len:
                for i in range(len(lines)):
                    # Check for file's change in number of lines
                    if i >= self.training_status_len:
                        self.training_box_status += lines[i]
                        self.training_status_len += 1
                    # Stop looping if Done is sent and reset everything
                    if lines[i] == 'Training Complete' or lines[i] == 'Training Canceled':
                        self.training_box_status += '\n.......End..........'
                        print("MATCH FOUND ________________")

                        # https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder
                        files = glob.glob('../Data/Built-Datasets/*')
                        for f in files:
                            os.remove(f)

                        # This deactivates the training complete line by
                        # adding a new line, and also adds a
                        # separator to the training status file to separate
                        # training sessions
                        open(status_filepath, "w").close()
                        self.training_loop = False
                        self.training_status = False
                        self.enable_training_user_input(t_status,
                                                        m_name_input, train_button,
                                                        models_to_test_select, num_epoch_select,
                                                        learning_rate_select, lead_up_day_select,
                                                        limit1_values_select,
                                                        limit2_values_select,
                                                        limit3_values_select,
                                                        momentum_consideration_day_select,
                                                        checkbox_putt_call,
                                                        checkbox_junk_bond_demand,
                                                        checkbox_mcclellan_summation_index)

                # Update the specified status through DOM manipulation
                self.js.document.getElementById(t_status).innerHTML \
                    = self.training_box_status

                # Make status scroll box scroll to bottom as it is updated
                self.js.document.getElementById(t_status).scrollTop = \
                    self.js.document.getElementById(t_status).scrollHeight

    # Store User's choices for training in a text file
    def store_user_training_choices(self, t_status,
                                    m_name_input, train_button,
                                    models_to_test_select, num_epoch_select,
                                    learning_rate_select, lead_up_day_select,
                                    limit1_values_select, limit2_values_select,
                                    limit3_values_select,
                                    momentum_consideration_day_select,
                                    checkbox_putt_call,
                                    checkbox_junk_bond_demand,
                                    checkbox_mcclellan_summation_index):

        # Grab the data from the webpage
        if self.js.document.getElementById(m_name_input).value != "":
            self.m_name_input = str(self.js.document.getElementById(m_name_input).value)
            self.models_to_test_select = str(self.js.document.getElementById(models_to_test_select).value)
            self.num_epoch_select = str(self.js.document.getElementById(num_epoch_select).value)
            self.learning_rate_select = str(self.js.document.getElementById(learning_rate_select).value)
            self.lead_up_day_select = str(self.js.document.getElementById(lead_up_day_select).value)
            self.limit1_values_select = str(self.js.document.getElementById(limit1_values_select).value)
            self.limit2_values_select = str(self.js.document.getElementById(limit2_values_select).value)
            self.limit3_values_select = str(self.js.document.getElementById(limit3_values_select).value)
            self.momentum_consideration_day_select = str(self.js.document.getElementById
                                                         (momentum_consideration_day_select).value)
            self.checkbox_putt_call = str(self.js.document.getElementById(checkbox_putt_call).value)
            self.checkbox_junk_bond_demand = str(self.js.document.getElementById(checkbox_junk_bond_demand).value)
            self.checkbox_mcclellan_summation_index = str(self.js.document.getElementById
                                                          (checkbox_mcclellan_summation_index).value)

            # User Selected options section
            # Putt/Call option
            if self.js.document.getElementById(checkbox_putt_call).value == "True":
                self.checkbox_putt_call = '1'
            else:
                self.checkbox_putt_call = '0'

            # Junk bond demand option
            if self.js.document.getElementById(checkbox_junk_bond_demand).value == "True":
                self.checkbox_junk_bond_demand = '1'
            else:
                self.checkbox_junk_bond_demand = '0'

            # Mcclellan Summation Index option
            if self.js.document.getElementById(checkbox_mcclellan_summation_index).value == "True":
                self.checkbox_mcclellan_summation_index = '1'
            else:
                self.checkbox_mcclellan_summation_index = '0'

        # Set path to the training config and status file
        training_config_filepath = "../Model-Training/training_config.txt"
        training_status_filepath = "./status.txt"

        # Label each choice
        model_name = "Model_Name: " + self.m_name_input
        models_to_test = "Models_to_Test: " + self.models_to_test_select
        num_epoch = "Number_of_Epochs: " + self.num_epoch_select
        learning_rate = "Base_Learning_Rate: " + self.learning_rate_select
        lead_up_day = "Lead_Up_Days: " + self.lead_up_day_select
        limit1 = "Limit1: " + self.limit1_values_select
        limit2 = "Limit2: " + self.limit2_values_select
        limit3 = "Limit3: " + self.limit3_values_select
        momentum = "Momentum_Consideration: " + self.momentum_consideration_day_select
        putt_call = "Putt_Call: " + self.checkbox_putt_call
        junk_bond = "Junk_Bond: " + self.checkbox_junk_bond_demand
        mcclellan_summation = "McClellan_Summation: " + self.checkbox_mcclellan_summation_index

        # Put user choices into a list
        lines_to_write = [model_name, models_to_test,
                          num_epoch, learning_rate,
                          lead_up_day, limit1, limit2,
                          limit3, momentum, putt_call,
                          junk_bond, mcclellan_summation]

        # Erase the files
        open(training_config_filepath, 'w').close()
        open(training_status_filepath, 'w').close()

        # Write the list of variables to
        f = open(training_config_filepath, 'a')
        for i in range(len(lines_to_write)):
            f.write(lines_to_write[i] + "\n")
        f.close()

        # After user input has been stored called the update method
        self.training_status = True
        self.training_status_box_update(t_status,
                                        m_name_input, train_button,
                                        models_to_test_select, num_epoch_select,
                                        learning_rate_select, lead_up_day_select,
                                        limit1_values_select,
                                        limit2_values_select,
                                        limit3_values_select,
                                        momentum_consideration_day_select,
                                        checkbox_putt_call,
                                        checkbox_junk_bond_demand,
                                        checkbox_mcclellan_summation_index)

    # Training Setters
    # Disables training user input and stores inputted user choices
    def disable_training_user_input(self, t_status,
                                    m_name_input, train_button,
                                    models_to_test_select, num_epoch_select,
                                    learning_rate_select, lead_up_day_select,
                                    limit1_values_select, limit2_values_select,
                                    limit3_values_select,
                                    momentum_consideration_day_select,
                                    checkbox_putt_call,
                                    checkbox_junk_bond_demand,
                                    checkbox_mcclellan_summation_index):

        self.js.document.getElementById(m_name_input).disabled = True
        self.js.document.getElementById(train_button).disabled = True
        self.js.document.getElementById(models_to_test_select).disabled = True
        self.js.document.getElementById(num_epoch_select).disabled = True
        self.js.document.getElementById(learning_rate_select).disabled = True
        self.js.document.getElementById(limit1_values_select).disabled = True
        self.js.document.getElementById(limit2_values_select).disabled = True
        self.js.document.getElementById(limit3_values_select).disabled = True
        self.js.document.getElementById(momentum_consideration_day_select).disabled = True
        self.js.document.getElementById(checkbox_putt_call).disabled = True
        self.js.document.getElementById(checkbox_junk_bond_demand).disabled = True
        self.js.document.getElementById(checkbox_mcclellan_summation_index).disabled = True
        self.js.document.getElementById(lead_up_day_select).disabled = True

    # Enables training user input
    def enable_training_user_input(self, t_status,
                                   m_name_input, train_button,
                                   models_to_test_select, num_epoch_select,
                                   learning_rate_select, lead_up_day_select,
                                   limit1_values_select, limit2_values_select,
                                   limit3_values_select,
                                   momentum_consideration_day_select,
                                   checkbox_putt_call,
                                   checkbox_junk_bond_demand,
                                   checkbox_mcclellan_summation_index):

        self.js.document.getElementById(m_name_input).disabled = False
        self.js.document.getElementById(train_button).disabled = False
        self.js.document.getElementById(models_to_test_select).disabled = False
        self.js.document.getElementById(num_epoch_select).disabled = False
        self.js.document.getElementById(learning_rate_select).disabled = False
        self.js.document.getElementById(limit1_values_select).disabled = False
        self.js.document.getElementById(limit2_values_select).disabled = False
        self.js.document.getElementById(limit3_values_select).disabled = False
        self.js.document.getElementById(momentum_consideration_day_select).disabled = False
        self.js.document.getElementById(checkbox_putt_call).disabled = False
        self.js.document.getElementById(checkbox_junk_bond_demand).disabled = False
        self.js.document.getElementById(checkbox_mcclellan_summation_index).disabled = False
        self.js.document.getElementById(lead_up_day_select).disabled = False

    def cancel_training(self, t_status):
        #print("IM HERE IN CANCEL TRAINING")
        start_stop_training(0, self.process) #kill the process running training

    def predict(self, report_select, p_status, report_image, predict_button, prediction_text, download_link, download_button):
        # get the value the user selected
        model_name = str(self.js.document.getElementById(report_select).value)

        # Run prediction code
        # https://stackoverflow.com/questions/43274476/is-there-a-way-to-check-if-a-subprocess-is-still-running
        self.js.document.getElementById(p_status).value = 'Loading Prediction...'
        self.js.document.getElementById(predict_button).disabled = True
        self.js.document.getElementById(report_select).disabled = True
        self.js.document.getElementById(download_button).disabled = False
        self.js.document.getElementById(predict_button).disabled = False
        self.js.document.getElementById(report_select).disabled = False
        self.js.document.getElementById(p_status).value = '...Load Complete'
        self.js.document.getElementById(report_image).src = "./static/predictions/" + \
                                                            model_name +\
                                                            '/' + model_name\
                                                            + '.png'

        prediction_text_filepath = "./static/predictions/" + \
                                                            model_name +\
                                                            '/' + model_name\
                                                            + '.txt'
        prediction_text_lines = ""
        f = open(prediction_text_filepath, "r")
        lines = f.readlines()
        f.close()
        # print("Prediction text filepath: ", prediction_text_filepath)
        for i in range(len(lines)):
            prediction_text_lines += lines[i]
        # print("Prediction text Lines: ", prediction_text_lines)

        # Re-Update the specified status through DOM manipulation
        self.js.document.getElementById(prediction_text).value \
            = prediction_text_lines


        # Update the drop down box when the predict button is pressed incase training has recently finished
        reports_list = os.listdir("./static/predictions/")
        self.js.document.getElementById(report_select).value = reports_list

        #Enable the download button and change the link to the proper zip file
        self.js.document.getElementById(download_link).setAttribute("href", "./static/predictions/"
                                                                    + model_name
                                                                    + "/"
                                                                    + model_name
                                                                    + '.zip')
        self.js.document.getElementById(download_link).setAttribute("download", model_name
                                                                    + '.zip')

    def prediction_update(self, report_select):
        reports_list = os.listdir("./static/predictions/")
        self.js.document.getElementById(report_select).value = reports_list

@app.route('/')
def home():
    return Jyserverapp.render(render_template("index.html"))


@app.route('/training')
def training():
    models_to_test_values = []
    lead_up_day_values = []
    momentum_consideration_day_values = []
    num_epoch_values = []
    learning_rate_values = []
    limit1_values = []
    limit2_values = []
    limit3_values = []

    for j in range(3, 46):
        # Generate list of number of models to test
        models_to_test_values.append(j)

    for i in range(3, 60):
        # Generate list of lag time values
        lead_up_day_values.append(i)

        # Generate list of days that will be considered for momentum calculations
        momentum_consideration_day_values.append(i)

    # Generate the values that will populate the drop down selector for number of epochs
    for w in range(5, 20000):
        num_epoch_values.append(w)

        # Generate the range of learning rates available for the selector
        learning_rate_values = [0.1, 0.01, 0.001, 0.0001]

    # Generate the lower limit values for types of networks to create
    for k in range(1, 16):
        limit1_values.append(k)

    # Generate the limit2 values for types of networks to create
    for n in range(16, 36):
        limit2_values.append(n)

    # Generate the upper limit values for types of networks to create
    for m in range(36, 61):
        limit3_values.append(m)

    return Jyserverapp.render(render_template("training.html",
                                              models_to_test_values
                                              =models_to_test_values,
                                              lead_up_day_values
                                              =lead_up_day_values,
                                              momentum_consideration_day_values
                                              =momentum_consideration_day_values,
                                              num_epoch_values
                                              =num_epoch_values,
                                              learning_rate_values
                                              =learning_rate_values,
                                              limit1_values
                                              =limit1_values,
                                              limit2_values
                                              =limit2_values,
                                              limit3_values
                                              =limit3_values))


@app.route('/prediction')
def prediction():  # put application's code here

    # Inspired by here Pulled from here: https://stackoverflow.com/questions/29206384/python-folder-names-in-the-directory
    reports_list = os.listdir("./static/predictions")
    return Jyserverapp.render(render_template("prediction.html",
                                              reports=reports_list))


if __name__ == '__main__':
    # Clear up old models that are not being used
    try:
        trained_model_files = glob.glob('../Model-Training/Trained-Models/*')
        for j in trained_model_files:
            os.remove(j)
    except:
        print("Could not delete old trained models")
    app.run()

