# Initial Network Testing
# Written by Nick Alvey

#####################################################################################################################
# Import Necessary Components
#####################################################################################################################

# For Data Parsing

import pandas as pd
import glob
import os
import numpy as np
from scipy.stats import linregress
import datetime
import matplotlib.pyplot as plt
import sys
from keras.callbacks import Callback
# For Neural Networks

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model

# Add access to the data folder
sys.path.insert(1, '../Data')

#####################################################################################################################

# Begin Initialization of Algorithm

#####################################################################################################################

# List CPU's Available
print("------------------------------------------------------------")
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
print("------------------------------------------------------------")
print("Run with Tensorflow Version: " + tf.__version__)
print("------------------------------------------------------------")

#####################################################################################################################

# Get CSV for Volatility and Momentum Calculations
# This can be over any number of ETF's, or just a single one

#####################################################################################################################

def CreateDataset(lead_up_days = 2, consideration_days = 90, putcallflag = 0,
                  junkbondflag = 0, mcclellanflag = 0, future_num_days = 10):
    # Number of days in sample to consider leading up to current day, lowest it should be is 2
    ETFpath = '../Data/3x-ETF/'
    N = lead_up_days
    """

    :param ETFpath:
    :param N:
    :param consideration_days:
    :param putcallflag:
    :param junkbondflag:
    :param mcclellanflag:
    :param future_num_days:
    :return: The compiled ETF dataset


    """
    Whole_ETF_3x = None

    # ETFpath = '../Data/3x-ETF/'  # Reference '3x-ETF-All-Files/' to run over all 3x ETF's

    # Run over each ETF desired
    ETFfiles = glob.glob(os.path.join(ETFpath, "*.csv"))

    # Flag to initialize Neural Network input
    first_run_flag = 0
    number_of_ETFfiles = len(ETFfiles)
    counter = 0

    print("Checking for dataset...")
    if os.path.exists("../Data/Built-Datasets/" + str(tfd['Model_Name'])):
        print_and_write_status("ETF_out.csv already exists, Loading dataset...")
        Whole_ETF_3x = pd.read_csv('../Data/Built-Datasets/ETF_out.csv')
    else:
        print_and_write_status("\n~No dataset found~")
        print_and_write_status("\n\nBuilding Dataset...")
        for ETF_csvs in ETFfiles:
            counter += 1
            print_and_write_status("...Dataset creation " + "{:.2f}".format(float((counter/number_of_ETFfiles)*100)) +
                                   "% complete")

            # Added to reference the name of the CSV file later in script
            path = ETF_csvs[:25]
            name = ETF_csvs[15:]

            # print("Path is: ", ETF_csvs[:25])
            # print("Name is: ", ETF_csvs[15:])

            #print('NAME IS AT TOP: ', name)
            # ETFfile = 'TQQQ.csv'
            ETF_3x = pd.read_csv(ETF_csvs)

        #####################################################################################################################

            # Price - Derived Calculations

        #####################################################################################################################

            # Calculating Historic Volatility from Raw 3x ETF Data

            # Theory Yang-Zhang (OHLC)
            # Sum of overnight volatility and weighted average of Rogers-Satchell volatility
            # NOTE: In the Paper: MEASURING HISTORICAL VOLATILITY, temp_overnight_vol and temp_openclose_vol if
            # implemented exactly would be 0 every time - it subtracts one value from its equal. Implementations of this
            # algorithm do not do that, so I did not either.

            # Number of days in sample to consider leading up to current day, lowest it should be is 2
            # N = 2

            # Scaling factor
            F = 1

            k = 0.34 / (1.34 + (N + 1) / (N - 1))

            yang_zhang_vol_list = [np.nan] * N

            for day in range(N, len(ETF_3x)):
                temp_rog_satch_vol = 0  # initialize to zero for every consideration
                temp_overnight_vol = 0  # initialize to zero for every consideration
                temp_openclose_vol = 0  # initialize to zero for every consideration
                for i in range(N):
                    temp_rog_satch_vol += np.log(ETF_3x['High'][day - i] / ETF_3x['Close'][day - i]) * np.log(
                        ETF_3x['High'][day - i] / ETF_3x['Open'][day - i]) + np.log(
                        ETF_3x['Low'][day - i] / ETF_3x['Close'][day - i]) * np.log(
                        ETF_3x['Low'][day - i] / ETF_3x['Open'][day - i])
                    temp_overnight_vol += (np.log(ETF_3x['Open'][day - i] / ETF_3x['Close'][day - np.absolute((i - 1))])) ** 2
                    temp_openclose_vol += (np.log(ETF_3x['Close'][day - i] / ETF_3x['Open'][day - i])) ** 2
                rog_satch_vol = temp_rog_satch_vol * F / N
                overnight_vol = temp_overnight_vol * F / (N - 1)
                openclose_vol = temp_openclose_vol * F / (N - 1)

                yang_zhang_vol = np.sqrt(F) * np.sqrt(overnight_vol + k * openclose_vol + (1 - k) * rog_satch_vol)

                yang_zhang_vol_list.append(yang_zhang_vol)

            ETF_3x['Volatility'] = yang_zhang_vol_list

            # print(name + " Volatility Calculated and Appended to Dataframe")
            # print("------------------------------------------------------------")

        #####################################################################################################################

            # Calculating Momentum from Raw 3x ETF Data

            # Used from Tutorial here: https://teddykoker.com/2019/05/momentum-strategy-from-stocks-on-the-move-in-python/
            # Theory from Andreas Clenow

            # Number of days to calculate momentum over - it is not recommended to go below 30, and
            # 90 is recommended by link. This number changes the output drastically.

            if len(ETF_3x) < consideration_days:
                consideration_days = len(ETF_3x)
            else:
                # consideration_days = 90
                pass

            momentum_list = [np.nan] * consideration_days

            for days in range(consideration_days, len(ETF_3x)):
                consideration_days_list = []
                for datapoint in range(consideration_days):
                    consideration_days_list.append(ETF_3x['Close'][days - datapoint])
                returns = np.log(consideration_days_list)
                x = np.arange(len(consideration_days_list))
                slope, _, rvalue, _, _ = linregress(x, returns)

                momentum = (1 + slope) ** 252 * (rvalue ** 2)

                momentum_list.append(momentum)

            ETF_3x['Momentum'] = momentum_list

            # print(name + " " + str(consideration_days) + "-day Momentum Calculated and Appended to Dataframe")
            # print("------------------------------------------------------------")

        #####################################################################################################################

            # Sentimental Factors

        #####################################################################################################################

            # Compiling Put/Call Ratios
            if putcallflag == 0:
                PutCallpath = '../Data/Put-Call-Ratio/'
                PutCallfile = 'totalpc.csv'

                PutCall_rawdata = pd.read_csv(PutCallpath + PutCallfile)

                ETF_3x['Put/Call Ratio'] = np.nan

                for days in range(len(ETF_3x)):
                    etf_datetime = datetime.datetime.strptime(ETF_3x['Date'][days], "%Y-%m-%d")
                    putcall_index = PutCall_rawdata[PutCall_rawdata['date'] == etf_datetime.strftime('%Y-%m-%d')].index.values
                    if len(PutCall_rawdata.iloc[putcall_index, PutCall_rawdata.columns.get_loc('p_c_ratio')]) == 0:
                        ETF_3x.iloc[days, ETF_3x.columns.get_loc('Put/Call Ratio')] = np.nan
                    else:
                        ETF_3x.iloc[days, ETF_3x.columns.get_loc('Put/Call Ratio')] = \
                            PutCall_rawdata.iloc[putcall_index, PutCall_rawdata.columns.get_loc('p_c_ratio')]

                # print(name + " Put/Call Ratio Compiled and Appended to Dataframe")
                # print("------------------------------------------------------------")
            else:
                pass

        #####################################################################################################################

            # Calculating Junk Bond Demand (Volume)

            # Indicator for down days or up days not implemented
            if junkbondflag == 0:
                JunkBondpath = '../Data/Junk-Bond-ETF/'  # Should be updated to eventual folder name
                JunkBondfiles = glob.glob(os.path.join(JunkBondpath, "*.csv"))

                frames = []
                for junk_bonds_csvs in JunkBondfiles:
                    temp_JunkBond = pd.read_csv(junk_bonds_csvs)
                    frames.append(temp_JunkBond["Date"])
                    frames.append(temp_JunkBond["Volume"])

                headers = ["Date1", "Volume1", "Date2", "Volume2", "Date3", "Volume3", "Date4", "Volume4", "Date5",
                           "Volume5"]  # hardcoded since requirement is "Top 5 ETF's to be Considered"
                JunkBond_rawdata = pd.concat(frames, axis=1, keys=headers)
                JunkBond_rawdata['Total Volume'] = np.nan

                junkbond_datetime_start = datetime.date(2007, 1, 1)
                junkbond_datetime_end = datetime.date.today()

                days = junkbond_datetime_start
                listpair = []

                while days < junkbond_datetime_end:
                    daily_volume = 0

                    if len(JunkBond_rawdata[JunkBond_rawdata['Date1'] == days.strftime('%Y-%m-%d')].index.tolist()) > 0:
                        index = JunkBond_rawdata[JunkBond_rawdata['Date1'] == days.strftime('%Y-%m-%d')].index.tolist()
                        daily_volume += JunkBond_rawdata['Volume1'][index[0]]
                    if len(JunkBond_rawdata[JunkBond_rawdata['Date2'] == days.strftime('%Y-%m-%d')].index.tolist()) > 0:
                        index = JunkBond_rawdata[JunkBond_rawdata['Date2'] == days.strftime('%Y-%m-%d')].index.tolist()
                        daily_volume += JunkBond_rawdata['Volume2'][index[0]]
                    if len(JunkBond_rawdata[JunkBond_rawdata['Date3'] == days.strftime('%Y-%m-%d')].index.tolist()) > 0:
                        index = JunkBond_rawdata[JunkBond_rawdata['Date3'] == days.strftime('%Y-%m-%d')].index.tolist()
                        daily_volume += JunkBond_rawdata['Volume3'][index[0]]
                    if len(JunkBond_rawdata[JunkBond_rawdata['Date4'] == days.strftime('%Y-%m-%d')].index.tolist()) > 0:
                        index = JunkBond_rawdata[JunkBond_rawdata['Date4'] == days.strftime('%Y-%m-%d')].index.tolist()
                        daily_volume += JunkBond_rawdata['Volume4'][index[0]]
                    if len(JunkBond_rawdata[JunkBond_rawdata['Date5'] == days.strftime('%Y-%m-%d')].index.tolist()) > 0:
                        index = JunkBond_rawdata[JunkBond_rawdata['Date5'] == days.strftime('%Y-%m-%d')].index.tolist()
                        daily_volume += JunkBond_rawdata['Volume5'][index[0]]

                    listpair.append([days.strftime('%Y-%m-%d'), daily_volume])

                    days = days + datetime.timedelta(days=1)

                JunkBond_interpreted_data = pd.DataFrame(listpair, columns=['Date', 'Volume'])

                ETF_3x['Junk Bond Demand'] = np.nan

                for days in range(len(ETF_3x)):
                    etf_datetime = datetime.datetime.strptime(ETF_3x['Date'][days], "%Y-%m-%d")

                    junkbond_index = JunkBond_interpreted_data[
                        JunkBond_interpreted_data['Date'] == etf_datetime.strftime('%Y-%m-%d')].index.values

                    ETF_3x.iloc[days, ETF_3x.columns.get_loc('Junk Bond Demand')] = \
                        JunkBond_interpreted_data.iloc[junkbond_index, JunkBond_interpreted_data.columns.get_loc('Volume')]

                # print(name + " Junk Bond Demand Calculated and Appended to Dataframe")
                # print("------------------------------------------------------------")
            else:
                pass

        #####################################################################################################################

            # Calculating McClellan Summation Index on stock in question

            # Sources:
            # https://www.investopedia.com/terms/m/mcclellansummation.asp
            # https://www.investopedia.com/terms/m/mcclellanoscillator.asp
            # https://www.investopedia.com/terms/e/ema.asp

            # Initialize temporary variables
            if mcclellanflag == 0:
                advance = []
                EMA19 = []
                EMA39 = []
                SMA19 = 0
                SMA39 = 0

                MCsummation_index = 0
                ETF_3x['McClellan Summation Index'] = np.nan

                # Requires Date to be structured Old --> New
                # Adjusted for Size (normalized per Adjusted Oscillator formula)

                for index in range(len(ETF_3x)):

                    # Calculate Advances and Declines for each day
                    advance.append((ETF_3x['Close'][index] - ETF_3x['Open'][index]) /
                                   (ETF_3x['Close'][index] + ETF_3x['Open'][index]))

                    if index == 17:
                        SMA19 = np.average(advance[0:index])

                    elif index == 18:
                        EMA19.append((advance[index]) - SMA19 * 0.1 + SMA19)

                    elif index > 18:
                        EMA19.append((advance[index]) - EMA19[index - 19] * 0.1 + EMA19[index - 19])

                        if index == 37:
                            SMA39 = np.average(advance[0:index])

                        elif index == 38:
                            EMA39.append((advance[index]) - SMA39 * 0.05 + SMA19)

                        elif index > 38:
                            EMA39.append((advance[index]) - EMA39[index - 39] * 0.05 + EMA39[index - 39])

                            # Convert to McClellan Oscillator

                            adjusted_MCOscillator = EMA19[index - 19] - EMA39[index - 39]

                            # Convert to McClellan Summation Index

                            MCsummation_index = MCsummation_index + adjusted_MCOscillator

                            ETF_3x.iloc[index, ETF_3x.columns.get_loc('McClellan Summation Index')] = MCsummation_index

                # print(name + " McClellan Summation Index Calculated and Appended to Dataframe")
                # print("------------------------------------------------------------")
            else:
                pass

        #####################################################################################################################

            # Add in Profit for Traded Data via QuantConnect Backtesting of Provided Algorithm

        #####################################################################################################################

            Quantpath = '../Data/3X-ETF-Backtests-CSharp/'

            Quantfile = name
            #print("Name is: ",name)
            Quant_rawdata = None

            # Note, for now, columns are off by one, so "Date" here equates to the column of "Value"

            skip_flag = 0
            try:
                Quant_rawdata = pd.read_csv(Quantpath + name)
            except:
                print("ALERT: Profit Data not Formatted in the Same Way as Rest of Profit Data, Please Check")
                print("This ETF will be Ignored and Removed from Dataset")
                print("------------------------------------------------------------")
                skip_flag = 1

            if skip_flag == 0:
                Quant_dates = Quant_rawdata['date'].tolist()
                ETF_3x['Profit Percentage'] = np.nan

                start = 0  # Assuming starting money amount is $100,000, should be entered by user
                temp_percent = 0

                for days in range(len(ETF_3x)):
                    flag = 0
                    temp = 0
                    for dates in Quant_dates:
                        if ETF_3x['Date'][days] == dates:
                            fill_loc = Quant_rawdata.index[Quant_rawdata['date'] == dates].tolist()
                            if len(fill_loc) > 1:
                                fill_loc = fill_loc[1]
                            if float(Quant_rawdata['fill-quantity'][fill_loc]) != 0:
                                temp += float(Quant_rawdata['fill-quantity'][fill_loc])

                    if start == 0:
                        temp_percent = 1

                    if temp == 0:
                        pass

                    else:
                        if start == 0:
                            temp_percent = 1
                        else:
                            temp_percent = temp / start * 100

                    start += temp

                    # do temp_percent for percentage change, start for total amount in inventory
                    ETF_3x.iloc[days, ETF_3x.columns.get_loc('Profit Percentage')] = temp_percent

                #print(name + " Profit Percentage by Trading Algorithm Compiled and Appended to Dataframe")
                #print("------------------------------------------------------------")
            else:
                ETF_3x['Profit Percentage'] = np.nan

        #####################################################################################################################

            # Calculate Future Difference in Closing Price

        #####################################################################################################################

            # future_num_days = 15  # starting value, can be changed to any number
            #
            #
            # ETF_3x['Closing Price Difference ' + str(future_num_days) + ' days from now'] = np.nan
            #
            # for days in range(len(ETF_3x) - (future_num_days + 1)):
            #     ETF_3x.iloc[days, ETF_3x.columns.get_loc('Closing Price Difference ' + str(future_num_days) + ' days from now')] = 0
            #     for n in range(future_num_days):
            #             ETF_3x.iloc[days, ETF_3x.columns.get_loc('Closing Price Difference ' + str(future_num_days) + ' days from now')] += \
            #                 ETF_3x.iloc[days + n + 1, ETF_3x.columns.get_loc('Close')]
            #
            # # print(name + " " + str(future_num_days) + "-day Delta in Close Price Compiled and Appended to Dataframe")
            # # print("------------------------------------------------------------")

        #####################################################################################################################

        #####################################################################################################################

            # Add Time Lag Variables

        #####################################################################################################################

            # num_days = 30  # starting value, can be changed to any number
            #
            # delta_between = 1  # starting value, can be changed to any number
            #
            # for n in range(num_days // delta_between):  # must be whole number
            #     ETF_3x['High Time Lag ' + str(n)] = np.nan
            #
            #     for days in range(len(ETF_3x)):
            #         if days >= n * delta_between:
            #             ETF_3x.iloc[days, ETF_3x.columns.get_loc('High Time Lag ' + str(n))] = \
            #                 ETF_3x.iloc[days - n * delta_between, ETF_3x.columns.get_loc('High')]
            #
            # print(name + " " + str(num_days) + "-day High Price Lag with " + str(delta_between) + "-day Spacing Compiled and "
            #                                                                                       "Appended to Dataframe")
            # print("------------------------------------------------------------")

        #####################################################################################################################

            # num_days = 30  # starting value, can be changed to any number
            #
            # delta_between = 1  # starting value, can be changed to any number
            #
            # for n in range(num_days // delta_between):  # must be whole number
            #     ETF_3x['Low Time Lag ' + str(n)] = np.nan
            #
            #     for days in range(len(ETF_3x)):
            #         if days >= n * delta_between:
            #             ETF_3x.iloc[days, ETF_3x.columns.get_loc('Low Time Lag ' + str(n))] = \
            #                 ETF_3x.iloc[days - n * delta_between, ETF_3x.columns.get_loc('Low')]
            #
            # print(name + " " + str(num_days) + "-day Low Price Lag with " + str(delta_between) + "-day Spacing Compiled and "
            #                                                                                       "Appended to Dataframe")
            # print("------------------------------------------------------------")

        #####################################################################################################################

            # num_days = 30  # starting value, can be changed to any number
            #
            # delta_between = 1  # starting value, can be changed to any number
            #
            # for n in range(num_days // delta_between):  # must be whole number
            #     ETF_3x['Open Time Lag ' + str(n)] = np.nan
            #
            #     for days in range(len(ETF_3x)):
            #         if days >= n * delta_between:
            #             ETF_3x.iloc[days, ETF_3x.columns.get_loc('Open Time Lag ' + str(n))] = \
            #                 ETF_3x.iloc[days - n * delta_between, ETF_3x.columns.get_loc('Open')]
            #
            # print(name + " " + str(num_days) + "-day Open Price Lag with " + str(delta_between) + "-day Spacing Compiled and"
            #                                                                                      " Appended to Dataframe")
            # print("------------------------------------------------------------")

        #####################################################################################################################

            # num_days = 30  # starting value, can be changed to any number
            #
            # delta_between = 1  # starting value, can be changed to any number
            #
            # for n in range(num_days // delta_between):  # must be whole number
            #     ETF_3x['Close Time Lag ' + str(n)] = np.nan
            #
            #     for days in range(len(ETF_3x)):
            #         if days >= n * delta_between:
            #             ETF_3x.iloc[days, ETF_3x.columns.get_loc('Close Time Lag ' + str(n))] = \
            #                 ETF_3x.iloc[days - n * delta_between, ETF_3x.columns.get_loc('Volatility')]
            #
            # print(name + " " + str(num_days) + "-day Close Price Lag with " + str(delta_between) + "-day Spacing Compiled "
            #                                                                                        "and Appended to Dataframe")
            # print("------------------------------------------------------------")

        #####################################################################################################################

            # num_days = 30  # starting value, can be changed to any number
            #
            # delta_between = 1  # starting value, can be changed to any number
            #
            # for n in range(num_days // delta_between):  # must be whole number
            #     ETF_3x['Volume Time Lag ' + str(n)] = np.nan
            #
            #     for days in range(len(ETF_3x)):
            #         if days >= n * delta_between:
            #             ETF_3x.iloc[days, ETF_3x.columns.get_loc('Volume Time Lag ' + str(n))] = \
            #                 ETF_3x.iloc[days - n * delta_between, ETF_3x.columns.get_loc('Volume')]
            #
            # print(name + " " + str(num_days) + "-day Volume Lag with " + str(delta_between) + "-day Spacing Compiled and "
            #                                                                                   "Appended to Dataframe")
            # print("------------------------------------------------------------")

        #####################################################################################################################

            num_days = 10  # starting value, can be changed to any number

            delta_between = 1  # starting value, can be changed to any number

            for n in range(num_days // delta_between):  # must be whole number
                ETF_3x['Volatility Time Lag ' + str(n)] = np.nan

                for days in range(len(ETF_3x)):
                    if days >= n * delta_between:
                        ETF_3x.iloc[days, ETF_3x.columns.get_loc('Volatility Time Lag ' + str(n))] = \
                            ETF_3x.iloc[days - n * delta_between, ETF_3x.columns.get_loc('Volatility')]

            # print(name + " " + str(num_days) + "-day Volatility Lag with " + str(delta_between) + "-day Spacing Compiled and "
                                                                                                 # "Appended to Dataframe")
            # print("------------------------------------------------------------")

        #####################################################################################################################

            # for n in range(num_days // delta_between):  # must be whole number
            #     ETF_3x['Momentum Time Lag ' + str(n)] = np.nan
            #
            #     for days in range(len(ETF_3x)):
            #         if days >= n * delta_between:
            #             ETF_3x.iloc[days, ETF_3x.columns.get_loc('Momentum Time Lag ' + str(n))] = \
            #                 ETF_3x.iloc[days - n * delta_between, ETF_3x.columns.get_loc('Momentum')]
            #
            # print(name + " " + str(num_days) + "-day Momentum Lag with " + str(delta_between) + "-day Spacing Compiled and "
            #                                                                                     "Appended to Dataframe")
            # print("------------------------------------------------------------")

        #####################################################################################################################

            # for n in range(num_days // delta_between):  # must be whole number
            #     ETF_3x['Put/Call Time Lag ' + str(n)] = np.nan
            #
            #     for days in range(len(ETF_3x)):
            #         if days >= n * delta_between:
            #             ETF_3x.iloc[days, ETF_3x.columns.get_loc('Put/Call Time Lag ' + str(n))] = \
            #                 ETF_3x.iloc[days - n * delta_between, ETF_3x.columns.get_loc('Put/Call Ratio')]
            #
            # print(name + " " + str(num_days) + "-day Put/Call Lag with " + str(delta_between) + "-day Spacing Compiled and "
            #                                                                                     "Appended to Dataframe")
            # print("------------------------------------------------------------")

        #####################################################################################################################

            # for n in range(num_days // delta_between):  # must be whole number
            #     ETF_3x['Junk Bond Demand Time Lag ' + str(n)] = np.nan
            #
            #     for days in range(len(ETF_3x)):
            #         if days >= n * delta_between:
            #             ETF_3x.iloc[days, ETF_3x.columns.get_loc('Junk Bond Demand Time Lag ' + str(n))] = \
            #                 ETF_3x.iloc[days - n * delta_between, ETF_3x.columns.get_loc('Junk Bond Demand')]
            #
            # print(name + " " + str(num_days) + "-day Junk Bond Demand Lag with " + str(delta_between) +
            #       "-day Spacing Compiled and Appended to Dataframe")
            # print("------------------------------------------------------------")

        #####################################################################################################################

            # for n in range(num_days // delta_between):  # must be whole number
            #     ETF_3x['McClellan Summation Index Time Lag ' + str(n)] = np.nan
            #
            #     for days in range(len(ETF_3x)):
            #         if days >= n * delta_between:
            #             ETF_3x.iloc[days, ETF_3x.columns.get_loc('McClellan Summation Index Time Lag ' + str(n))] = \
            #                 ETF_3x.iloc[days - n * delta_between, ETF_3x.columns.get_loc('McClellan Summation Index')]
            #
            # print(name + " " + str(num_days) + "-day McClellan Summation Index Lag with " + str(delta_between) +
            #       "-day Spacing Compiled and Appended to Dataframe")
            # print("------------------------------------------------------------")

        #####################################################################################################################

            # for n in range(num_days // delta_between):  # must be whole number
            #     ETF_3x['Profit Percentage Time Lag ' + str(n)] = np.nan
            #
            #     for days in range(len(ETF_3x)):
            #         if days >= n * delta_between:
            #             ETF_3x.iloc[days, ETF_3x.columns.get_loc('Profit Percentage Time Lag ' + str(n))] = \
            #                 ETF_3x.iloc[days - n * delta_between, ETF_3x.columns.get_loc('Profit Percentage')]
            #
            # print(name + " " + str(num_days) + "-day Profit Percentage Lag with " + str(delta_between) +
            #       "-day Spacing Compiled and Appended to Dataframe")
            # print("------------------------------------------------------------")

        #####################################################################################################################

            # Some Last Cleanup of the Dataframe

        #####################################################################################################################

            # If desired, Turn Date into Separate Column for Year, Month, and Date

            # ETF_3x['Year'] = np.nan
            # ETF_3x['Month'] = np.nan
            # ETF_3x['Day'] = np.nan

            # for days in range(len(ETF_3x)):
            #     etf_datetime = datetime.datetime.strptime(ETF_3x['Date'][days], "%Y-%m-%d")
            #     ETF_3x.iloc[days, ETF_3x.columns.get_loc('Year')] = etf_datetime.strftime('%Y')
            #     ETF_3x.iloc[days, ETF_3x.columns.get_loc('Month')] = etf_datetime.strftime('%m')
            #     ETF_3x.iloc[days, ETF_3x.columns.get_loc('Day')] = etf_datetime.strftime('%d')

        #####################################################################################################################

            # Drop Data that isn't valuable, for now that's just the "Data" Column because it's not convertible to a float
            # If it is desired, uncomment the section above

            ETF_3x = ETF_3x.drop(columns=['Date'])

        #####################################################################################################################

            # Removes all Nan values
            ETF_3x = ETF_3x.dropna()
            ETF_3x = ETF_3x.reset_index(drop=True)

            # if skip_flag == 1:
                # print("This ETF is being removed from the Dataset")
                # print("------------------------------------------------------------")
            #else:
                # print("The ETF Dataset for " + name + " has been Finalized and is Being Added to the Overall Data Set")
                # print("------------------------------------------------------------")

            if first_run_flag == 0:
                Whole_ETF_3x = ETF_3x
                first_run_flag = 1
            else:
                Whole_ETF_3x = pd.concat([Whole_ETF_3x, ETF_3x], ignore_index=True)

        #####################################################################################################################

            # Some Troubleshooting Prints - Printing out the shape of the Dataframe, as well as outputting it to a CSV
            # To view potential errors
        #
            # print("------------------------------------------------------------")
            # print("SHAPE IS: ")
            # print(Whole_ETF_3x.shape)
            # print("------------------------------------------------------------")
        print_and_write_status("Saving dataset...")
        Whole_ETF_3x.to_csv('../Data/Built-Datasets/' + str(tfd['Model_Name']))

    return Whole_ETF_3x

# future_num_days = 15  # starting value, can be changed to any number

#####################################################################################################################

# Preparing to Run Neural Network

#####################################################################################################################

def CreateNeuralNetwork(Whole_ETF_3x, models_to_test = 5, lim1 = 15, lim2 = 35,
                        lim3 = 60, base_epochs = 1, base_learning_rate = 0.1):

    """

    :param Whole_ETF_3x: Pandas Dataframe of the dataset
    :param models_to_test: Number of Neural Networks to ensemble together
    :param lim1: variable to change architecture of Neural Network
    :param lim2: variable to change architecture of Neural Network
    :param lim3: variable to change architecture of Neural Network
    :param base_epochs: base number of epochs for the neural networks
    :param base_learning_rate: base learning rate of neural network
    :return: path to saved Neural Networks


    """

    print_and_write_status("Data Set Preparing for Neural Network Testing")
    print_and_write_status("------------------------------------------------------------")

    # Separating Training Data from Test Data

    Train_X = Whole_ETF_3x.sample(frac=0.90, random_state=0)
    Test_X = Whole_ETF_3x.drop(Train_X.index)

    train_features = Train_X.copy()
    test_features = Test_X.copy()

    train_labels = train_features.pop('Profit Percentage')
    test_labels = test_features.pop('Profit Percentage')

    train_features = np.asarray(train_features).astype(float)
    test_features = np.asarray(test_features).astype(float)

    train_labels = np.asarray(train_labels).astype(float)
    test_labels = np.asarray(test_labels).astype(float)

    # Normalize for input layer of Neural Network

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))

    print_and_write_status("The Data has been prepared, the Neural Network is being Created")
    print_and_write_status("------------------------------------------------------------")

    #####################################################################################################################

    # Initialize Model(s)

    #####################################################################################################################

    # Input the number of Neural Networks to Ensemble together, initialize Test Model

    # models_to_test = 3
    test_model = None

    #####################################################################################################################

    # Begin Running Training for each model

    # Some initial details:

    #       - This is a feed-forward Neural Network with an Adam Loss Function
    #       - This neural network has 3 layers, each of which is different based on which number NN it is in the ensemble
    #       - The learning rate is changed based on which NN it is in the ensemble
    #       - The number of epochs is changed based on which NN it is in the ensemble
    #       - There is a slight learning decay which can be tuned if desired
    #       - It uses mean squared error to more aggressively deal with errors

    #####################################################################################################################

    for model_num in range(models_to_test):

        # If statement to run different kinds of Neural Networks - in this configuration, after NN #15, it will
        # Change from a 3 Hidden-Layer Neural Network to a 2 Hidden-Layer Network. This is an attempt to add
        # Variation to the Networks to help boost the outcome when they are averaged together.

        if model_num < lim1:

            test_model = keras.Sequential([
                normalizer,
                layers.Dense(model_num + 10, activation='relu'),
                layers.Dense(model_num + 10, activation='relu'),
                layers.Dense(model_num + 10, activation='relu'),
                layers.Dense(1)
            ])

        elif model_num < lim2:

            test_model = keras.Sequential([
                normalizer,
                layers.Dense(model_num, activation='relu'),
                layers.Dense(model_num, activation='relu'),
                layers.Dense(1)
            ])

        elif model_num < lim3:

            test_model = keras.Sequential([
                normalizer,
                layers.Dense(model_num - 5, activation='relu'),
                layers.Dense(model_num - 5, activation='relu'),
                layers.Dense(model_num - 5, activation='relu'),
                layers.Dense(1)
            ])

        else:

            test_model = keras.Sequential([
                normalizer,
                layers.Dense(model_num - 5, activation='relu'),
                layers.Dense(model_num - 5, activation='relu'),
                layers.Dense(1)
            ])


        # Help from https://keras.io/api/optimizers/learning_rate_schedules/exponential_decay/
        # Learning rate will decline as the number of epochs increases

        initial_learning_rate = base_learning_rate * (2 / (models_to_test + 1))
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=100000,
            decay_rate=0.97,
            staircase=True)

        test_model.compile(loss='mean_squared_error',
                           optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), metrics=['mse'])

        sys.stdout = open("../Website-GUI/status.txt", 'a')

        print("\n\nBeginning Training.....\n")

        print("The Neural Network ID Number: " + str(model_num) + " has been Created and Compiled")
        print("------------------------------------------------------------")

        print("The Neural Network ID Number: " + str(model_num) + " is Starting to Run Over the Training Data")
        print("------------------------------------------------------------")

        history = test_model.fit(
            train_features,
            train_labels,
            validation_split=0.20,
            verbose=2, epochs=base_epochs + 10 * models_to_test,)

        print("Saving Neural Network ID Number: " + str(model_num))
        print("------------------------------------------------------------")

        # Save Neural Network in Folder for 1) Later Reference, or 2) Manipulation from GUI

        test_model.save('Trained-Models/' + str(model_num))

    #####################################################################################################################

    # Create Ensemble Prediction

    #####################################################################################################################

    # Help from https://medium.com/randomai/ensemble-and-store-models-in-keras-2-x-b881a6d7693f

    print("All Neural Networks Have Been Saved, Starting Ensemble Evaluation")
    print("------------------------------------------------------------")

    models = []
    for i in range(models_to_test):
        modelTemp = load_model('Trained-Models/' + str(i))
        models.append(modelTemp)

    mean_ens_guess = []

    # Other options for choosing how to combine ensembled models

    # mode_ens_guess = []
    # max_ens_guess = []

    for point in test_features:
        raw_guess = []
        for model in models:
            # print(point)
            # print(len(point))
            # print(model)
            raw_guess.append(model.predict(point))

        # print(raw_guess)
        mean_ens_guess.append(np.mean(raw_guess))

        # Other options for choosing how to combine ensembled models

        # mode_ens_guess.append(np.mode(raw_guess))
        # max_ens_guess.append(np.argmax(raw_guess)) - classification

    y_output = [model.predict(test_features) for model in models]
    y_average = layers.average(y_output)

    #####################################################################################################################

    # Plot Results of Predictions

    #####################################################################################################################

    print("The Neural Network is Starting to Run over the Test Data")
    print("------------------------------------------------------------")
    guess = test_model.predict(test_features)

    print("Plotting Results")
    print("------------------------------------------------------------")

    # Commented Out for Testing

    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [Guess]')
    plt.legend()
    plt.grid(True)

    # Single (Latest) Neural Network Results

    plt.subplot(1, 3, 2)
    x = tf.linspace(0.0, len(test_labels) - 1, len(test_labels))
    plt.plot(x, test_labels, label='How Much Actually Traded')
    plt.plot(x, guess, label='Single Network Guess at Price')
    plt.legend()

    # Ensembled Neural Network Results

    plt.subplot(1, 3, 3)
    x = tf.linspace(0.0, len(test_labels) - 1, len(test_labels))
    plt.plot(x, test_labels, label='How Much Actually Traded')
    plt.plot(x, mean_ens_guess, label='Ensembled Network Guess at Price')
    plt.legend()

    # Display the Plot

    plt.show()

    #####################################################################################################################

    # Storing results for later

    test_results = {'test_model': test_model.evaluate(test_features, test_labels, verbose=0)}
    print("Done.")
    #####################################################################################################################
    return 'Trained-Models/'
    sys.stdout.close()
    print_and_write_status("Training Complete")
if __name__ == "__main__":
    # Send all out put to status file
    def print_and_write_status(string):
        sys.stdout = open("../Website-GUI/status.txt", 'a')
        print(string)
        sys.stdout.close()

    # Put the training config into a dictionary
    with open("../Model-Training/training_config.txt", "r") as training_config_file:
        lines = training_config_file.readlines()

    # tfd stands for 'training config dict'
    tfd = {}
    for line in lines:
        for i in range(len(line)):
            if line[i] == ":":
                key = line[:i]
                value = line[(i+2):-1]
                tfd[key] = value


    ETF_created = CreateDataset(int(tfd["Lead_Up_Days"]), int(tfd["Momentum_Consideration"]), int(tfd["Putt_Call"]),
                                int(tfd["Junk_Bond"]), int(tfd["McClellan_Summation"]))

    location_of_models = CreateNeuralNetwork(ETF_created)


