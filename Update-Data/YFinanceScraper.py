"""This is a Scraper that is using the yfinance library to make use of Yahoo Finance data without having to as n/
check every single page. It's consolidated what would be a very tedious scraping operation into a simple solution
thanks to Yahoo Finance's generous API."""
import os
import sys
import yfinance as yf
import requests
from datetime import date

# Add the path to the Data folder
sys.path.insert(1, '../Data')

# Run the CBOE scraper code first
exec(open("./scrape_cp_ratio.py").read())

# Now run the YFinanceScraper code below....
"""These are the 3x ETF tickers provided by: https://etfdb.com/themes/leveraged-3x-etfs/ """
tickers = ["TQQQ", "SOXL", "FAS", "UPRO", "SPXL", "TECL", "SQQQ", "TNA",
           "FNGU", "LABU", "UDOW", "ERX", "NUGT", "SPXU", "YINN", "DPST", "TZA", "SPXS", "TMF", "TMV", "URTY", "SDOW",
           "DFEN", "NAIL", "CURE", "TTT", "BRZU", "SRTY", "SOXS", "EDC", "DRN", "FAZ", "RETL", "FNGD", "TECS", "INDL",
           "MIDU", "TPOR", "YANG", "LABD", "UMDD", "SBND", "EURL", "DUSL", "KORU", "TYO", "EDZ", "DRV", "UBOT", "PILL",
           "ERY", "TYD", "UTSL", "OILU", "MEXX", "SMDD", "OILD"]

# These are 'junktickers' pulled from yahoo finance
junktickers = ["FTSL", "HYG", "JNK", "SRLN", "USHY"]

mcClellan = ["DOW", "SPX"]

# This function prints and updates the Update-Data-Status.txt file
def print_and_write_status(string):
    print(string)
    with open("Update-Data-Status.txt", 'a') as f:
        f.writelines("\n" + string)


def get_3xETF_tickers():
    # create 3X-ETF directory within the current directory if it doesn't already exist
    cur_dir = os.getcwd()
    out_dir = "../Data/3X-ETF"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for name in tickers:
        ticker = yf.Ticker(name)

        # prints to console to show user progress
        print_and_write_status("...Updating " + str(name))

        # downloading historical market data
        data_df = ticker.history(period="max")

        # Save this ETF's  to a CSV file
        data_df.to_csv('../Data/3X-ETF/' + name + '.csv')


def get_junk_tickers():
    # create Junk-Bond-ETF directory within the current directory if it doesn't already exist
    out_dir = '../Data/Junk-Bond-ETF'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for name in junktickers:
        ticker = yf.Ticker(name)

        # prints to console to show user progress
        print_and_write_status("...Updating " + str(name))

        # get historical market data
        data_df = ticker.history(period="max")

        # Write  to a csv file
        data_df.to_csv('../Data/Junk-Bond-ETF/' + name + '.csv')


def get_mcClellan():
    # Create McClellan Summation Index directory within the current directory if it doesn't already exist
    out_dir = ('../Data/McClellan-Summation-Index')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Get the SPX  --------------------------------

    # Get today's date
    today = str(date.today())

    # Remove dashes from the date
    today = today.replace('-', '')
    url = str("https://stooq.com/q/d/l/?s=^spx&d1=19900101&d2=" + today + "&i=d")

    # Prints to console to show user progress
    print_and_write_status("...Updating SPX ")

    # Downloading CSV file from
    download = requests.get(url)

    # Downloading CSV file
    with open("../Data/McClellan-Summation-Index/SPX.csv", 'wb') as f:
        # Writing downloaded data to SPX.csv
        f.write(download.content)

    # Get the DOW's  -------------------------------
    dow = yf.Ticker("DOW")
    # prints to console to show user progress
    print_and_write_status("...Updating DOW ")
    # get historical market data
    dowHist = dow.history(period="max")

    # Save to a csv file
    dowHist.to_csv('../Data/McClellan-Summation-Index/' + "DOW" + '.csv')


print_and_write_status("Updating 3xETF's...")
get_3xETF_tickers()

print_and_write_status("Updating Junk Bonds...")
get_junk_tickers()

print_and_write_status("Updating mcClellan...")
get_mcClellan()

print_and_write_status("Update Complete")