# TODO: find some way to take care of data on market close days
# imports
import pandas as pd
import os
import sys
import datetime
import requests
import pickle
import wget
from random import randint
from time import sleep

# Add the path to the Data folder
sys.path.insert(1, '../Data')
totalpc_filename = "totalpc.csv"
pickle_file = "scraped_data.pkl"
output_file = "../Data/Put-Call-Ratio/" + totalpc_filename

# Setup function that prints and writes status to Update-Data-Status file
def print_and_write_status(string):
    print(string)
    with open("Update-Data-Status.txt", 'a') as f:
        f.writelines("\n" + string)



# Make the Put-Call-Ratio folder if it doesnt exhist
if not os.path.exists(output_file):
    os.makedirs(output_file)

def get(url):
    headers = {}
    try:
        resp = requests.get(url, timeout=30)
        if resp.ok:
            return resp.text
    except requests.exceptions.RequestException as e:
        return None


def clean_fresh_totalpc_download():
    # The following cell loads totalpc.csv
    # It then appends the data and saves it as a new file
    df = pd.read_csv(output_file)
    df.columns = ["date", "calls", "puts", "total", "p_c_ratio"]
    df = df.drop(labels=[0, 1], axis=0)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.to_csv(output_file)


def remove_market_close_days(scrapeDict):
    opt_out_days = [
        "2019-11-28",
        "2019-12-25",
        "2020-01-01",
        "2020-01-20",
        "2020-02-17",
        "2020-04-10",
        "2020-05-25",
        "2020-07-03",
        "2020-09-07",
        "2020-11-26",
        "2020-12-25",
        "2021-01-01",
        "2021-01-18",
        "2021-02-15",
        "2021-04-02",
        "2021-05-31",
        "2021-07-05",
        "2021-09-06",
        "2021-11-25",
        "2021-12-24",
        "2022-01-17",
    ]
    # Getting rid of holidays listed under [opt_out_days]
    for i in opt_out_days:
        try:
            del scrapeDict[datetime.datetime.strptime(i, "%Y-%m-%d").date()]
        except KeyError:
            continue
    print_and_write_status("Adjusted for opt-out-days:", len(scrapeDict.keys()))
    return scrapeDict


# this function scrapes CBOE data from start_date to current_date
# then saves that data as a pickle file
# then loads totalpc.csv and appends to it
def run_scraper(start_date=datetime.date(2019, 10, 5)):
    # start_date = datetime.datetime.strftime(start_date, "%Y-%m-%d")
    # Run the following cell to collect data again(warning: this will overwrite the pickle file)
    # Make sure to set the start date and pickle file name
    # scrape data from CBOE
    # after 10-4-2019 mm-dd-yyyy
    # url format yyyy-mm-dd
    print_and_write_status("Updating Put-Call-Ratio's...")

    base_url = (
        "https://markets.cboe.com/us/options/market_statistics/daily/?mkt=cone&dt="
    )
    outfile = open(pickle_file, "wb")
    """This part is to scrape from CBOE website and save the file"""
    scrapeDict = {}
    weekdays = []
    end_date = datetime.date.today()
    delta = datetime.timedelta(days=1)
    run_date = start_date
    while run_date <= end_date:
        if run_date.weekday() not in [5, 6]:  # ie. Mon-Fri only
            weekdays.append(run_date)
        run_date += delta
    if len(weekdays)>0:

        print_and_write_status("Updating Put-Call-Ratio's...")

        for i, get_date in enumerate(weekdays):
            html_date = datetime.datetime.strftime(get_date, "%Y-%m-%d")
            print_and_write_status(get_date, end="|")
            data = get(base_url + html_date)
            if data == None:
                print_and_write_status("no data")
                continue
            scrapeDict[get_date] = pd.read_html(data)
            sleep(1)
            # dump to disk every 5 entries
            if i % 5 == 0:
                pickle.dump(scrapeDict, outfile)

        # remove data from when market was closed
        scrapeDict = remove_market_close_days(scrapeDict)
        # overwrites existing pickle data
        outfile.close()
        df = pd.DataFrame(scrapeDict)
        print_and_write_status("Done!")
        # now remove un-needed data from the dataframe
        # format: DATE(mm/dd/yyyy) | CALLS | PUTS | TOTAL | P/C Ratio
        df = df.transpose()
        df = df.drop(axis="columns", columns=range(2, 9))
        # turn name of series into another column
        # eventually to turn it into the index
        df["date"] = df.apply(lambda row: row.name, axis=1)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df["calls"] = df.apply(lambda row: row.iloc[1].iloc[1][1], axis=1)
        df["puts"] = df.apply(lambda row: row.iloc[1].iloc[1][2], axis=1)
        df["total"] = df.apply(lambda row: row.iloc[1].iloc[1][3], axis=1)
        df["p_c_ratio"] = df.apply(lambda row: row.iloc[0].iloc[0][1], axis=1)
        df = df.drop(axis="columns", columns=[0, 1])
        print_and_write_status(df.tail(), df.shape)

        # The following cell loads totalpc.csv
        # It then appends the data and saves it as a new file
        total_df = pd.read_csv(output_file, index_col=0, parse_dates=True)
        print_and_write_status(total_df.tail(), total_df.shape)
        final_df = pd.concat([total_df, df], axis=0, verify_integrity=True, sort=True)
        final_df = final_df.astype({"calls": "int32", "total": "int32", "puts": "int32"})
        final_df = final_df[["calls", "puts", "total", "p_c_ratio"]]
        print_and_write_status(final_df.tail(), final_df.shape)
        print_and_write_status(final_df.dtypes)
        print_and_write_status(final_df.tail())
        # cleanup
        final_df = final_df.dropna(axis=0)
        final_df.to_csv(output_file)
    else:
        print_and_write_status("...No more data to add")


def main():
    # cases:
    # case 1: we already have totalpc.csv
    # totalpc.csv needs to be updated to current day
    # assuming it was created by this script and run_scraper
    # case 2: we don't have totalpc.csv
    # check if file exists
    # flags
    existing_data = False

    # case 2 taken care of first
    with os.scandir("../Data/Put-Call-Ratio") as files_in_current_directory:
        for file in files_in_current_directory:
            # fyi this checks if the pickle_name var string is in any of the file names
            if totalpc_filename in file.name:
                existing_data = True

    # if historical data file doesn't exist get it from CBOE
    if existing_data == False:
        print_and_write_status("Downloading Put-Call-Ratio's...")
        url = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv"
        wget.download(url, output_file)
        # we now have the file with old data
        # now we must do an initial run to get it up to date
        # default arg takes care of data starting Oct 5 2019
        clean_fresh_totalpc_download()
        run_scraper()
        # this gives us scraped_data.pkl and #totalpc.csv

    # case 1:
    # update totalpc.csv to current_day
    if existing_data == True:
        # find last date in dataset and set that as start date
        # end date will be current date as usual
        df = pd.read_csv(output_file)
        start_date = datetime.datetime.strptime(df.iloc[-1].date, "%Y-%m-%d").date()
        # since start_date is already in the csv, we'll increment it by 1
        start_date += datetime.timedelta(days=1)
        run_scraper(start_date)
        # this will upate totalpc.csv to the current day


if __name__ == "__main__":
    main()
