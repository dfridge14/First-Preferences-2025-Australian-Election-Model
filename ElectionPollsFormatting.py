import pandas as pd
import numpy as np
from itertools import product
import os,time
from datetime import datetime
from pathlib import Path

# automatic error debugging
import sys
import pdb
import traceback

def exception_handler(type, value, tb):

    print("\a")  # Rings the system bell
    os.system('echo -e "\\a"')  # Extra bell command for reliability
    os.system('tput bel')  # This forces the terminal to beep

    traceback.print_exception(type, value, tb)  # Print the error as usual
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()  # Start debugger at the error location

sys.excepthook = exception_handler



base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else Path.home() / "Australian Election"
os.chdir(base_dir)

NO_ADDITIONAL_COLUMNS = 2 # Date, SS, UND (possibly also brand add in case of missing ss allocation) - removed UND!!!!!

election_years = ['2004','2007','2010','2013','2016','2019','2022','2025']
num_parties_per_election_year = {'2010':4,'2013':4,'2016':4,'2019':5,'2022':6,'2025': 5} # '2004':'2006','2007':'2006',


# 1. convert date ranges to medium dates
# 2. convert dates to numerical values starting at last election
# 3. For missing sample size, infer from other polls of the same outlet (need outlet)
# 4. for approximate ss, strip of non-numeric characters

# Function to extract the midpoint of date ranges

def parse_date_range(date_str):

    # Case 1: 18–19, 25–26 Sep 2021
    # Case 2: '29–30 May, 5–6 Jun 2021'
    # Case 3: '24–25, 31 Jul–1 Aug 2021'
    # Case 4: '28 Feb–1 Mar, 5–6 Mar 2021'


    # Ensure date_str is a string and not NaN
    if not isinstance(date_str, str):
        return None

    # Try parsing a single date (e.g., "10-Aug-20", "10 Aug 20")
    try:
        return datetime.strptime(date_str, "%d-%b-%y")  # Try first format
    except ValueError:
        pass  # Ignore if it doesn't match the first format

    try:
        return datetime.strptime(date_str, "%d %b %Y")  # Try second format
    except ValueError:
        pass  

    # Handle data like "13–19 May 2022"
    # If there is a dash, it separates 2 dates (1 dash = 1 range, 2 dashes = 2 ranges)
    # Case 1: single dash, dash straight after a number : "13–19 May 2022"
    
    if ',' not in date_str: # date_str is not composed of multiple ranges

        range_split = date_str.split('–')

        if len(range_split) != 2: # 
            import pdb;pdb.set_trace()
            raise ValueError("Expected exactly 2 parts separated by a -")
        
        end_date = datetime.strptime(range_split[1], "%d %b %Y")

        start_date_split = range_split[0].split(' ')
        end_date_split = range_split[1].split(' ')

        if start_date_split == [range_split[0]]: # "13–19 May 2022" guaranteed
            start_date_str = range_split[0] + ' '+ end_date_split[1]+' '+end_date_split[2]
            start_date = datetime.strptime(start_date_str, "%d %b %Y")

        else: # 25 Apr–1 May 2022 guaranteed
            start_date_str = range_split[0] + ' ' + end_date_split[-1]
            start_date = datetime.strptime(start_date_str, "%d %b %Y")

    else: # split occurs by comma - section before harbours first date, section after harbours 2nd date
        two_ranges = date_str.split(',')

        if len(two_ranges) != 2: 
            raise ValueError("Why are there 3 commas???")
        
        first_range, second_range = two_ranges[0],two_ranges[1].strip()

        year = second_range.split(' ')[-1]

        end_date = datetime.strptime(second_range.split('–')[-1], "%d %b %Y")
        
        first_range_split = first_range.split('–') # egs. ['28 Feb', '1 Mar'],['18', '19'],['29', '30 May']
        
        first_start_date_split = first_range_split[0].split(' ')# e.g. ['28', 'Feb'], ['18'], ['29']
        second_start_date_split = first_range_split[1].split(' ')
        #second_end_date_split = range_split[1].split(' ')

        if first_start_date_split == [first_range_split[0]]: # either ['18', '19'] or ['29', '30 May']

            if second_start_date_split == [first_range_split[1]]: # no info about months i.e. ['18', '19']  --> only use first date!!!
                start_date = first_start_date_split[0]
                # still need month!!! - use second range to extract
                second_range_split = [part.strip() for part in second_range.split('–')]#second_range.split('–')
                second_range_split_d1, second_range_split_d2 = second_range_split[0],second_range_split[1]

                if second_range_split_d1.split(' ') == [second_range_split_d1]: # e.g. [25] --> move to d2! # remove space after comma
                    start_date = start_date + ' ' + second_range_split_d2.split(' ')[1] + ' ' + year # add last date's month & year

                else: # month is there!!!!!
                    start_date = start_date + ' ' + second_range_split_d1.split(' ')[-1] + ' ' + year


            else:
                start_date = first_start_date_split[0] + ' ' + second_start_date_split[-1] + ' ' + year # '29' + 'May'

        else: # ['28 Feb', '1 Mar'] --> extract just the first element
            
            start_date = first_range_split[0] + ' ' + year

        start_date = datetime.strptime(start_date, "%d %b %Y") # format fully

    print(start_date,end_date)

    median_date = start_date + (end_date - start_date) / 2

    return median_date



election_year = '2022'
Polling_type = 'National'

last_election_date = {'2025': "21/05/22", '2022':"18/05/19", '2019':"02/07/16", '2016':"07/09/13", '2013':"21/08/10"}


if Polling_type == 'National':

    if election_year == '2022':
        
        Opinion_Polls_2022_National = pd.read_csv(f"{election_year}ElectionPollFormatted.csv").iloc[:,:num_parties_per_election_year[election_year]+NO_ADDITIONAL_COLUMNS].dropna(how='all')

        # Convert date format to datetime.date
        dates = pd.Series(Opinion_Polls_2022_National.iloc[:,0])
        parsed_median_dates = dates.apply(parse_date_range) 

        parsed_median_dates = pd.to_datetime(parsed_median_dates) # datetime objects

        last_election_date = datetime.strptime(Opinion_Polls_2022_National.iloc[-1,0], "%d-%b-%y") 
        days_since_election = (parsed_median_dates - last_election_date).dt.days

        Opinion_Polls_2022_National.iloc[:,0] = days_since_election
        Opinion_Polls_2022_National = Opinion_Polls_2022_National.rename(columns={"Date": "Days since last election"}) # not active yet
        Opinion_Polls_2022_National.loc[:,'Sample size'] = Opinion_Polls_2022_National.loc[:,'Sample size'].fillna(1000) # essential polls seem likely to be 1000 each

        Opinion_Polls_2022_National.iloc[:, 2:8] = Opinion_Polls_2022_National.iloc[:, 2:8].round(3)
        Opinion_Polls_2022_National.loc[:,['Days since last election','Sample size']] = Opinion_Polls_2022_National.loc[:,['Days since last election','Sample size']].astype(int)  
        Poll_Swings_National = Opinion_Polls_2022_National.iloc[:-1,]

        import pdb;pdb.set_trace()

        Poll_Swings_National.to_csv(f"NationalPollsforMGRW{election_year}.csv", index=False)


    if election_year == '2019':
        Opinion_Polls_National = pd.read_csv(f"{election_year}ElectionPollFormatted.csv").dropna(how='all')

        # Convert date format to datetime.date
        dates = pd.Series(Opinion_Polls_National.iloc[:,0])


        median_dates = pd.to_datetime(dates, errors='coerce')
        import pdb;pdb.set_trace()


        last_election_date = datetime.strptime(Opinion_Polls_National.iloc[-1,0], '%m/%d/%y') 
        days_since_election = (median_dates - last_election_date).dt.days

        import pdb;pdb.set_trace()


        Opinion_Polls_National.iloc[:,0] = days_since_election
        Opinion_Polls_National = Opinion_Polls_National.rename(columns={"Date": "Days since last election"}) 

        Opinion_Polls_National.iloc[:, 2:] = Opinion_Polls_National.iloc[:, 2:].apply(pd.to_numeric, errors='raise') # eliminate strings
        Opinion_Polls_National.iloc[:, 2:] = Opinion_Polls_National.iloc[:, 2:].round(3)
        Poll_Swings_National = Opinion_Polls_National.iloc[:-1,]
        Poll_Swings_National['Sample size'] = Poll_Swings_National['Sample size'].astype(int)  


        Poll_Swings_National = Poll_Swings_National.sort_values(by='Days since last election').reset_index(drop=True)


        # make Other category - bring together KAP,ACP,XEN, Other
        import pdb;pdb.set_trace()

        Insignificant_parties_2019 = ['XEN','KAP','ACP']
        #Insignificant_parties_sum = Poll_Swings_National[Insignificant_parties_2019].sum(axis=1)
        Poll_Swings_National = Poll_Swings_National.drop(Insignificant_parties_2019, axis=1)

        Other_col = 1 - Poll_Swings_National[['COAL','ALP','GRN','ON','UAPP']].sum(axis = 1)
        Poll_Swings_National.loc[:,'OTH'] = Other_col

        Poll_Swings_National.to_csv(f"NationalPollsforMGRW{election_year}.csv", index=False)



    if election_year == '2016':

        Opinion_Polls_National = pd.read_csv(f"{election_year}ElectionPollFormatted.csv").iloc[:,:num_parties_per_election_year[election_year]+NO_ADDITIONAL_COLUMNS].dropna(how='all')

        import pdb;pdb.set_trace()

        # Convert date format to datetime.date
        dates = pd.Series(Opinion_Polls_National.iloc[:,0])
        parsed_median_dates = dates.apply(parse_date_range) 

        parsed_median_dates = pd.to_datetime(parsed_median_dates) # datetime objects

        last_election_date = datetime.strptime(Opinion_Polls_National.iloc[-1,0], "%d-%b-%y") 
        days_since_election = (parsed_median_dates - last_election_date).dt.days

        import pdb;pdb.set_trace()


        Opinion_Polls_National.iloc[:,0] = days_since_election
        Opinion_Polls_National = Opinion_Polls_National.rename(columns={"Date": "Days since last election"}) # not active yet

        Opinion_Polls_National.iloc[:, 2:] = Opinion_Polls_National.iloc[:, 2:].apply(pd.to_numeric, errors='raise') # eliminate strings
        Opinion_Polls_National.iloc[:, 2:] = Opinion_Polls_National.iloc[:, 2:].round(3)
        Poll_Swings_National = Opinion_Polls_National.iloc[:-1,]
        Poll_Swings_National['Sample size'] = Poll_Swings_National['Sample size'].astype(int)  


        Poll_Swings_National = Poll_Swings_National.sort_values(by='Days since last election').reset_index(drop=True)
        import pdb;pdb.set_trace()


        Poll_Swings_National.to_csv(f"NationalPollsforMGRW{election_year}.csv", index=False)





        # Compute the median date
        #median_date = np.median(parsed_dates.dropna())








elif Polling_type == 'Electorate':

    if election_year == '2022':
        Opinion_Polls_Electorate = pd.read_csv(f"SeatPolls{election_year}Adjusted.csv").iloc[:,:13].drop('Pollster', axis = 1)

        # Convert date format to datetime.date
        dates = pd.Series(Opinion_Polls_Electorate.iloc[:,0]).str.strip()
        parsed_median_dates = dates.apply(parse_date_range) 

        parsed_median_dates = pd.to_datetime(parsed_median_dates) # datetime objects

        last_election_date =  datetime.strptime(last_election_date[election_year], '%d/%m/%y') 
        days_since_election = (parsed_median_dates - last_election_date).dt.days

        Opinion_Polls_Electorate.iloc[:,0] = days_since_election
        Opinion_Polls_Electorate.rename(columns={"Date": "Days since last election"}, inplace=True) # not active yet

        Opinion_Polls_Electorate.iloc[:, 3:] = Opinion_Polls_Electorate.iloc[:, 3:].round(3)
        Opinion_Polls_Electorate['Sample size'] = Opinion_Polls_Electorate['Sample size'].astype(int)  


        Opinion_Polls_Electorate = Opinion_Polls_Electorate.sort_values(by='Days since last election').reset_index(drop=True)

        import pdb;pdb.set_trace()

        Opinion_Polls_Electorate.to_csv(f"SeatPolls{election_year}Formatted.csv", index=False)

    if election_year == '2019':
        Opinion_Polls_Electorate = pd.read_csv(f"SeatPolls{election_year}.csv").drop('Pollster', axis = 1)

        # Convert date format to datetime.date
        dates = pd.Series(Opinion_Polls_Electorate.iloc[:,0])


        median_dates = pd.to_datetime(dates, format='%d/%m/%y', dayfirst=True, errors='coerce')


        last_election_date = datetime.strptime(last_election_date[election_year], '%d/%m/%y') 
        days_since_election = (median_dates - last_election_date).dt.days


        Opinion_Polls_Electorate.iloc[:,0] = days_since_election
        Opinion_Polls_Electorate = Opinion_Polls_Electorate.rename(columns={"Date": "Days since last election"}) # not active yet

        Opinion_Polls_Electorate.iloc[:, 3:] = Opinion_Polls_Electorate.iloc[:, 3:].round(3)
        Opinion_Polls_Electorate['Sample size'] = Opinion_Polls_Electorate['Sample size'].astype(int)  


        Opinion_Polls_Electorate = Opinion_Polls_Electorate.sort_values(by='Days since last election').reset_index(drop=True)

        import pdb;pdb.set_trace()


        Opinion_Polls_Electorate.to_csv(f"SeatPolls{election_year}Formatted.csv", index=False)

    if election_year == '2016':
        Opinion_Polls_Electorate = pd.read_csv(f"SeatPolls{election_year}.csv").drop('Pollster', axis = 1)

        # Convert date format to datetime.date
        dates = pd.Series(Opinion_Polls_Electorate.iloc[:,0])
        median_dates = pd.to_datetime(dates, format='%d/%m/%Y', dayfirst=True, errors='coerce')


        last_election_date = datetime.strptime(last_election_date[election_year], '%d/%m/%y') 
        days_since_election = (median_dates - last_election_date).dt.days

        Opinion_Polls_Electorate.iloc[:,0] = days_since_election
        Opinion_Polls_Electorate = Opinion_Polls_Electorate.rename(columns={"Date": "Days since last election"}) # not active yet

        Opinion_Polls_Electorate.iloc[:, 3:] = Opinion_Polls_Electorate.iloc[:, 3:].round(3)
        Opinion_Polls_Electorate['Sample size'] = Opinion_Polls_Electorate['Sample size'].astype(int)  


        Opinion_Polls_Electorate = Opinion_Polls_Electorate.sort_values(by='Days since last election').reset_index(drop=True)

        import pdb;pdb.set_trace()


        Opinion_Polls_Electorate.to_csv(f"SeatPolls{election_year}Formatted.csv", index=False)


    if election_year == '2013':
        Opinion_Polls_Electorate = pd.read_csv(f"SeatPolls{election_year}Adjusted.csv").drop('Pollster', axis = 1)

        # Convert date format to datetime.date
        dates = pd.Series(Opinion_Polls_Electorate.iloc[:,0])
        median_dates = pd.to_datetime(dates, format='%d/%m/%y', dayfirst=True, errors='coerce')


        last_election_date = datetime.strptime(last_election_date[election_year], '%d/%m/%y') 
        days_since_election = (median_dates - last_election_date).dt.days

        Opinion_Polls_Electorate.iloc[:,0] = days_since_election
        Opinion_Polls_Electorate = Opinion_Polls_Electorate.rename(columns={"Date": "Days since last election"}) # not active yet

        Opinion_Polls_Electorate.iloc[:, 3:] = Opinion_Polls_Electorate.iloc[:, 3:].round(3)
        Opinion_Polls_Electorate['Sample size'] = Opinion_Polls_Electorate['Sample size'].astype(int)  


        Opinion_Polls_Electorate = Opinion_Polls_Electorate.sort_values(by='Days since last election').reset_index(drop=True)

        import pdb;pdb.set_trace()


        Opinion_Polls_Electorate.to_csv(f"SeatPolls{election_year}Formatted.csv", index=False)

        



