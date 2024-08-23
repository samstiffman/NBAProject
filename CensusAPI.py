# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 09:12:34 2022

@author: samst
"""

# -*- coding: utf-8 -*-
"""
Census Docs:
    https://www.census.gov/programs-surveys/acs/technical-documentation/code-lists.html
    https://www.census.gov/programs-surveys/cps/technical-documentation/complete.html
Census Data Link:
    https://www.census.gov/data/developers/data-sets/acs-5year.html
Data Set:     
    Data Profile 


"""
import requests

# ########### CSV Setup #################
# Create a blank CSV
fname = 'rawCensusData.csv'
file = open(fname, "w")

#Header Row
# cols = "id, name, releaseDate, genre, Rating\n"
# file.write(cols)
file.close()


# Using census API
# State 27 is Minnesota
# Split by county
# DP02 is school stuff P is percentages
# 0061PE 25+ 9-12th grade no HS grad
# 0062PE 25+ HS Grad
# 0063PE 25+ no colledge grad
# 0064PE 25+ associate degree
# 0065PE 25+ bachelors
# 0066PE 25+ Grad or proff deg
ED = "DP02_0061PE,DP02_0062PE,DP02_0063PE,DP02_0064PE,DP02_0065PE,DP02_0066PE"
# Money
# DP03 is work and money stuff
# these are % household incomes + benefits (2020 inflation adjusted)
# 0052PE is % household < 10k
# 0053PE is % household  10-15k
# 0054PE is % household  15-25k
# 0055PE is % household  25-35k
# 0056PE is % household  35-50k
# 0057PE is % household  50-75k
# 0058PE is % household  75-100k
# 0059PE is % household  100-150k
# 0060PE is % household  150-200k
# 0061PE is % household  >200k
MONEY = "DP03_0052PE,DP03_0053PE,DP03_0054PE,DP03_0055PE,DP03_0056PE,DP03_0057PE,DP03_0058PE,DP03_0059PE,DP03_0060PE,DP03_0061PE"
APIKey = '23db7fcba79c52e2c9c19cdd98ab3e8d1d9cc844'
APIURL = 'https://api.census.gov/data/2020/acs/acs5/profile'
getBody = {
        'get' : "NAME,REGION,"+ED+","+MONEY,
        'for' : 'county:*',
        'in' : 'state:27',
        'key' : APIKey
    }

# # Make API Get Call
response = requests.get(APIURL, getBody)
result = response.json()
print(result)




# ######### Writing Raw Data ############
file = open(fname, 'a', encoding="utf-8")
for datum in result:
    # print(datum)
    file.write(",".join(map(str, datum)) + "\n")
file.close()











