import pandas as pd 

import re
from tqdm import tqdm
import urllib
from urllib.request import urlopen


def scrape_aaindex_indicies():
    #find all accession numbers located on AAindex and save to list 
    #open URL of AAindex accession number guide
    url = 'https://www.genome.jp/aaindex/AAindex/list_of_indices?fbclid=IwAR3gFoW9Iy1k_krEh1Jbc05TWrF8JWT2usx1hVNFDlcMREe2rgQGGntuRMw'
    page = urlopen(url)
    html = page.read().decode("utf-8")

    #find every instance of a new row
    positions = [m.start() for m in re.finditer('\n', html)]

    #each accession numbers is 10 characters long so we take...
    #ten characters after each instance of a new row
    names_dirty = [html[i+1:i+11] for i in positions]

    #isolate the "dirty" names that contain 10 cahracters...
    #with exactly 6 numbers within
    names_clean = []
    for _, sample in enumerate(names_dirty):
        numbers = sum(c.isdigit() for c in sample)
        if numbers == 6:
            names_clean.append(sample)

    return names_clean
    #output list: list of AAindex accession numbers

def find_accession_numbers(clean_names):
    #find all accession numbers located on AAindex and save to list 
    AA_list = ['A', 'R', 'N', 'D', 'C', 
            'Q', 'E', 'G', 'H', 'I', 
            'L', 'K', 'M', 'F', 'P', 
            'S', 'T', 'W', 'Y', 'V']

    mega_set = pd.DataFrame({'Amino Acid Code': AA_list})

    for row, sample in enumerate(clean_names):

        url = 'https://www.genome.jp/entry/aaindex:'+ sample
        
        #open url to specific accession number
        page = urlopen(url)
        html = page.read().decode("utf-8")
        
        #isolate AAindex title
        title_index_start = html.find('<title>') + len('<title>')
        title_index_end = html.find('</title>')
        title = html[title_index_start:title_index_end]
        
        #isolate AA information 
        start_index = html.find('A/L')
        end_index = html[start_index:].find('//')
        data = html[start_index:start_index+end_index]
        first_row_start = data.find('\n')
        first_row_end = data[first_row_start:]

        #split string into each amino acids property
        property_list = first_row_end.split()

        #save property list to external dataframe...
        #with accession number title at the identifier
        mega_set[str(title)] = property_list

    return mega_set
