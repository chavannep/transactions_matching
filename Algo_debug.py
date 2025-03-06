#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
from os import makedirs, listdir, rename, path, remove, getlogin, chmod
import shutil
import stat
from re import findall, sub
import random
import os

# Tailor made subroutines
from python_utils import *

timestamp_start = datetime.now()

def clock_splitter():
    timestamp_end = datetime.now()
    elapsed_time = (timestamp_end - timestamp_start) / pd.Timedelta(seconds=1)
    time_hr = int(elapsed_time/3600)
    time_mn = int((elapsed_time - time_hr*3600)/60)
    time_sec = int(np.ceil(elapsed_time - time_hr*3600 - time_mn*60))
    return(time_hr, time_mn, time_sec)
    

# Algorithm version
algo_version = 'V2.1'  # Updated version number

# Max margin reachable (gap between matched data <= margin) 
date_margin = 3.0 # hours
quantity_margin = 0.5 # kWh

# Main results' folder
main_folder_name = 'Resultats'
# Other results' folders
others_folder_name = 'Resultats'

# Authorized file extensions
authorized_extension = ['csv', 'xlsx', 'CSV', 'XLSX']
authorized_extraction = ['.tar', '.zip', '.TAR', '.ZIP']








#########################
# Creating Results and Errors folders
#########################
if path.exists('./' + main_folder_name):
    chmod('./' + main_folder_name, 0o777)
    shutil.rmtree('./' + main_folder_name)

makedirs('./' + main_folder_name, exist_ok=True)

error_file_path = './' + main_folder_name + '/ERREUR.txt'
log_file_path = './' + main_folder_name + '/LOG.txt'

with open(error_file_path, 'w') as f:
    f.write('#################################################################')
    f.write('\n')
    f.write("                  Fichier de rapport d'erreurs")
    f.write('\n')
    f.write('#################################################################')
    f.write('\n\n')
    
error_file_path_len_1 = path.getsize(error_file_path)

with open(log_file_path, 'w') as f:
    pass



#########################
# Verifying Python algo in the folder
#########################
keyword = 'Algo'
python_file_cpt = 0
python_file_name = ''
for i in listdir('./'):
    if ((keyword.lower() in i.lower()) & ('.py' in i.lower())):
        python_file_name = i
        python_file_cpt += 1

python_file_bool = False
if python_file_name:
    python_file_bool = True
else:
    with open(error_file_path, 'a') as f:
        f.write('\n')
        f.write("Fichier Python manquant, réconciliation impossible. Veuillez le recharger.")
        f.write('\n\n')
    raise SystemExit()




#########################
# Verifying Bemo transactions file in the folder
#########################

keyword = 'Bemo'
bemo_file_cpt = 0
bemo_file_name = None

# First, check for a directory
for i in listdir('./'):
    if keyword.lower() in i.lower() and path.isdir('./' + i):
        bemo_file_name = i
        bemo_file_cpt += 1
        break

# If no directory found, look for specific files
if not bemo_file_name:
    for i in listdir('./'):
        if keyword.lower() in i.lower() and path.isfile('./' + i) and i.endswith(('.csv', '.CSV')):
            bemo_file_name = i
            bemo_file_cpt += 1
            break

bemo_file_bool = False
df_bemo = pd.DataFrame()

try:
    if bemo_file_name:
        if path.isdir('./' + bemo_file_name):
            # Directory containing the CSV files
            directory = './' + bemo_file_name
            
            # List to hold DataFrames
            dfs = []
            
            # Loop through files in the directory
            for filename in listdir(directory):
                if filename.endswith('.csv'):
                    try:
                        file_path = path.join(directory, filename)
                        df = pd.read_csv(file_path, decimal='.', delimiter=',')
                        dfs.append(df)
                        print(f"Successfully loaded {filename}")
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
                        
            if dfs:
                # Concatenate all DataFrames
                df_bemo = pd.concat(dfs, ignore_index=True)
                
                # Rename columns to add _bemo suffix
                for j in df_bemo.columns:
                    df_bemo.rename(columns={j: str(j + '_bemo')}, inplace=True)
                    
                bemo_file_bool = True
            else:
                raise Exception("No valid CSV files found in Bemo directory")
        else:
            # It's a direct file
            df_bemo = pd.read_csv('./' + bemo_file_name, decimal='.', delimiter=',')
            
            # Rename columns to add _bemo suffix
            for j in df_bemo.columns:
                df_bemo.rename(columns={j: str(j + '_bemo')}, inplace=True)
                
            bemo_file_bool = True
    else:
        raise Exception("No Bemo transactions file or directory found")
        
except Exception as e:
    print(f"Error loading Bemo transactions: {e}")
    with open(error_file_path, 'a') as f:
        f.write('\n')
        f.write(f"Fichier Bemo invalide, réconciliation impossible: {str(e)}. Veuillez le recharger.")
        f.write('\n\n')
    raise SystemExit()




#########################
# Verifying Checklist file in the folder
#########################

keyword = 'Checklist'
checklist_file_cpt = 0
checklist_file_name = None

for i in listdir('./'):
    if keyword.lower() in i.lower():
        checklist_file_name = i
        checklist_file_cpt += 1
        break

checklist_file_bool, df_checklist_params = False, pd.DataFrame()

if checklist_file_name:
    checklist_file_bool, df_checklist_params = specific_file_loader(str('./' + checklist_file_name), checklist_file_name, authorized_extension)

if not checklist_file_bool:
    with open(error_file_path, 'a') as f:
        f.write('\n')
        f.write("Fichier Checklist Mandataires manquant, réconciliation impossible. Veuillez le recharger.")
        f.write('\n\n')
    raise SystemExit()
else:
    # Process checklist params
    try:
        params_checklist = list(df_checklist_params.columns)
        
        # Check if required columns exist
        required_columns = ['Priority', 'Billing_Agent', 'Billing_Agent_Keyword']
        missing_columns = [col for col in required_columns if col not in params_checklist]
        
        if missing_columns:
            with open(error_file_path, 'a') as f:
                f.write('\n')
                f.write(f"Fichier Checklist mal formaté: colonnes manquantes: {missing_columns}. Réconciliation impossible.")
                f.write('\n\n')
            raise SystemExit()
            
        # Remove non-parameter columns
        if 'Priority' in params_checklist:
            params_checklist.remove('Priority')
        if 'Billing_Agent_Mail_Domain' in params_checklist:
            params_checklist.remove('Billing_Agent_Mail_Domain')
            
        # Sort and clean checklist
        df_checklist_params.sort_values(by=['Billing_Agent_Keyword'], ascending=[True], inplace=True)
        df_checklist_params.reset_index(drop=True, inplace=True) 
        df_checklist_params.drop_duplicates(subset=params_checklist, keep='first', inplace=True, ignore_index=True)
        df_checklist_params.sort_values(by=['Priority'], ascending=[True], inplace=True)
        df_checklist_params.reset_index(drop=True, inplace=True)
    except Exception as e:
        print(f"Error processing checklist: {e}")
        with open(error_file_path, 'a') as f:
            f.write('\n')
            f.write(f"Erreur de traitement du fichier Checklist: {str(e)}. Réconciliation impossible.")
            f.write('\n\n')
        raise SystemExit()




#########################
# Verifying Billing Reference file in the folder
#########################

keyword = 'Mapping'
reference_billing_file_cpt = 0
reference_billing_file_name = None

for i in listdir('./'):
    if keyword.lower() in i.lower():
        reference_billing_file_name = i
        reference_billing_file_cpt += 1
        break

reference_billing_file_bool, df_reference_billing = False, pd.DataFrame()

if reference_billing_file_name:
    reference_billing_file_bool, df_reference_billing = specific_file_loader(str('./' + reference_billing_file_name), reference_billing_file_name, authorized_extension)
    
if not reference_billing_file_bool:
    with open(error_file_path, 'a') as f:
        f.write('\n')
        f.write("Fichier Mapping Mandataires manquant, réconciliation impossible. Veuillez le recharger.")
        f.write('\n\n')
    raise SystemExit()




#########################
# Looping through CPO bills in the folder
#########################
keyword = 'Factures'

# Find factures directory
billingAgent_name_list = []
billingAgent_folder_bool = False
dir_name = None

for i in listdir('./'):
    if keyword.lower() in i.lower():
        dir_name = i
        concerned_year, concerned_month = date_detector(dir_name)
        concerned_year = [int(item) for item in concerned_year]
        concerned_month = [int(item) for item in concerned_month]
        billingAgent_folder_bool = True
        break

if not billingAgent_folder_bool:
    with open(error_file_path, 'a') as f:
        f.write('\n')
        f.write("Dossier Factures manquant, réconciliation impossible. Veuillez le recharger.")
        f.write('\n\n')
    raise SystemExit()

# If no date found in directory name, use current date
if not concerned_year:
    current_year = datetime.now().year
    concerned_year = [current_year]
    
if not concerned_month:
    current_month = datetime.now().month
    concerned_month = [current_month]

print(f"Processing data for year(s): {concerned_year}, month(s): {concerned_month}")

# Archive extraction loop
billingAgent_name_list = []
file_nb_to_compute = 0
file_path_1 = './' + dir_name + '/'

# Count billing agents and files
try:
    # Billing agents loop
    for i in listdir(file_path_1):
        billing_agent_path = file_path_1 + i + '/'
        if not path.isdir(billing_agent_path):
            continue
            
        if i.lower() in [x.lower() for x in df_checklist_params['Billing_Agent_Keyword'].unique()]:
            if i not in billingAgent_name_list:
                billingAgent_name_list.append(i)
                
            # Year loop
            for j in listdir(billing_agent_path):
                year_path = billing_agent_path + j + '/'
                if not path.isdir(year_path):
                    continue
                    
                if int(j) in concerned_year:
                    # Month loop
                    for k in listdir(year_path):
                        month_path = year_path + k + '/'
                        if not path.isdir(month_path):
                            continue
                            
                        if int(k) in concerned_month:
                            # Create results folder
                            makedirs(month_path + others_folder_name + '/', exist_ok=True)

                            # Extract archives if present
                            file_list = listdir(month_path)
                            file_to_extract = [l for l in file_list if any(m in l for m in authorized_extraction)]
                            for l in file_to_extract:
                                try:
                                    shutil.unpack_archive(month_path + l, month_path)
                                except Exception as e:
                                    print(f"Error extracting {l}: {e}")
                                
                            # Count files
                            file_nb_to_compute += len([l for l in listdir(month_path) 
                                                    if path.isfile(month_path + l) and 
                                                    any(l.lower().endswith(m.lower()) for m in authorized_extension)])
                        else:
                            # Remove non concerned month in this concerned year folder
                            shutil.rmtree(month_path, onerror=remove_readonly)    
                else:
                    # Remove non concerned year
                    shutil.rmtree(year_path, onerror=remove_readonly)
        else:
            with open(error_file_path, 'a') as f:
                f.write('\n')
                f.write(f"{{'Mandataire de facturation CPO non enregistré (checklist/mapping) : ':<120}}{str(i):<30}")
                f.write('\n')
except Exception as e:
    print(f"Error scanning billing agent folders: {e}")
    with open(error_file_path, 'a') as f:
        f.write('\n')
        f.write(f"Erreur lors du scan des dossiers: {str(e)}")
        f.write('\n\n')

# Sort billing agents by priority
if billingAgent_name_list and checklist_file_bool:
    try:
        # Create a case-insensitive priority dictionary
        order_dict = {}
        for idx, row in df_checklist_params.iterrows():
            order_dict[row['Billing_Agent_Keyword'].lower()] = row['Priority']
            
        # Sort billing agents by priority
        billingAgent_name_list.sort(key=lambda x: order_dict.get(x.lower(), float('inf')))
    except Exception as e:
        print(f"Error sorting billing agents: {e}")

#########################
# Data featuring
#########################

## Cleaning
df_bemo.dropna(how='all', inplace=True)
df_bemo.drop_duplicates(inplace=True)
df_bemo.reset_index(drop=True, inplace=True)


## Loading variables

# Start date
var_startdate_bemo = None
cond = (df_checklist_params['Billing_Agent'] == 'Bemo')
if 'StartDate_param' in df_checklist_params.columns and any(cond):
    if isinstance(df_checklist_params.loc[cond, 'StartDate_param'].iloc[0], str):
        var_startdate_bemo = df_checklist_params.loc[cond, 'StartDate_param'].iloc[0] + '_bemo'

# Quantity
var_quantity_bemo = None
if 'Quantity_param' in df_checklist_params.columns and any(cond):
    if isinstance(df_checklist_params.loc[cond, 'Quantity_param'].iloc[0], str):
        var_quantity_bemo = df_checklist_params.loc[cond, 'Quantity_param'].iloc[0] + '_bemo'

# Price
var_price_bemo = None
if 'Price_param' in df_checklist_params.columns and any(cond):
    if isinstance(df_checklist_params.loc[cond, 'Price_param'].iloc[0], str):
        var_price_bemo = df_checklist_params.loc[cond, 'Price_param'].iloc[0] + '_bemo'

# Check if required columns exist
if not var_startdate_bemo or not var_quantity_bemo or not var_price_bemo:
    with open(error_file_path, 'a') as f:
        f.write('\n')
        f.write("Paramètres Bemo manquants dans le fichier Checklist. Réconciliation impossible.")
        f.write('\n\n')
    raise SystemExit()

# Check if columns exist in the dataframe
missing_cols = []
for col in [var_startdate_bemo, var_quantity_bemo, var_price_bemo]:
    if col not in df_bemo.columns:
        missing_cols.append(col)
        
if missing_cols:
    with open(error_file_path, 'a') as f:
        f.write('\n')
        f.write(f"Colonnes manquantes dans le fichier Bemo: {missing_cols}. Réconciliation impossible.")
        f.write('\n\n')
    raise SystemExit()

#########################
# Formatting data
#########################

# Start date in local time
df_bemo[var_startdate_bemo], bool_datetime_bemo = datetime_filt(df_bemo[var_startdate_bemo])

if not bool_datetime_bemo:
    with open(error_file_path, 'a') as f:
        f.write('\n')
        f.write(f"Erreur de conversion de la date dans le fichier Bemo. Réconciliation impossible.")
        f.write('\n\n')
    raise SystemExit()

# Quantity
df_bemo[var_quantity_bemo] = float_filt(df_bemo[var_quantity_bemo])

# Price
df_bemo[var_price_bemo] = float_filt(df_bemo[var_price_bemo])

## Quantity conversion test
test_quantile = 0.5
cond = df_bemo[var_quantity_bemo].isna()
quantity_bemo_test_quantile = 0
if not df_bemo[~cond].empty:
    quantity_bemo_test_quantile = df_bemo[var_quantity_bemo][~cond].quantile([test_quantile])[test_quantile]
    if quantity_bemo_test_quantile > 100:
        df_bemo[var_quantity_bemo] = df_bemo[var_quantity_bemo] / 1000
        
#########################
# Restricting dates
#########################
cond = ((df_bemo[var_startdate_bemo].dt.year.isin(concerned_year)) &
        (df_bemo[var_startdate_bemo].dt.month.isin(concerned_month)))
df_bemo = df_bemo[cond]
df_bemo.reset_index(drop=True, inplace=True)

#########################
# Restricting Billing agents
#########################
if billingAgent_name_list:
    # Get list of CPO IDs for the billing agents in our list
    cpo_id_list = []
 
    for agent in billingAgent_name_list:
        # Case-insensitive match for billing agent keywords
        cond = (df_reference_billing['Billing_Agent_Keyword'].str.lower() == agent.lower())
        if any(cond):
            agent_cpo_ids = list(df_reference_billing['Cpo_Id'][cond].unique())
            cpo_id_list.extend(agent_cpo_ids)
            print(f"Found {len(agent_cpo_ids)} CPO IDs for agent {agent}")
    
    # Filter Bemo data to only include relevant CPO IDs
    if cpo_id_list:
        cond = (df_bemo['Cpo_Id_bemo'].isin(cpo_id_list))
        df_bemo = df_bemo[cond]
        df_bemo.reset_index(drop=True, inplace=True)
    else:
        with open(error_file_path, 'a') as f:
            f.write('\n')
            f.write("Aucun CPO ID trouvé pour les mandataires de facturation. Réconciliation impossible.")
            f.write('\n\n')
        raise SystemExit()
else:
    with open(error_file_path, 'a') as f:
        f.write('\n')
        f.write("Aucun mandataire de facturation trouvé. Réconciliation impossible.")
        f.write('\n\n')
    raise SystemExit()

# Original indicators
total_bemo_amount_sum = df_bemo[var_price_bemo].sum()
total_bemo_count_sum = df_bemo.shape[0]

# Initialize summary variables
amount_match_bemo_sum = 0
amount_match_billingAgent_sum = 0
amount_unmatch_bemo_sum = 0
amount_unmatch_billingAgent_sum = 0
count_match_sum = 0
count_unmatch_bemo_sum = 0
count_unmatch_billingAgent_sum = 0

# File counter
billingAgent_count = 1
file_count = 1
file_path_1 = './' + dir_name

# Billing agents loop
for i in billingAgent_name_list:
    print(f'Billing agent {i} ({billingAgent_count} out of {len(billingAgent_name_list)}) (file {file_count} out of {file_nb_to_compute})')   
    billingAgent_count += 1
    
    file_path_2 = file_path_1 + '/' + i + '/'


    # Get the current billing agent keyword (preserve original case)
    concerned_billingAgent_keyword = i
    
    # Find the full billing agent name in the checklist
    cond = (df_checklist_params['Billing_Agent_Keyword'].str.lower() == concerned_billingAgent_keyword.lower())
    if any(cond):
        concerned_billingAgent = df_checklist_params.loc[cond, 'Billing_Agent'].iloc[0]
    else:
        print(f"Billing agent {i} not found in checklist")
        continue
    
    
    # Restricting Bemo dataset to concerned billing agent
    cond = (df_reference_billing['Billing_Agent_Keyword'].str.lower() == concerned_billingAgent_keyword.lower())
    cpo_id_list = list(df_reference_billing['Cpo_Id'][cond].unique())
    
    if len(cpo_id_list) > 0:
        cond = df_bemo['Cpo_Id_bemo'].isin(cpo_id_list)
        df_bemo_billingAgent = df_bemo[cond].copy()
        df_bemo_billingAgent.reset_index(drop=True, inplace=True)

        # Size of the specific file
        bemo_billingAgent_amount_sum = df_bemo_billingAgent[var_price_bemo].sum()
        bemo_billingAgent_count_sum = df_bemo_billingAgent.shape[0]

    if len(cpo_id_list) > 0 and df_bemo_billingAgent.shape[0] > 0:
        # Remove from main Bemo dataframe to avoid duplicate processing
        df_bemo = duplicates_dropper(df_bemo, df_bemo_billingAgent, list(df_bemo.columns))

        # Year loop
        for j in listdir(file_path_2):
            file_path_3 = file_path_2 + j + '/' 
            # Month loop
            for k in listdir(file_path_3):
                file_path_4 = file_path_3 + k + '/'

                # Create log file for billing agent
                billingAgent_log_file_path = file_path_4 + others_folder_name + '/LOG.txt'
                with open(billingAgent_log_file_path, 'w') as f:
                    f.write('#################################################################')
                    f.write('\n')
                    f.write(f"{"Mandataire de facturation : ":<80}{str(concerned_billingAgent):<30}")
                    f.write('\n')
                    f.write('#################################################################')
                    f.write('\n\n')
                    f.write(f"{"Utilisateur :":<80}{str(getlogin())}")
                    f.write('\n')
                    f.write("Version de l'algorithme de réconciliation : " + str(algo_version))
                    f.write('\n')
                    f.write("Réconciliations réalisées le : " + str(datetime.now().strftime('%d/%m/%Y %H:%M:%S'))) 
                    f.write('\n\n')
                    f.write(f"{"Année(s) concernée(s) :":<80}{str(concerned_year)}")
                    f.write('\n')
                    f.write(f"{"Mois concerné(s) :":<80}{str(concerned_month)}")
                    f.write('\n\n')

                # File loop
                for l in listdir(file_path_4):
                    file_path = file_path_4 + l
                    if path.isfile(file_path):
                        try:
                            file_extension = l.split('.')[1].lower()
                        except IndexError:
                            continue  # Skip files without extension
                            
                        if file_extension in [ext.lower() for ext in authorized_extension]:
                            file_count += 1
                            print(f'Processing file {l} ({file_count} out of {file_nb_to_compute})')
                        
                            billingAgent_file_bool, df_billingAgent = generic_file_loader(str(file_path), l, authorized_extension)
                            
                            if billingAgent_file_bool and not df_billingAgent.empty:
                                # Add suffix to column names
                                for m in df_billingAgent.columns:
                                    df_billingAgent.rename(columns={m: str(m + '_billingAgent')}, inplace=True)
                        
                                # Clean data                    
                                df_billingAgent.dropna(how='all', inplace=True)
                                df_billingAgent.drop_duplicates(inplace=True)
                                df_billingAgent.reset_index(drop=True, inplace=True)
                                          
                                # Call main matching algorithm               
                                df_match_bemo, amount_match_bemo, amount_match_billingAgent, amount_unmatch_bemo, amount_unmatch_billingAgent, count_match, count_unmatch_bemo, count_unmatch_billingAgent = matching_algo(
                                    df_checklist_params, df_bemo_billingAgent, df_billingAgent, concerned_billingAgent_keyword, 
                                    concerned_year, concerned_month, file_path_4, l, file_count, 
                                    error_file_path, log_file_path)
                                
                                # Update totals
                                amount_match_bemo_sum += amount_match_bemo
                                amount_match_billingAgent_sum += amount_match_billingAgent
                                amount_unmatch_bemo_sum += amount_unmatch_bemo
                                amount_unmatch_billingAgent_sum += amount_unmatch_billingAgent
                                count_match_sum += count_match
                                count_unmatch_bemo_sum += count_unmatch_bemo
                                count_unmatch_billingAgent_sum += count_unmatch_billingAgent
    
                                # Remove matched records from Bemo data for next iterations
                                if not df_match_bemo.empty:
                                    df_bemo_billingAgent = duplicates_dropper(df_bemo_billingAgent, df_match_bemo, list(df_bemo_billingAgent.columns))
                                
                                # Log results
                                try:
                                    with open(billingAgent_log_file_path, 'a') as f:
                                        f.write(f"{"Fichier :":<80}{str(l)}")
                                        f.write('\n')
                                        f.write('\n')
                                        f.write("Transactions Réconciliées")
                                        f.write('\n')
                                        
                                        # Calculate percentages safely
                                        match_percent = 0
                                        if bemo_billingAgent_count_sum > 0:
                                            match_percent = 100 * count_match / bemo_billingAgent_count_sum
                                            
                                        f.write(f"{"    - Décompte Bemo/Mandataire de facturation : ":<80}{str(round(count_match,0)):<10}{str('#'):<1}{str(' / '):<3}{str(round(bemo_billingAgent_count_sum,0)):<10}{str(' (soit '):<7}{str(round(match_percent,1)):<5}{str(' % des transactions Bemo pour ce mandataire pendant cette période)'):<50}")
                                        f.write('\n')
                                        
                                        # Calculate value percentages safely
                                        value_percent = 0
                                        if bemo_billingAgent_amount_sum > 0:
                                            value_percent = 100 * amount_match_bemo / bemo_billingAgent_amount_sum
                                            
                                        f.write(f"{"    - Valorisation Bemo : ":<80}{str(round(amount_match_bemo,0)):<10}{str('€'):<1}{str(' / '):<3}{str(round(bemo_billingAgent_amount_sum,0)):<10}{str(' (soit '):<7}{str(round(value_percent,1)):<5}{str(' % des transactions Bemo pour ce mandataire pendant cette période)'):<50}")
                                        f.write('\n')
                                        f.write(f"{"    - Valorisation Mandataires de facturation : ":<80}{str(round(amount_match_billingAgent,1)):<10}{str('€'):<1}")
                                        f.write('\n')
                                        f.write('\n')
                                        f.write("Transactions Non Réconciliées")
                                        f.write('\n')
                                        
                                        # Calculate unmatched percentages safely
                                        unmatch_percent = 0
                                        if bemo_billingAgent_count_sum > 0:
                                            unmatch_percent = 100 * count_unmatch_bemo / bemo_billingAgent_count_sum
                                            
                                        f.write(f"{"    - Décompte Bemo : ":<80}{str(round(count_unmatch_bemo,0)):<10}{str('#'):<1}{str(' / '):<3}{str(round(bemo_billingAgent_count_sum,0)):<10}{str(' (soit '):<7}{str(round(unmatch_percent,1)):<5}{str(' % des transactions Bemo pour cette période'):<50}")
                                        f.write('\n')
                                        f.write(f"{"    - Décompte Mandataires de facturation : ":<80}{str(round(count_unmatch_billingAgent,0)):<10}{str('#'):<1}")
                                        f.write('\n')
                                        
                                        # Calculate unmatched value percentages safely
                                        unmatch_value_percent = 0
                                        if bemo_billingAgent_amount_sum > 0:
                                            unmatch_value_percent = 100 * amount_unmatch_bemo / bemo_billingAgent_amount_sum
                                            
                                        f.write(f"{"    - Valorisation Bemo : ":<80}{str(round(amount_unmatch_bemo,0)):<10}{str('€'):<1}{str(' / '):<3}{str(round(total_bemo_amount_sum,0)):<10}{str(' (soit '):<7}{str(round(unmatch_value_percent,1)):<5}{str(' % des transactions Bemo pour ce mandataire pendant cette période)'):<50}")
                                        f.write('\n')
                                        f.write(f"{"    - Valorisation Mandataires de facturation : ":<80}{str(round(amount_unmatch_billingAgent,1)):<10}{str('€'):<1}")
                                        f.write('\n\n')


                                except Exception as e:
                                    print(f"Error writing to log file: {e}")
    
                                # Concatenate results files
                                try:
                                    df_concat_match = file_concatener(file_path_4 + others_folder_name + '/', 'Match')
                                    if not df_concat_match.empty:
                                        df_concat_match.to_csv(file_path_4 + others_folder_name + '/Match.csv', 
                                                              index=True, index_label=False, header=True, 
                                                              encoding='utf-16', sep='\t')
                                
                                    df_concat_unmatch_bemo = file_concatener(file_path_4 + others_folder_name + '/', 'Unmatch_Bemo')
                                    if not df_concat_unmatch_bemo.empty:
                                        df_concat_unmatch_bemo.to_csv(file_path_4 + others_folder_name + '/Unmatch_Bemo.csv', 
                                                                     index=False, index_label=False, header=True, 
                                                                     encoding='utf-16', sep='\t')
                                
                                    df_concat_unmatch_billingAgent = file_concatener(file_path_4 + others_folder_name + '/', 'Unmatch_billingAgent')
                                    if not df_concat_unmatch_billingAgent.empty:
                                        df_concat_unmatch_billingAgent.to_csv(file_path_4 + others_folder_name + '/Unmatch_billingAgent.csv', 
                                                                            index=False, index_label=False, header=True, 
                                                                            encoding='utf-16', sep='\t')
                                except Exception as e:
                                    print(f"Error concatenating results: {e}")
                            else:
                                print(f"Skipping file {l} (not valid or empty)")
                                                                
                        # Update elapsed time in log
                        time_hr, time_mn, time_sec = clock_splitter()
                        try:
                            with open(billingAgent_log_file_path, 'a') as f:
                                f.write("Temps de calcul : " + str(time_hr) + " h " + str(time_mn) + " mn " + str(time_sec) + " s")
                                f.write('\n')
                                f.write(f"{{'-----------------------------------------------------------------':<100}}")
                                f.write('\n\n')

                        except Exception as e:
                            print(f"Error updating log with timing: {e}")

                # Remove empty Results folder if it exists
                results_path = file_path_4 + others_folder_name + '/'
                if path.exists(results_path) and not listdir(results_path):
                    try:
                        chmod(results_path, 0o777)
                        shutil.rmtree(results_path)
                    except Exception as e:
                        print(f"Error removing empty results folder: {e}")
    else:
        # File not taken into account (Billing Agent empty or not registered...)
        print(f"Skipping billing agent {i} (no CPO IDs or no Bemo transactions)")
        # Year loop
        for j in listdir(file_path_2):
            file_path_3 = file_path_2 + j + '/' 
            # Month loop
            for k in listdir(file_path_3):
                file_path_4 = file_path_3 + k + '/'
                # Removing empty Results folder
                if path.exists(file_path_4 + others_folder_name + '/'):
                    try:
                        chmod(file_path_4 + others_folder_name + '/', 0o777)
                        shutil.rmtree(file_path_4 + others_folder_name + '/')
                    except Exception as e:
                        print(f"Error removing results folder: {e}")
                        
                # File loop - increment counter for skipped files
                for l in listdir(file_path_4):
                    file_path = file_path_4 + l
                    if path.isfile(file_path):
                        try:
                            file_extension = l.split('.')[1].lower()
                            if file_extension in [ext.lower() for ext in authorized_extension]:
                                file_count += 1
                        except:
                            pass

# Write final summary log
time_hr, time_mn, time_sec = clock_splitter()
elapsed_time = str(time_hr) + ' h ' + str(time_mn) + ' mn ' + str(time_sec) + ' s'

try:
    with open(log_file_path, 'a') as f:
        f.write("########################################################")
        f.write('\n')
        f.write(f"{"Synthèse des réconciliations de factures des mandataires CPO":<80}")
        f.write('\n')
        f.write("########################################################")
        f.write('\n\n')
        f.write(f"{"Utilisateur :":<80}{str(getlogin())}")
        f.write('\n')
        f.write(f"{"Version de l'algorithme de réconciliation :":<80}{str(algo_version)}")
        f.write('\n')
        f.write(f"{"Réconciliations réalisées le :":<80}{str(datetime.now().strftime('%d/%m/%Y %H:%M:%S'))}")
        f.write('\n')
        f.write(f"{"Temps de calcul :":<80}{str(elapsed_time):<30}")
        f.write('\n\n')
        f.write(f"{"Année concernée :":<80}{str(concerned_year)}")
        f.write('\n')
        f.write(f"{"Mois concerné(s) :":<80}{str(concerned_month)}")
        f.write('\n')
        f.write(f"{"Nombre de mandataires explorés (voir détail ci-dessous) :":<80}{str(len(billingAgent_name_list))}")
        f.write('\n\n')
        f.write("Transactions Réconciliées") 
        f.write('\n')
        
        # Calculate match percentage
        match_percent = 0
        if total_bemo_count_sum > 0:
            match_percent = 100 * count_match_sum / total_bemo_count_sum
            
        f.write(f"{"    - Décompte : ":<80}{str(round(count_match_sum,0)):<10}{str('#'):<1}{str(' / '):<3}{str(round(total_bemo_count_sum,0)):<10}{str(' (soit '):<7}{str(round(match_percent,1)):<5}{str(' % des transactions Bemo pour cette période)'):<50}")
        f.write('\n')
        
        # Calculate match value percentage
        match_value_percent = 0
        if total_bemo_amount_sum > 0:
            match_value_percent = 100 * amount_match_bemo_sum / total_bemo_amount_sum
            
        f.write(f"{"    - Valorisation Bemo : ":<80}{str(round(amount_match_bemo_sum,0)):<10}{str('€'):<1}{str(' / '):<3}{str(round(total_bemo_amount_sum,0)):<10}{str(' (soit '):<7}{str(round(match_value_percent,1)):<5}{str(' % des transactions Bemo pour cette période)'):<50}")
        f.write('\n')
        f.write(f"{"    - Valorisation Mandataires de facturation : ":<80}{str(round(amount_match_billingAgent_sum,1)):<10}{str('€'):<1}")
        f.write('\n\n')
        f.write("Transactions Non Réconciliées")
        f.write('\n')
        
        # Calculate unmatch percentage
        unmatch_percent = 0
        if total_bemo_count_sum > 0:
            unmatch_percent = 100 * count_unmatch_bemo_sum / total_bemo_count_sum
            
        f.write(f"{"    - Décompte Bemo : ":<80}{str(round(count_unmatch_bemo_sum,0)):<10}{str('#'):<1}{str(' / '):<3}{str(round(total_bemo_count_sum,0)):<10}{str(' (soit '):<7}{str(round(unmatch_percent,1)):<5}{str(' % des transactions Bemo pour cette période)'):<50}")
        f.write('\n')    
        f.write(f"{"    - Décompte Mandataires de facturation : ":<80}{str(round(count_unmatch_billingAgent_sum,0)):<10}{str('#'):<1}")
        f.write('\n')
        
        # Calculate unmatch value percentage
        unmatch_value_percent = 0
        if total_bemo_amount_sum > 0:
            unmatch_value_percent = 100 * amount_unmatch_bemo_sum / total_bemo_amount_sum
            
        f.write(f"{"    - Valorisation Bemo : ":<80}{str(round(amount_unmatch_bemo_sum,0)):<10}{str('€'):<1}{str(' / '):<3}{str(round(total_bemo_amount_sum,0)):<10}{str(' (soit '):<7}{str(round(unmatch_value_percent,1)):<5}{str(' % des transactions Bemo pour cette période)'):<50}")
        f.write('\n')
        f.write(f"{"    - Valorisation Mandataires de facturation : ":<80}{str(round(amount_unmatch_billingAgent_sum,1)):<10}{str('€'):<1}")
        f.write('\n\n')
        f.write("Mandataires de facturation explorés")
        f.write('\n')
        for i in billingAgent_name_list:
            f.write("    - " + str(i))
            f.write('\n')
except Exception as e:
    print(f"Error writing final summary log: {e}")
        
# Remove error file if no errors were logged
if path.exists(error_file_path) and path.getsize(error_file_path) == error_file_path_len_1:
    try:
        remove(error_file_path)
    except Exception as e:
        print(f"Error removing empty error file: {e}")

print('Reconciliation completed successfully!')
