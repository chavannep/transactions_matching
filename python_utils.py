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


# Authorized file extensions
authorized_extension = ['csv', 'xlsx', 'CSV', 'XLSX']
authorized_extraction = ['.tar', '.zip', '.TAR', '.ZIP']


# # Subroutines

def remove_readonly(func, path, excinfo):
    chmod(path, stat.S_IWRITE)
    func(path)
    
def col_norm(s):
    s = s.astype(dtype='str', copy=True, errors='ignore')
    s = s.str.replace('*','')
    s = s.str.replace('-','')
    s = s.str.replace('_','')
    s = s.str.replace(' ','')
    s = s.str.lower()
    return s

def list_update_suffix(list_params, df, suffix):
    for elem in range(len(list_params)):
        list_params[elem] = str(list_params[elem]) + suffix
    list_params_tmp = []        
    for elem in list_params:
        if(elem in list(df.columns)):
            list_params_tmp.append(elem)
    list_params = list_params_tmp
    for elem in list_params:
        df[elem] = col_norm(df[elem])
    return list_params
    
def remove_non_numeric(text):
    return sub("[^0-9.,; ]", "", text)

def float_filt(s):
    try:
        s = s.astype(dtype='str', copy=True, errors='ignore')
        # Remove currency symbols
        s = s.str.replace('€', '')
        s = s.str.replace('EUR', '')
        s = s.str.replace(' ', '')
        s = s.str.replace(',', '.')
        s = s.str.replace(';', '.')
        s = pd.to_numeric(s, errors='coerce')
        s = abs(s)
        return s
    except Exception as e:
        # In case of error, return the original series
        print(f"Error in float_filt: {e}")
        return s

def datetime_filt(s):
    bool_datetime = False
    try:
        s = pd.to_datetime(s, infer_datetime_format=True, errors='coerce')
        s = s.dt.tz_localize(tz=None, ambiguous='infer', nonexistent='NaT')
        bool_datetime = True
    except Exception as e:
        # print(f"Error occurred: {e}")
        pass
    return (s, bool_datetime)

def file_concatener(path_file_to_concat, keyword):
    loop_count = 0
    df_concat = pd.DataFrame({'Col':[]})
    for file_name in listdir(path_file_to_concat):
        if(keyword in file_name):
            try:
                # Specified encoder internally used in this notebook
                df_result = pd.read_csv(path_file_to_concat + file_name, encoding='utf-16', sep='\t')
                if(loop_count == 0):
                    df_concat = df_result
                else:
                    df_concat = pd.concat([df_concat, df_result], join='outer', ignore_index=True)
                loop_count += 1
                remove(path_file_to_concat + file_name)
            except Exception as e:
                print(f"Error concatenating file {file_name}: {e}")
    df_concat.reset_index(drop=True, inplace=True)
    return df_concat

def duplicates_dropper(df_init, df_sub, columns_to_keep):
    """
    Improved version of duplicates_dropper that more reliably identifies and removes
    records that are present in both dataframes.
    """
    df_init.reset_index(drop=True, inplace=True)
    df_sub.reset_index(drop=True, inplace=True)
    
    # If df_sub is empty, just return df_init
    if df_sub.empty:
        return df_init
        
    # Check if df_sub might be a subset of df_init
    if df_init.shape[0] >= df_sub.shape[0]:
        # Find common string columns for comparison
        common_columns = list(set(df_init.select_dtypes(include=['object']).columns).intersection(
                             set(df_sub.select_dtypes(include=['object']).columns)))
        
        # If we have common columns, check for duplicates
        if common_columns:
            # Create a DataFrame with only the rows from df_init that don't match records in df_sub
            # First, create a key from common columns
            df_init['temp_key'] = df_init[common_columns].apply(lambda x: tuple(x), axis=1)
            df_sub['temp_key'] = df_sub[common_columns].apply(lambda x: tuple(x), axis=1)
            
            # Now filter out rows in df_init that have keys in df_sub
            df_fin = df_init[~df_init['temp_key'].isin(df_sub['temp_key'])]
            
            # Drop the temporary key column
            df_fin.drop('temp_key', axis=1, inplace=True)
            df_init.drop('temp_key', axis=1, inplace=True)
            df_sub.drop('temp_key', axis=1, inplace=True)
            
            # Keep only requested columns
            if not df_fin.empty:
                df_fin = df_fin[columns_to_keep]
            else:
                df_fin = pd.DataFrame(columns=columns_to_keep)
        else:
            # If no common columns, just return df_init
            df_fin = df_init
    else:
        df_fin = df_init
        
    df_fin.reset_index(drop=True, inplace=True)
    return df_fin

def params_finder(entity, df_checklist_params, var):
    cond = (df_checklist_params['Billing_Agent'] == entity)
    list_params = []
    try:
        for i in range(1, 11):
            test_var = var + str(i)
            if(test_var in df_checklist_params.columns):
                if isinstance(df_checklist_params[test_var][cond].iloc[0], str):
                    list_params.append(df_checklist_params[test_var][cond].iloc[0])
    except Exception as e:
        print(f"Error in params_finder for Billing Agent {entity}: {e}")
    return list_params


# Best merger function
def merge_tester(df1, df2, list_var_df1, list_var_df2):
    best_score = 0
    best_var1 = ''
    best_var2 = ''
    for i in list_var_df1:
        for j in list_var_df2:
            try:
                # Ensure that i and j are recognized as str in df1 and df2
                df1[i] = df1[i].astype(dtype='str', copy=True, errors='ignore')
                df2[j] = df2[j].astype(dtype='str', copy=True, errors='ignore')
                df3 = pd.merge(df1, df2, left_on=i, right_on=j, how='inner')
                # More clever than just df3.shape[0]
                cond = (df3[i].notnull())
                score = df3[cond].shape[0]
                # because this score does not take np.nan match
                # score = df3.shape[0]
                # print('Match', score)
            except Exception as e:
                # print(f"Error occurred: {e}")
                score = 0
                # print('Error')

            if ((score > best_score) and (pd.isnull(df3[i]).all()==False) and (pd.isnull(df3[j]).all()==False)):
            # # Otherwise : np.nan similarity causes merge dysfunction...
            # if( (score > best_score) ):
                # print('Best Match', score)
                best_score = score
                best_var1 = i
                best_var2 = j
    return(best_var1, best_var2)

# Detecting dates of interest for transactions
def date_detector(dir_path):
    concerned_year = []
    concerned_month = []
    match = findall('[0-9]+', dir_path)
    for elem in match:
        if((len(elem)>=1) & (len(elem)<=2)):
            concerned_month.append(elem)
        elif(len(elem)==4):
            concerned_year.append(elem)    
    return(concerned_year, concerned_month)
    

def generic_file_loader(file_path, file_name, authorized_extension):
    file_bool = False
    df_file = pd.DataFrame({'Col':[]})
    
    try:
        file_extension = file_name.split('.')[1].lower()
    except IndexError:
        print(f"Error: File {file_name} does not have an extension")
        return False, df_file
        
    if(file_extension in authorized_extension):       
        if(file_extension.lower() == 'xlsx'):
            try:
                # df_file = pd.read_excel(io=file_path, sheet_name=None)
                # if(len(df_cpo.keys())>1):
                #     print('Multiple worksheets detected')
                file_explo = False
                cpt_header=0
                while not (file_explo):
                    df_file = pd.read_excel(io=file_path, header=cpt_header, sheet_name=0)
                    if(df_file.columns.str.contains('Unnamed').sum()>2):
                        cpt_header+=1
                        # Scan first 10 rows of the file to detect a readable header
                        if(cpt_header==10):
                            break
                    else:
                        file_explo = True
                        file_bool = True
            except Exception as e:
                print(f"Error loading Excel file {file_name}: {e}")
                # raise SystemExit()
                pass
        
        elif(file_extension.lower() == 'csv'):
            class EncodingError(Exception):
                pass 
            
            # Try different encodings in order
            encodings = ['utf-16', 'utf-8', 'utf-8-sig', 'iso-8859-1', None]
            
            for encoding in encodings:
                try:
                    df_file = pd.read_csv(file_path, encoding=encoding, sep=None)
                    
                    # Check for BOM character
                    for col in list(df_file.columns):
                        if '\ufeff' in col:
                            raise EncodingError
                    
                    # Fix header if needed
                    if(df_file.columns.str.contains('Unnamed').sum()>2):
                        df_file = df_file.dropna(how='all')
                        df_file.columns = df_file.iloc[0]
                        df_file = df_file[1:].reset_index(drop=True)
                        
                    file_bool = True
                    break  # Success, exit the loop
                    
                except Exception as e:
                    # Try next encoding
                    continue
    else:
        print(f"File extension error for {file_name}: {file_extension} not in {authorized_extension}")
        
    return(file_bool, df_file)


def specific_file_loader(file_path, file_name, authorized_extension):
    file_bool = False
    df_file = pd.DataFrame({'Col':[]})
    
    try:
        file_extension = file_name.split('.')[1].lower()
    except IndexError:
        print(f"Error: File {file_name} does not have an extension")
        return False, df_file
        
    if(file_extension in authorized_extension):              
        if(file_extension.lower() == 'csv'):
            class EncodingError(Exception):
                pass 
            try:
                df_file = pd.read_csv(file_path, encoding='utf-8', sep=';')
                for col in list(df_file.columns):
                    if '\ufeff' in col:
                        raise EncodingError
                file_bool = True
            except Exception as e:
                print(f"Error loading CSV file {file_name}: {e}")
                # Try alternative encodings
                try:
                    df_file = pd.read_csv(file_path, encoding='utf-8-sig', sep=';')
                    file_bool = True
                except Exception as e:
                    print(f"Alternative encoding also failed for {file_name}: {e}")
    else:
        print(f"File extension error for {file_name}")
        
    return(file_bool, df_file)


#########################
# Main algorithm for matching operation
#########################

def matching_algo(df_checklist_params_dev, df_bemo_billingAgent_dev, df_billingAgent_dev, concerned_billingAgent_keyword_dev, concerned_year_dev, concerned_month_dev, file_path_dev, file_name_dev, file_count_dev, error_file_path_dev, log_file_path_dev):
    """
    Algorithm to match transactions between Bemo and billing agent data
    """
    # Create copies to avoid modifying originals
    df_bemo_billingAgent = df_bemo_billingAgent_dev.copy()
    df_billingAgent = df_billingAgent_dev.copy()
    
    #########################
    # Data featuring
    #########################
    
    ## Billing agent : loading variables

    # Start Date
    cond = (df_checklist_params['Billing_Agent_Keyword'].str.lower() == concerned_billingAgent_keyword_dev.lower())
    var_startdate_billingAgent = None
    if ('StartDate_param' in df_checklist_params.columns and
        any(cond) and
        isinstance(df_checklist_params.loc[cond, 'StartDate_param'].iloc[0], str)):
        var_startdate_billingAgent = str(df_checklist_params.loc[cond, 'StartDate_param'].iloc[0]) + '_billingAgent'

    # Quantity
    var_quantity_billingAgent = None
    if ('Quantity_param' in df_checklist_params.columns and
        any(cond) and
        isinstance(df_checklist_params.loc[cond, 'Quantity_param'].iloc[0], str)):
        var_quantity_billingAgent = str(df_checklist_params.loc[cond, 'Quantity_param'].iloc[0]) + '_billingAgent'

    # Price
    var_price_billingAgent = None
    if ('Price_param' in df_checklist_params.columns and
        any(cond) and
        isinstance(df_checklist_params.loc[cond, 'Price_param'].iloc[0], str)):
        var_price_billingAgent = str(df_checklist_params.loc[cond, 'Price_param'].iloc[0]) + '_billingAgent'
        
    ## Identifying best variables for CDR merge
    list_params_cdr_bemo = params_finder('Bemo', df_checklist_params, 'CDR_param_')
    list_params_cdr_bemo = list_update_suffix(list_params_cdr_bemo, df_bemo_billingAgent, '_bemo')

    list_params_cdr_billingAgent = params_finder(concerned_billingAgent_keyword_dev, df_checklist_params, 'CDR_param_')
    list_params_cdr_billingAgent = list_update_suffix(list_params_cdr_billingAgent, df_billingAgent, '_billingAgent')

    best_var_cdr_bemo, best_var_cdr_billingAgent = merge_tester(df_bemo_billingAgent, df_billingAgent, list_params_cdr_bemo, list_params_cdr_billingAgent)

    ## Identifying best variables for RFID selection    
    list_params_rfid_bemo = params_finder('Bemo', df_checklist_params, 'RFID_param_')
    list_params_rfid_bemo = list_update_suffix(list_params_rfid_bemo, df_bemo_billingAgent, '_bemo')

    list_params_rfid_billingAgent = params_finder(concerned_billingAgent_keyword_dev, df_checklist_params, 'RFID_param_')
    list_params_rfid_billingAgent = list_update_suffix(list_params_rfid_billingAgent, df_billingAgent, '_billingAgent')

    best_var_rfid_bemo, best_var_rfid_billingAgent = merge_tester(df_bemo_billingAgent, df_billingAgent, list_params_rfid_bemo, list_params_rfid_billingAgent)
    
    ## Identifying best variables for EVSE selection
    list_params_evse_bemo = params_finder('Bemo', df_checklist_params, 'EVSE_param_')
    list_params_evse_bemo = list_update_suffix(list_params_evse_bemo, df_bemo_billingAgent, '_bemo')

    list_params_evse_billingAgent = params_finder(concerned_billingAgent_keyword_dev, df_checklist_params, 'EVSE_param_')
    list_params_evse_billingAgent = list_update_suffix(list_params_evse_billingAgent, df_billingAgent, '_billingAgent')

    best_var_evse_bemo, best_var_evse_billingAgent = merge_tester(df_bemo_billingAgent, df_billingAgent, list_params_evse_bemo, list_params_evse_billingAgent)

    # Initialize match tracking column
    df_bemo_billingAgent['Match'] = 0
    
    # Check if required columns exist
    if not var_startdate_billingAgent or not var_price_billingAgent or not best_var_evse_billingAgent:
        print(f"Missing required columns for billing agent {concerned_billingAgent_keyword_dev}")
        with open(error_file_path_dev, 'a') as f:
            f.write(f"\nMissing required columns for billing agent {concerned_billingAgent_keyword_dev}\n")

        
        # Initialize empty result dataframes
        df_match = pd.DataFrame(columns=df_bemo_billingAgent.columns)
        df_unmatch_bemo = df_bemo_billingAgent.drop('Match', axis=1, errors='ignore')
        df_unmatch_billingAgent = df_billingAgent.copy()
        
        return (df_match, 0, 0, df_unmatch_bemo[var_price_bemo].sum(), 0, 0,
                df_unmatch_bemo.shape[0], df_unmatch_billingAgent.shape[0])
    
    # Start date in local time
    bool_datetime_billingAgent = False
    if var_startdate_billingAgent in df_billingAgent.columns:
        df_billingAgent[var_startdate_billingAgent], bool_datetime_billingAgent = datetime_filt(df_billingAgent[var_startdate_billingAgent])
    
    # Check if we have the minimum required parameters
    minimal_params_blocks_bool = (var_startdate_billingAgent in df_billingAgent.columns and
                                 var_price_billingAgent in df_billingAgent.columns and
                                 best_var_evse_billingAgent in df_billingAgent.columns and
                                 bool_datetime_billingAgent)
    
    # Initialize for quantity conversion tracking
    quantity_bemo_test_quantile = 0
    quantity_billingAgent_test_quantile = 0
            
    # Initialize match block dataframes
    df_match_block_1 = pd.DataFrame()
    df_match_block_2 = pd.DataFrame()
    
    if minimal_params_blocks_bool:
        # Format data
        # Price  
        df_billingAgent[var_price_billingAgent] = float_filt(df_billingAgent[var_price_billingAgent])
    
        # Restrict dates
        cond = ((df_billingAgent[var_startdate_billingAgent].dt.year.isin(concerned_year_dev)) &
                (df_billingAgent[var_startdate_billingAgent].dt.month.isin(concerned_month_dev)))
        df_billingAgent = df_billingAgent[cond].copy()
        df_billingAgent.reset_index(drop=True, inplace=True)
        
        #########################
        # Block 1 : direct merge on CDR id
        #########################
        minimal_params_block1_bool = (best_var_cdr_billingAgent in df_billingAgent.columns and
                                     best_var_cdr_bemo != '' and best_var_cdr_billingAgent != '')
                                     
        if minimal_params_block1_bool:
            # Dropping CDR duplicates
            df_bemo_billingAgent.sort_values(by=var_price_bemo, ascending=False, inplace=True)
            df_bemo_billingAgent.drop_duplicates(subset=best_var_cdr_bemo, keep='first', inplace=True)
            df_bemo_billingAgent.reset_index(drop=True, inplace=True)
            df_bemo_billingAgent.sort_values(by=var_startdate_bemo, ascending=True, inplace=True)
            df_bemo_billingAgent.reset_index(drop=True, inplace=True)
    
            df_billingAgent.sort_values(by=var_price_billingAgent, ascending=False, inplace=True)
            df_billingAgent.drop_duplicates(subset=best_var_cdr_billingAgent, keep='first', inplace=True)
            df_billingAgent.reset_index(drop=True, inplace=True)
            df_billingAgent.sort_values(by=var_startdate_billingAgent, ascending=True, inplace=True)
            df_billingAgent.reset_index(drop=True, inplace=True)
        
            # CDR merge algo
            df_bemo_billingAgent[best_var_cdr_bemo] = df_bemo_billingAgent[best_var_cdr_bemo].astype(dtype='str', copy=True, errors='ignore')
            df_billingAgent[best_var_cdr_billingAgent] = df_billingAgent[best_var_cdr_billingAgent].astype(dtype='str', copy=True, errors='ignore')
            
            # Filter out NaN values before merging
            bemo_cdr_valid = df_bemo_billingAgent[df_bemo_billingAgent[best_var_cdr_bemo].notna()]
            billing_cdr_valid = df_billingAgent[df_billingAgent[best_var_cdr_billingAgent].notna()]
            
            df_match_block_1 = pd.merge(bemo_cdr_valid, billing_cdr_valid,
                                        left_on=best_var_cdr_bemo,
                                        right_on=best_var_cdr_billingAgent,
                                        how='inner')
            
            # Mark matched records
            df_match_block_1['Match'] = 1
            
            # Remove matched records from main dataframes
            if not df_match_block_1.empty:
                # Create list of CDR values that were matched
                matched_cdrs = df_match_block_1[best_var_cdr_bemo].unique()
                
                # Filter out matched records from main dataframes
                df_bemo_billingAgent = df_bemo_billingAgent[~df_bemo_billingAgent[best_var_cdr_bemo].isin(matched_cdrs)].copy()
                df_billingAgent = df_billingAgent[~df_billingAgent[best_var_cdr_billingAgent].isin(matched_cdrs)].copy()
                
                # Reset indices
                df_bemo_billingAgent.reset_index(drop=True, inplace=True)
                df_billingAgent.reset_index(drop=True, inplace=True)

        #########################
        # Block 2 : indirect merge with minimum gap
        #########################
        minimal_params_block2_bool = (best_var_rfid_billingAgent in df_billingAgent.columns and
                                      var_quantity_billingAgent in df_billingAgent.columns and
                                      best_var_rfid_bemo != '' and best_var_rfid_billingAgent != '' and
                                      best_var_evse_bemo != '' and best_var_evse_billingAgent != '')
                                      
        if minimal_params_block2_bool:
            # Quantity   
            df_billingAgent[var_quantity_billingAgent] = float_filt(df_billingAgent[var_quantity_billingAgent])

            # Quantity conversion test
            test_quantile = 0.5
            cond = df_billingAgent[var_quantity_billingAgent].isna()
            if not df_billingAgent[~cond].empty:
                quantity_billingAgent_test_quantile = df_billingAgent[var_quantity_billingAgent][~cond].quantile([test_quantile])[test_quantile]
                
                # Convert if needed for consistency
                if quantity_billingAgent_test_quantile > 100:
                    df_billingAgent[var_quantity_billingAgent] = df_billingAgent[var_quantity_billingAgent] / 1000
                
            # Prepare for min gap algo
            df_bemo_billingAgent[best_var_evse_bemo] = df_bemo_billingAgent[best_var_evse_bemo].astype(dtype='str', copy=True, errors='ignore')
            df_bemo_billingAgent[best_var_rfid_bemo] = df_bemo_billingAgent[best_var_rfid_bemo].astype(dtype='str', copy=True, errors='ignore')
            df_billingAgent[best_var_evse_billingAgent] = df_billingAgent[best_var_evse_billingAgent].astype(dtype='str', copy=True, errors='ignore')
            df_billingAgent[best_var_rfid_billingAgent] = df_billingAgent[best_var_rfid_billingAgent].astype(dtype='str', copy=True, errors='ignore')
            
            # Initialize dataframes
            columns_name = list(df_bemo_billingAgent.columns) + list(df_billingAgent.columns)
            df_match_block_2 = pd.DataFrame(columns=columns_name)
            df_unmatch_bemo_temp = pd.DataFrame(columns=df_bemo_billingAgent.columns)
            
            # Treating remaining Bemo transactions by decreasing amount of money
            df_bemo_billingAgent.sort_values(by=var_price_bemo, ascending=False, na_position='last', inplace=True)
            df_bemo_billingAgent.reset_index(drop=True, inplace=True)
        
            for index_bemo in df_bemo_billingAgent.index:
                # Skip NaN or missing values
                if (pd.isna(df_bemo_billingAgent.at[index_bemo, best_var_rfid_bemo]) or
                    pd.isna(df_bemo_billingAgent.at[index_bemo, best_var_evse_bemo])):
                    new_row = pd.DataFrame([df_bemo_billingAgent.loc[index_bemo]])
                    df_unmatch_bemo_temp = pd.concat([df_unmatch_bemo_temp, new_row], ignore_index=True)
                    continue
                    
                # Filter for potential matches
                cond_same_rfid_evse = ((df_billingAgent[best_var_rfid_billingAgent] == df_bemo_billingAgent.at[index_bemo, best_var_rfid_bemo]) &
                                      (df_billingAgent[best_var_evse_billingAgent] == df_bemo_billingAgent.at[index_bemo, best_var_evse_bemo]))
                
                # Skip if no potential matches
                billing_matches = df_billingAgent[cond_same_rfid_evse]
                if billing_matches.empty:
                    new_row = pd.DataFrame([df_bemo_billingAgent.loc[index_bemo]])
                    df_unmatch_bemo_temp = pd.concat([df_unmatch_bemo_temp, new_row], ignore_index=True)
                    continue
                
                # Calculate match score for each potential match
                billing_matches = billing_matches.copy()
                billing_matches['RMSE'] = np.nan
                
                for index_billingAgent in billing_matches.index:
                    try:
                        # Calculate gaps in date and quantity
                        gap_startdate = pd.Timedelta(
                            abs(billing_matches.at[index_billingAgent, var_startdate_billingAgent] - 
                                df_bemo_billingAgent.at[index_bemo, var_startdate_bemo]),
                            unit='days') / pd.Timedelta(hours=1)
                            
                        gap_quantity = abs(
                            billing_matches.at[index_billingAgent, var_quantity_billingAgent] -
                            df_bemo_billingAgent.at[index_bemo, var_quantity_bemo])
                       
                        # If within thresholds, calculate RMSE
                        if gap_startdate <= date_margin and gap_quantity <= quantity_margin:
                            billing_matches.at[index_billingAgent, 'RMSE'] = np.sqrt(gap_startdate**2 + gap_quantity**2)
                    except Exception as e:
                        print(f"Error calculating match score: {e}")
                        continue
                
                # Find best match
                billing_matches.sort_values(by='RMSE', ascending=True, na_position='last', inplace=True)
                
                # If we found a match
                if not billing_matches.empty and not pd.isna(billing_matches['RMSE'].iloc[0]):
                    best_match = billing_matches.iloc[0]
                    
                    # Create a row for the matched pair
                    new_row_data = {}
                    
                    # Add Bemo data
                    for col in df_bemo_billingAgent.columns:
                        new_row_data[col] = df_bemo_billingAgent.at[index_bemo, col]
                        
                    # Add billing agent data
                    for col in df_billingAgent.columns:
                        new_row_data[col] = best_match[col]
                    
                    # Create dataframe and add to matches
                    new_row = pd.DataFrame([new_row_data])
                    df_match_block_2 = pd.concat([df_match_block_2, new_row], ignore_index=True)
                    
                    # Remove matched record from billing agent dataframe
                    best_match_index = best_match.name
                    df_billingAgent = df_billingAgent[df_billingAgent.index != best_match_index].copy()
                else:
                    # No match found
                    new_row = pd.DataFrame([df_bemo_billingAgent.loc[index_bemo]])
                    df_unmatch_bemo_temp = pd.concat([df_unmatch_bemo_temp, new_row], ignore_index=True)
            
            # End of block 2
            if not df_match_block_2.empty:
                df_match_block_2['Match'] = 2
                df_match_block_2.reset_index(drop=True, inplace=True)
                
            # Any unmatched Bemo records
            df_unmatch_bemo_temp.reset_index(drop=True, inplace=True)
        
        # If block 2 wasn't executed, all remaining Bemo records are unmatched
        else:
            df_unmatch_bemo_temp = df_bemo_billingAgent.copy()
    
    else:
        # We don't have required columns, so all records are unmatched
        df_unmatch_bemo_temp = df_bemo_billingAgent.copy()
        
    # Remaining billing agent records are unmatched
    df_unmatch_billingAgent = df_billingAgent.copy()
    
    #########################
    # Concatenating matched results
    #########################
    # Get sizes of match blocks
    df_match_block_1_shape = df_match_block_1.shape[0] if not df_match_block_1.empty else 0
    df_match_block_2_shape = df_match_block_2.shape[0] if not df_match_block_2.empty else 0

    # Back to original format for quantity if conversion was applied
    if quantity_bemo_test_quantile > 100:
        if df_match_block_1_shape > 0:
            df_match_block_1[var_quantity_bemo] = 1000 * df_match_block_1[var_quantity_bemo]
        if df_match_block_2_shape > 0:
            df_match_block_2[var_quantity_bemo] = 1000 * df_match_block_2[var_quantity_bemo]
        if not df_unmatch_bemo_temp.empty:
            df_unmatch_bemo_temp[var_quantity_bemo] = 1000 * df_unmatch_bemo_temp[var_quantity_bemo]
    
    if quantity_billingAgent_test_quantile > 100:
        if df_match_block_1_shape > 0:
            df_match_block_1[var_quantity_billingAgent] = 1000 * df_match_block_1[var_quantity_billingAgent]
        if df_match_block_2_shape > 0:
            df_match_block_2[var_quantity_billingAgent] = 1000 * df_match_block_2[var_quantity_billingAgent]
        if not df_unmatch_billingAgent.empty:
            df_unmatch_billingAgent[var_quantity_billingAgent] = 1000 * df_unmatch_billingAgent[var_quantity_billingAgent]

    # Combine match blocks
    if df_match_block_1_shape == 0 and df_match_block_2_shape == 0:
        # No matches
        df_match = pd.DataFrame(columns=df_bemo_billingAgent.columns)
        match_bool = False
    elif df_match_block_1_shape > 0 and df_match_block_2_shape == 0:
        # Only block 1 matches
        df_match = df_match_block_1
        match_bool = True
    elif df_match_block_1_shape == 0 and df_match_block_2_shape > 0:
        # Only block 2 matches
        df_match = df_match_block_2
        match_bool = True
    else:
        # Both blocks have matches
        df_match = pd.concat([df_match_block_1, df_match_block_2], ignore_index=True)
        match_bool = True

    # Add price delta for matched records
    if match_bool:
        df_match.sort_values(by=var_price_bemo, ascending=False, inplace=True)
        df_match.reset_index(drop=True, inplace=True)  
        df_match['Delta_price'] = df_match[var_price_billingAgent] - df_match[var_price_bemo]

    # Clean up unmatched records
    if 'Match' in df_unmatch_bemo_temp.columns:
        df_unmatch_bemo = df_unmatch_bemo_temp.drop('Match', axis=1)
    else:
        df_unmatch_bemo = df_unmatch_bemo_temp.copy()
        
    df_unmatch_bemo.sort_values(by=var_price_bemo, ascending=False, inplace=True)
    df_unmatch_bemo.reset_index(drop=True, inplace=True) 
    
    df_unmatch_billingAgent.sort_values(by=var_price_billingAgent, ascending=False, inplace=True)
    df_unmatch_billingAgent.reset_index(drop=True, inplace=True)

    # Calculate summary statistics
    if match_bool:
        amount_match_bemo_dev = df_match[var_price_bemo].sum()
        amount_match_billingAgent_dev = df_match[var_price_billingAgent].sum()
        count_match_dev = df_match.shape[0]
    else:
        amount_match_bemo_dev = 0
        amount_match_billingAgent_dev = 0
        count_match_dev = 0

    amount_unmatch_bemo_dev = df_unmatch_bemo[var_price_bemo].sum()

    if minimal_params_blocks_bool:
        df_unmatch_billingAgent[var_price_billingAgent] = float_filt(df_unmatch_billingAgent[var_price_billingAgent])
        amount_unmatch_billingAgent_dev = df_unmatch_billingAgent[var_price_billingAgent].sum()
    else:
        amount_unmatch_billingAgent_dev = 0
        
    count_unmatch_bemo_dev = df_unmatch_bemo.shape[0]
    count_unmatch_billingAgent_dev = df_unmatch_billingAgent.shape[0]

    # Export results to files
    try:
        output_dir = file_path_dev + others_folder_name + '/'
        os.makedirs(output_dir, exist_ok=True)
        
        if df_match.shape[0] > 0:
            df_match.to_csv(output_dir + str(file_count_dev) + '_Match_' + file_name_dev + '.csv', 
                           index=False, index_label=False, header=True, encoding='utf-16', sep='\t')
                           
        if df_unmatch_bemo.shape[0] > 0:
            df_unmatch_bemo.to_csv(output_dir + str(file_count_dev) + '_Unmatch_Bemo_' + file_name_dev + '.csv',
                                  index=False, index_label=False, header=True, encoding='utf-16', sep='\t')
                                  
        if df_unmatch_billingAgent.shape[0] > 0:
            df_unmatch_billingAgent.to_csv(output_dir + str(file_count_dev) + '_Unmatch_billingAgent_' + file_name_dev + '.csv',
                                          index=False, index_label=False, header=True, encoding='utf-16', sep='\t')
    except Exception as e:
        print(f"Error exporting results: {e}")
        with open(error_file_path_dev, 'a') as f:
            f.write(f"\nError exporting results for {file_name_dev}: {str(e)}\n")
        
    return(df_match, amount_match_bemo_dev, amount_match_billingAgent_dev, amount_unmatch_bemo_dev, 
           amount_unmatch_billingAgent_dev, count_match_dev, count_unmatch_bemo_dev, count_unmatch_billingAgent_dev)