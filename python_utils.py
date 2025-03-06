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