from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import logging
import tensorflow as tf
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.spatial.distance import hamming

#import excel file
import pandas as pd

#select excel file
from tkinter import filedialog,Tk

#Removing Initial Cluster
import os

def menu(ExcelName):
    data = str(input('Select datasets cost or imt2020: '))
    print()
    if data == 'cost':
        ExcelName = 'Ex'
        print('1 - 01_Indoor_B1_LOS_Single_Results')
        print('2 - 02_Indoor_B2_LOS_Single_Results')
        print('3 - 03_SemiUrban_B1_LOS_Single_Results')
        print('4 - 04_SemiUrban_B2_LOS_Single_Results')
        print('5 - 05_SemiUrban_B1_NLOS_Single_Results')
        print('6 - 06_SemiUrban_B2_NLOS_Single_Results')
        print('7 - 07_SemiUrban_B1_LOS_Multiple_Results')
        print('8 - 08_SemiUrban_B2_LOS_Multiple_Results')
        print()
        option = int(input('Select a number for the filename: '))
        while option !=999:
            if option == 1:
                ExcelName = '01_Indoor_B1_LOS_Single_Results'
                print(ExcelName+'\n')
                return ExcelName
            elif option == 2:
                ExcelName = '02_Indoor_B2_LOS_Single_Results'
                print(ExcelName+'\n')
                return ExcelName
            elif option == 3:
                ExcelName = '03_SemiUrban_B1_LOS_Single_Results'
                print(ExcelName+'\n')
                return ExcelName
            elif option == 4:
                ExcelName = '04_SemiUrban_B2_LOS_Single_Results'
                print(ExcelName+'\n')
                return ExcelName
            elif option == 5:
                ExcelName = '05_SemiUrban_B1_NLOS_Single_Results'
                print(ExcelName+'\n')
                return ExcelName
            elif option == 6:
                ExcelName = '06_SemiUrban_B2_NLOS_Single_Results'
                print(ExcelName+'\n')
                return ExcelName
            elif option == 7:
                ExcelName = '07_SemiUrban_B1_LOS_Multiple_Results'
                print(ExcelName+'\n')
                return ExcelName
            elif option == 8:
                ExcelName = '08_SemiUrban_B2_LOS_Multiple_Results'
                print(ExcelName+'\n')
                return ExcelName
            else:
                print('Invalid Option.')
                print()
                menu(ExcelName)
                option = int(input('Select a number for the filename: '))
                print()
    
    elif data == 'imt2020':
            ExcelName = 'Ex'
            print('1 - 01_InH_A_LOS_Results')
            print('2 - 02_InH_A_NLOS_Results')
            print('3 - 03_RMa_A_LOS_Results')
            print('4 - 04_RMa_A_NLOS_Results')
            print('5 - 05_UMa_A_LOS_Results')
            print('6 - 06_UMa_A_NLOS_Results')
            print('7 - 07_UMi_A_LOS_Results')
            print('8 - 08_UMi_A_NLOS_Results')
            print()
            option = int(input('Select a number for the filename: '))
            while option !=999:
                if option == 1:
                    ExcelName = '01_InH_A_LOS_Results'
                    print(ExcelName+'\n')
                    return ExcelName
                elif option == 2:
                    ExcelName = '02_InH_A_NLOS_Results'
                    print(ExcelName+'\n')
                    return ExcelName
                elif option == 3:
                    ExcelName = '03_RMa_A_LOS_Results'
                    print(ExcelName+'\n')
                    return ExcelName
                elif option == 4:
                    ExcelName = '04_RMa_A_NLOS_Results'
                    print(ExcelName+'\n')
                    return ExcelName
                elif option == 5:
                    ExcelName = '05_UMa_A_LOS_Results'
                    print(ExcelName+'\n')
                    return ExcelName
                elif option == 6:
                    ExcelName = '06_UMa_A_NLOS_Results'
                    print(ExcelName+'\n')
                    return ExcelName
                elif option == 7:
                    ExcelName = ' 07_UMi_A_LOS_Results'
                    print(ExcelName+'\n')
                    return ExcelName
                elif option == 8:
                    ExcelName = '08_UMi_A_NLOS_Results'
                    print(ExcelName+'\n')
                    return ExcelName
                else:
                    print('Invalid Option.')
                    print()
                    menu(ExcelName)
                    option = int(input('Select a number for the filename: '))
                    print()
    
def excel():
        data = str(input('Select excel datasets cost or imt2020: '))
        if data == 'cost':
            file_path = r'C:\Users\Mark Macapagal\Desktop\MasterCode2\save\cost\initial_cluster.npy'
        elif data == 'imt2020':
            file_path = r'C:\Users\Mark Macapagal\Desktop\MasterCode2\save\imt2020\initial_cluster.npy'
        if os.path.exists(file_path):
            os.remove(file_path)
            print('Initial Cluster has been Deleted')
        else:
            print('No Initial Cluster Found.')
        print()
        print('1 - Create a New Excel File')
        print('2 - Proceed')
        print()
        
    
def acc(y_true, y_pred):

    """ Calculate clustering accuracy

    Require scikit-learn installed

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    ind = linear_assignment(w.max() - w)
    ind=np .array(list(zip(*ind)))
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def k_means(x_train, n_class, n_init=100):
    """ compute k_means algorithm

    use scikit-learn to compute k-means

    :param x_train: data points
    :param n_class: number of clusters
    :param n_init: The number of different initialization
    :return: k_means model
    """
    k_mean = KMeans(n_clusters=n_class, n_init=n_init)
    km_model = k_mean.fit(x_train)
    return km_model

def relabel(labels):
    keys = list(dict.fromkeys(labels))
    return [keys.index(label) for label in labels]

def cluster_performance(y_pred, y_train, label='kmean'):
    """ calculate performance of clustering


    :param y_pred: Predication vector
    :param y_train: Ground truth vector
    :param label: Method name
    :return: NMI, ACC, ARI
    """    
    k_means_nmi = metrics.normalized_mutual_info_score(y_train, y_pred)
    k_means_ari = metrics.adjusted_rand_score(y_train, y_pred)
    k_means_acc = acc(np.int0(y_train), y_pred)
    k_means_jac = 1 - hamming(np.int0(y_train),relabel(y_pred))
    print('{} NMI is {}'.format(label, k_means_nmi))
    print('{} ARI is {}'.format(label, k_means_ari))
    print('{} Acc is {}'.format(label, k_means_acc))
    print('{} Jac is {}'.format(label, k_means_jac))
    #logging.info("NMI - {:0.2f},ARI - {:0.2f},ACC - {:0.2f},JAC - {:0.2f}".format(k_means_nmi, k_means_ari, k_means_acc,k_means_jac))
    logging.info("NMI - {:0.9f},ARI - {:0.2f},ACC - {:0.2f},JAC - {:0.2f}".format(k_means_nmi, k_means_ari, k_means_acc,k_means_jac))
    logging.info("Numbers in num_list are: {}".format(' '.join(map(str, y_pred))))
    return k_means_nmi, k_means_acc, k_means_ari, k_means_jac
    
def load_data(data_name):
   
    if data_name == 'cost':
        print('Select an Excel File in the File Explorer\n')
        Tk().withdraw()
        filepath = filedialog.askopenfilename(
        initialdir=r'C:\Users\Mark Macapagal\Desktop\MasterCode2\DATASETS',
        title="Open Excel file",
        filetypes=(("Excel files", "*.xlsx"), ("all files", "*.*"))
        )
        if not filepath:
            print("NO FILE SELECTED.")
            return filepath
        elif filepath:
            sheet_Num=str(input('Enter Sheet Number: '))
            print()
            t=pd.read_excel(filepath,sheet_name='Sheet'+sheet_Num)
            print(t)
            data=t.to_numpy()
            x_train=data[:,:-2]
            y_train = relabel(data[:,-1])
            return [x_train, y_train]
    
    if data_name == 'imt2020':
        print('Select an Excel File in the File Explorer\n')
        Tk().withdraw()
        filepath = filedialog.askopenfilename(
        initialdir=r'C:\Users\Mark Macapagal\Desktop\MasterCode2\DATASETS',
        title="Open Excel file",
        filetypes=(("Excel files", "*.xlsx"), ("all files", "*.*"))
        )
        if not filepath:
            print("NO FILE SELECTED.")
            return filepath
        elif filepath:
            sheet_Num=str(input('Enter Sheet Number: '))
            print()
            t=pd.read_excel(filepath,sheet_name='Sheet'+sheet_Num)
            print(t)
            data=t.to_numpy()
            x_train=data[:,:-2]
            y_train = relabel(data[:,-1])
            return [x_train, y_train]
     
            

       
        
            
      
        
    
      
       
       