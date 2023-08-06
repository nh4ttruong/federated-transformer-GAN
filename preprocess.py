
import importlib
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from xgboost import XGBClassifier
# Feature Importance
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import shuffle
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import re

USE_THREE_BOTS = 0

DEBUG = 1

network_traffic = 1

CAT_FEATURES = 0

CTU_NERIS = 0
USE_ONE_BOT = 1
USE_ALL_BOTS = 0
USE_FOUR_BOTS = 0

REMOVE_SKEW = 0

USE_FEATURE_REDUCTION = 1


def top_features(data, classifier=""):

    X = data.drop(["Label"], axis=1).values
    y = data["Label"].values

    clf = XGBClassifier()

    # if(classifier=='XGB'):
    print("Evaluation ---->> XBG")
    clf = XGBClassifier()

    # if ALL_CLASSIFIERS:
    #     if (classifier=='DT'):
    #         print('Evaluation ---->> DT')
    #         clf = DecisionTreeClassifier()
    #     elif (classifier=='RF'):
    #         print('Evaluation ---->> RF')
    #         clf=RandomForestClassifier(n_estimators=100)

    clf.fit(X, y)

    importance = clf.feature_importances_

    importance = [importance.tolist()]
    print(importance)

    important_features_dict = {}

    # important_list = list(map(lambda el:[el], importance))

    dictlist = []

    print(importance)

    importance_df = pd.DataFrame(
        importance, columns=data.drop(["Label"], axis=1).columns
    )

    print(importance_df)

    for x, i in enumerate(importance_df.columns):
        important_features_dict[x] = i

    important_features_list = sorted(
        important_features_dict, key=important_features_dict.get, reverse=True
    )

    print("Most important features: %s" % important_features_list)

    importance_df = importance_df.loc[:, (importance_df != 0).any(axis=0)]

    print(len(importance_df.columns))

    return importance_df.columns

    # # # summarize feature importance
    # # for i,v in enumerate(importance):
    # #     print('Feature: %0d, Score: %.5f' % (i,v))
    # # # plot feature importance
    # pyplot.bar([x for x in range(len(importance))], importance)
    # pyplot.show()

def data_analysis(dataset):
  sns.set(rc={'figure.figsize':(6, 4)})
  sns.set_theme()
  
  dataset = dataset['Label'].replace({0: 'Benign', 1: 'Botnet'})
  ax = sns.countplot(x='Label', data=dataset)
  ax.set(xlabel='Label', ylabel='Count')
  plt.show()

def split_dataset(dataname, file_path, amount_part=4):
    
    save_path = os.path.dirname(file_path)
    print(save_path)
    clients_path = os.path.join(save_path, 'clients')
    
    try:
        os.makedirs(clients_path)
    except FileExistsError:
        pass
    
    dataset = pd.read_csv(file_path,low_memory=False)
    
    train_set, test_set = np.split(dataset.sample(frac=1, random_state=42), [int(.7*len(dataset))])
    
    train_set_label_benign = train_set[train_set['Label'] == 0]
    train_set_label_bot = train_set[train_set['Label'] == 1]
    
    parts_label_benign = np.array_split(train_set_label_benign, amount_part)
    parts_label_bot = np.array_split(train_set_label_bot, amount_part)
    
    parts = []
    for i in range(amount_part):
        part = pd.concat([parts_label_benign[i], parts_label_bot[i]])
        parts.append(part.sample(frac=1, random_state=42))
    
    # Save each part to file
    for i, part in enumerate(parts):
        print(f'# Part {i}:', part['Label'].value_counts())
        part.to_csv(os.path.join(clients_path, f'{dataname}_train_part_{i}.csv'), index=False)
    
    # Save the train and test sets to file
    train_set.to_csv(os.path.join(save_path, f'{dataname}_train_set.csv'), index=False)
    test_set.to_csv(os.path.join(save_path, f'{dataname}_test_set.csv'), index=False)
    
    print(f'# Total train_set:', train_set['Label'].value_counts())
    print(f'# Total test_set:', test_set['Label'].value_counts())


def prepare_cic_2018(PATH="", INPUT_FILE_NAME="", CSV_ONE_BOT=0):

    # =============================================================================================
    # Read Data
    # =============================================================================================

    data = pd.read_csv(os.path.join(PATH, INPUT_FILE_NAME), low_memory=False)

    print(
        "Processing File: " + str(INPUT_FILE_NAME) + " DATA shape: " + str(data.shape)
    )

    print("The shape of dataset is: ", data.shape)

    data["Label"] = data["Label"].replace(["Benign", "Bot"], [0, 1])

    normal = data.loc[data["Label"] == 0].copy()
    bots = data.loc[data["Label"] == 1].copy()

    print("Before Preprocesing: Total: " + str(data.shape))
    print("Before Preprocesing: Normal: " + str(normal.shape))
    print("Before Preprocesing: Bots: " + str(bots.shape))

    # =============================================================================================
    # Check for any NULL values in the data & remove if any
    # =============================================================================================

    # print(data.describe())

    # replace inf with nan and then drop the rows with nans

    if DEBUG:
        print("Data Shape before droping NULL and INF values: ")

        print(data.shape)

    data = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
        .reset_index(drop=True)
        .copy()
    )

    # for cols in data.columns.tolist()[1:]:

    if DEBUG:

        print("Data Shape after droping NULL and INF values: ")

        print(data.shape)

    # =============================================================================================
    # Drop all categorical features
    # =============================================================================================

    data = data.drop(["Dst Port", "Timestamp", "Protocol"], axis=1).copy()
    if DEBUG:
        print("data_df after removing categorical features")
        print(data.shape)

    # =============================================================================================

    data_df = data.drop(["Label"], axis=1)

    data_cols = data_df.columns

    data_df = data_df[data[data_cols] >= 0].copy()  # Remove -ve values

    # print(data_df.describe())

    if DEBUG:

        print("data_df after removing Label column")
        print(data_df.shape)

    # =============================================================================================
    # Convert all values to float
    # =============================================================================================

    data_df = data_df.astype(float)

    if DEBUG:
        print(" Data Columns after converting to Float")
        print(data_df.describe())

    # =============================================================================================
    # Compute skew greater than 1 and less than -1 for suppressing outliers
    # =============================================================================================
    if REMOVE_SKEW:

        skew = data_df.skew(axis=0, skipna=True)

        print(skew)

        high_skew_list = skew[skew > 20].index.tolist()
        low_skew_list = skew[skew < 20].index.tolist()

        if DEBUG:

            print("low skew list" + str(low_skew_list))
            print("high skew list" + str(high_skew_list))

        data_df[high_skew_list] = np.log(data_df[high_skew_list].values + 1).copy()
        data_df[low_skew_list] = np.log(data_df[low_skew_list].values + 1).copy()

    # =============================================================================================
    # Remove any left over inf values
    # =============================================================================================

    inf_indx = data_df.index[np.isinf(data_df).any(1)]

    if DEBUG:
        print("INF values before removing: " + str(inf_indx))

    data_df = data_df.drop(inf_indx, axis=0).copy()
    data = data.drop(inf_indx, axis=0).copy()

    data_cols = data_df.columns
    data[data_cols] = data_df.copy()

    data = data.reset_index(drop=True).copy()

    if DEBUG:
        print("INF values removed and Data reindexed")

    # =============================================================================================
    # remove any columns with std = Zero
    # =============================================================================================

    if DEBUG:
        print("Data before removing std = 0 columns")
        print(data_df.shape)
        print(data.shape)

    data_df = data_df.loc[:, data_df.var() == 0]

    data = data.drop(data_df.columns, axis=1)

    if DEBUG:

        print("Data after removing std = 0 columns")
        print(data.shape)

    # =============================================================================================
    # Scale data between 0 and 1 for GAN input
    # =============================================================================================

    data_df = data.drop(["Label"], axis=1)

    # print(data_df.describe())

    print(data_df.describe())

    print(
        "HEREEEEEEEEEEEEEEEEEEEEEEEEEEE+++++++++++++++++++++++++++++++>>>>>>>>>>>>>>>>>>>>>>"
    )

    data_df -= data_df.min()
    data_df /= data_df.max()

    print(data_df.describe())

    # =============================================================================================
    # Check if there is any NaN
    # =============================================================================================
    inf_indx = data_df.isnull().sum().sum()

    if DEBUG:

        print("INF values: " + str(inf_indx))

    inf_indx = data.index[np.isinf(data_df).any(1)]

    if DEBUG:
        print("Any Left over INF values: " + str(inf_indx))

    data[data_df.columns] = data_df.copy()

    data_df = data["Label"].copy()

    # =============================================================================================
    # remove any columns with all values = Zero
    # =============================================================================================

    data = data.loc[:, (data != 0).any(axis=0)]

    # =============================================================================================

    if DEBUG:
        print(" Data Columns after removing Flow ID: " + str(data.columns))

    # =============================================================================================
    # Check for any NULL values in the data & remove if any
    # =============================================================================================

    # replace inf with nan and then drop the rows with nans
    if DEBUG:
        print("Data Shape before droping NULL and INF values: ")
        print(data.shape)

    data = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
        .reset_index(drop=True)
        .copy()
    )

    if DEBUG:
        print("Data Shape before droping NULL and INF values: ")
        print(data.shape)

    # selected_features = ['Idle Max', 'Idle Mean', 'Packet Length Min', 'FIN Flag Count', 'FWD Init Win Bytes', 'Bwd Packet Length Min', 'Flow IAT Min', 'Idle Min', 'Subflow Fwd Bytes', 'Fwd IAT Min', 'Fwd Packet Length Min', 'Packet Length Std', 'Fwd Packets/s', 'Bwd Packets/s', 'Label']
    # data = data[selected_features].copy()

    print(data.describe())
    data = round(data, 8)

    if USE_FEATURE_REDUCTION:
        selected_columns = top_features(data, "XGB")
        print(selected_columns)

        data_frame = data.drop(["Label"], axis=1)

        data_frame = data[selected_columns].copy()

        data_frame["Label"] = data["Label"].copy()

        data = data_frame.copy()

        print(data.shape)

    # ========== Extract a Chunk of Bots ============================================================

    normal = data.loc[data["Label"] == 0].copy()
    bots = data.loc[data["Label"] == 1].copy()

    print("After Preprocesing: Bots: " + str(bots.shape))

    bots = bots[0 : 512 * 5]

    # =============================================================================================

    data = pd.concat([bots, normal]).reset_index(drop=True).copy()

    normal = data.loc[data["Label"] == 0].copy()
    bots = data.loc[data["Label"] == 1].copy()

    print("After Preprocesing: Total: " + str(data.shape))
    print("After Preprocesing: Normal: " + str(normal.shape))
    print("After Preprocesing: Bots Chunk: " + str(bots.shape))

    # data= data.drop(['TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofBwdPackets', 'FlowIATMin', 'FwdIATMin', 'BwdIATMin', 'FwdPSHFlags', 'BwdHeaderLength', 'FINFlagCount', 'SYNFlagCount', 'RSTFlagCount', 'ECEFlagCount', 'FwdHeaderLength', 'SubflowFwdPackets', 'SubflowBwdPackets' , 'SubflowBwdBytes', 'act_data_pkt_fwd', 'ActiveMean', 'ActiveStd', 'ActiveMax', 'ActiveMin', 'IdleStd'], axis=1).copy()
    # =============================================================================================
    OUTPUT_FILE_NAME = "preprocessed_" + str(INPUT_FILE_NAME)
    OUTPUT_FILE_PATH = os.path.join(PATH, OUTPUT_FILE_NAME)
    
    print("File: " + str(OUTPUT_FILE_NAME) + " is saving ...")
    data.to_csv(OUTPUT_FILE_PATH, index=False)
    print("File: " + str(OUTPUT_FILE_NAME) + "saved to directory")

    # =============================================================================================

    return data, OUTPUT_FILE_PATH


def prepare_UNSW_IoT(PATH="", INPUT_FILE_NAME=""):

    # =============================================================================================
    # Read Data
    # =============================================================================================
    data = pd.read_csv(os.path.join(PATH, INPUT_FILE_NAME), low_memory=False, sep = ';', index_col=0)

    print(
        "Processing File: " + str(INPUT_FILE_NAME) + " DATA shape: " + str(data.shape)
    )

    # =============================================================================================
    # Check for any NULL values in the data & remove if any
    # =============================================================================================

    # replace inf with nan and then drop the rows with nans

    if DEBUG:
        print("Data Shape before droping NULL and INF values: ")

        print(data.shape)

    data = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
        .reset_index(drop=True)
        .copy()
    )

    # for cols in data.columns.tolist()[1:]:

    if DEBUG:

        print("Data Shape after droping NULL and INF values: ")

        print(data.shape)

    # =============================================================================================
    # Drop all categorical features
    # =============================================================================================
    
    data = data.drop(
        [
            "pkSeqID",
            "saddr",
            "sport",
            "daddr",
            "dport",
            "seq",
            "category",
            "subcategory",
            "proto",
        ],
        axis=1,
    ).copy()
    
    if DEBUG:
        print("data_df after removing categorical features")
        print(data.shape)

    # =============================================================================================

    # data_df = data.drop(["attack"], axis=1)
    data = data.rename(columns={"attack": "Label"})
    
    data.columns = data.columns.str.capitalize()

    if USE_FEATURE_REDUCTION:

        selected_columns = top_features(data, "XGB")
        print(selected_columns)

        data_frame = data.drop(["Label"], axis=1)

        data_frame = data[selected_columns].copy()

        data_frame["Label"] = data["Label"].copy()

        data = data_frame.copy()

        print(data.shape)
        
    data_cols = data.columns
    
    print(data_cols)

    print(data.describe)

    data_df = data[data[data_cols] >= 0].copy()

    if DEBUG:

        print("data_df after removing attack column")
        print(data_df.shape)

    # =============================================================================================
    # Convert all values to float
    # =============================================================================================

    data_df = data_df.astype(float)
    data_df['Label'] = data_df['Label'].astype('int')

    if DEBUG:
        print(" Data Columns after converting to Float: " + str(data_df.columns))

    # =============================================================================================
    # Compute skew greater than 1 and less than -1 for suppressing outliers
    # =============================================================================================
    if REMOVE_SKEW:

        skew = data_df.skew(axis=0, skipna=True)

        print(skew)

        high_skew_list = skew[skew > 10].index.tolist()
        low_skew_list = skew[skew < 10].index.tolist()

        if DEBUG:

            print("low skew list" + str(low_skew_list))
            print("high skew list" + str(high_skew_list))

        data_df[high_skew_list] = np.log(data_df[high_skew_list].values + 1).copy()
        data_df[low_skew_list] = np.log(data_df[low_skew_list].values + 1).copy()

    # =============================================================================================
    # Remove any left over inf values
    # =============================================================================================

    inf_indx = data_df.index[np.isinf(data_df).any(1)]

    if DEBUG:
        print("INF values before removing: " + str(inf_indx))

    data_df = data_df.drop(inf_indx, axis=0).copy()
    data = data.drop(inf_indx, axis=0).copy()

    data_cols = data_df.columns
    data[data_cols] = data_df.copy()

    data = data.reset_index(drop=True).copy()

    if DEBUG:
        print("INF values removed and Data reindexed")

    # =============================================================================================
    # remove any columns with std = Zero
    # =============================================================================================

    if DEBUG:
        print("Data before removing std = 0 columns")
        print(data_df.shape)
        print(data.shape)

    data_df = data_df.loc[:, data_df.var() == 0]

    data = data.drop(data_df.columns, axis=1)

    if DEBUG:

        print("Data after removing std = 0 columns")
        print(data.shape)

    # =============================================================================================
    # Scale data between -1 and 1 for GAN input
    # =============================================================================================
    # data_df = data.drop(["Label"], axis=1)

    data_df -= data_df.min()
    data_df /= data_df.max()

    # =============================================================================================
    # Check if there is any NaN
    # =============================================================================================
    inf_indx = data_df.isnull().sum().sum()

    if DEBUG:

        print("INF values: " + str(inf_indx))

    inf_indx = data.index[np.isinf(data_df).any(1)]

    if DEBUG:
        print("Any Left over INF values: " + str(inf_indx))

    data[data_df.columns] = data_df.copy()
    data = round(data, 8).copy()

    # =============================================================================================
    # Check for any NULL values in the data & remove if any
    # =============================================================================================

    # replace inf with nan and then drop the rows with nans
    if DEBUG:
        print("Data Shape before droping NULL and INF values: ")
        print(data.shape)

    # REMOVE column contain "Unname"
    data = data.drop(data.columns[data.columns.str.contains('unnamed',case = False)],axis = 1)

    data = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
        .reset_index(drop=True)
        .copy()
    )
    
    if DEBUG:
        print("Data Shape before droping NULL and INF values: ")
        print(data.shape)

    print(data.describe())

    # =============================================================================================
    OUTPUT_FILE_NAME = "preprocessed_" + str(INPUT_FILE_NAME)
    OUTPUT_FILE_PATH = os.path.join(PATH, OUTPUT_FILE_NAME)
    
    print("File: " + str(OUTPUT_FILE_NAME) + " is saving ...")
    data.to_csv(OUTPUT_FILE_PATH, index=False)
    print("File: " + str(OUTPUT_FILE_NAME) + "saved to directory")

    # =============================================================================================
    # data = data.drop(["response_body_len", "is_sm_ips_ports"], axis=1).copy()

    return data, OUTPUT_FILE_PATH

def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x] 
    df.drop(name, axis=1, inplace=True)
   
def prepare_EdgeIIoT(PATH="", INPUT_FILE_NAME=""):
    df = pd.read_csv(os.path.join(PATH, INPUT_FILE_NAME), low_memory=False)
    
    minority_classes = ['Port_Scanning', 'XSS', 'Ransomware', 'Fingerprinting', 'MITM']
    
    data = df[['arp.opcode', 'arp.hw.size', 'icmp.checksum', 'icmp.seq_le', 'icmp.unused',
           'http.content_length', 'http.response', 'http.tls_port', 'tcp.ack',
           'tcp.ack_raw', 'tcp.checksum', 'tcp.connection.fin',
           'tcp.connection.rst', 'tcp.connection.syn', 'tcp.connection.synack',
           'tcp.flags', 'tcp.flags.ack', 'tcp.len', 'tcp.seq', 'udp.stream',
           'udp.time_delta', 'dns.qry.name', 'dns.qry.qu',
           'dns.qry.type', 'dns.retransmission', 'dns.retransmit_request',
           'dns.retransmit_request_in', 'mqtt.conflag.cleansess', 'mqtt.conflags', 'mqtt.hdrflags', 'mqtt.len',
           'mqtt.msg_decoded_as', 'mqtt.msgtype', 'mqtt.proto_len',
           'mqtt.topic_len', 'mqtt.ver', 'mbtcp.len', 'mbtcp.trans_id', 'mbtcp.unit_id', 'Attack_label']]
    
    data.info()
    
    data = data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x.replace('.', '_')))
    data.dropna(axis=0, how='any', inplace=True)
    data.drop_duplicates(subset=None, keep="first", inplace=True)
    data = shuffle(data)
    df.isna().sum()
    
    data = data.rename(columns={"Attack_label": "Label"})
    
    data.shape
    
    data.info()

    print(data['Label'].value_counts())
    
    OUTPUT_FILE_NAME = "preprocessed_" + str(INPUT_FILE_NAME)
    OUTPUT_FILE_PATH = os.path.join(PATH, OUTPUT_FILE_NAME)

    print("File: " + str(OUTPUT_FILE_NAME) + " is saving ...")
    data.to_csv(OUTPUT_FILE_PATH, index=False)
    print("File: " + str(OUTPUT_FILE_NAME) + "saved to directory")
    
    return data, OUTPUT_FILE_PATH


def prepare_InSDN(PATH, INPUT_FILE_NAME, DEBUG=True):

    data = pd.read_csv(os.path.join(PATH, INPUT_FILE_NAME), low_memory=False)
    
    print("Before columns:", data.columns)
    drop_columns = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
    data = data.drop(drop_columns, axis=1)
    
    print("After columns:", data.columns)
    
    data["Label"] = np.where(data['Label'] == "Normal", 0, 1)
    data["Label"].value_counts()
    
    data = data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    data.dropna(axis=0, how='any', inplace=True)
    data.drop_duplicates(subset=None, keep="first", inplace=True)
    data = shuffle(data)
    data.isna().sum()
    
    print("USE_FEATURE_REDUCTION == True")
    USE_FEATURE_REDUCTION = True
    
    if USE_FEATURE_REDUCTION:
        selected_columns = top_features(data, "XGB")
        print(selected_columns)

        data_frame = data.drop(["Label"], axis=1)
        data_frame = data[selected_columns].copy()
        data_frame["Label"] = data["Label"].copy()
        data = data_frame.copy()
        print(data.shape)

    # ========== Extract a Chunk of Bots ============================================================

    normal = data.loc[data["Label"] == 0].copy()
    bots = data.loc[data["Label"] == 1].copy()

    print("After Preprocesing: Bots: " + str(bots.shape))
    
    # get 1536 bot
    bots = bots[0 : 512 * 20] 

    data = pd.concat([bots, normal]).reset_index(drop=True).copy()

    normal = data.loc[data["Label"] == 0].copy()
    bots = data.loc[data["Label"] == 1].copy()

    print("After Preprocesing: Total: " + str(data.shape))
    print("After Preprocesing: Normal: " + str(normal.shape))
    print("After Preprocesing: Bots Chunk: " + str(bots.shape))

    OUTPUT_FILE_NAME = "preprocessed_" + str(INPUT_FILE_NAME)
    OUTPUT_FILE_PATH = os.path.join(PATH, OUTPUT_FILE_NAME)
    
    print("File: " + str(OUTPUT_FILE_NAME) + " is savingggg ...")
    data.to_csv(OUTPUT_FILE_PATH, index=False)
    print("File: " + str(OUTPUT_FILE_NAME) + " saved to directory")
    
    return data, OUTPUT_FILE_PATH


