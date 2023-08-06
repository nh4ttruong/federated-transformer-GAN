from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve
from header import *
import header
import importlib
import csv
importlib.reload(header)  # For reloading after making changes

from transformer import *

DEBUG = 1

# ==================================================================

def get_data_batch(train, batch_size, seed):
    # # random sampling - some samples will have excessively low or high sampling, but easy to implement
    # np.random.seed(seed)return return_matrix
    # x = train.loc[ np.random.choice(train.index, batch_size) ].values
    # print("seed is ======>>>> " + str(seed))
    # iterate through shuffled indices, so every sample gets covered evenly
    start_i = (batch_size * seed) % len(train)

    stop_i = start_i + batch_size

    shuffle_seed = (batch_size * seed) // len(train)
    # print("shuffle_seed is ======>>>> " + str(shuffle_seed))
    np.random.seed(shuffle_seed)
    # wasteful to shuffle every time
    train_ix = np.random.choice(
        list(train.index), replace=False, size=len(train))
    # duplicate to cover ranges past the end of the set
    train_ix = list(train_ix) + list(train_ix)
    x = train.loc[train_ix[start_i: stop_i]].values

    x = pd.DataFrame(x)

    x.columns = train.columns
    
    return_matrix = np.reshape(x, (batch_size, -1))
    return return_matrix

# ==================================================================

def classifier_params(prediction, y_test):
    accuracy = accuracy_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    precision = precision_score(y_test, prediction, zero_division=1.0)
    f1 = f1_score(y_test, prediction)

    cm = confusion_matrix(y_test, prediction)
    
    TP = cm[1, 1]
    FN = cm[1, 0]
    FP = cm[0, 1]
    TN = cm[0, 0]
    
    FNR = FN / (FN + TP)
    FPR = FP / (FP + TN)
    DR = TP / (TP + FN)

    results = [recall, precision, FPR, FNR, DR]
    
    print("accuracy:", accuracy)
    print("recall:", recall)
    print("precision:", precision)
    print("f1-score:", f1)
    print("F_Positive_Rate:", FPR)
    print("F_Negative_Rate:", FNR)
    print("Detection_Rate:", DR)
    print()
    
    return results

def Evaluate_Parameter(x, g_z, data_cols, label_cols=[], seed=0, with_class=False, data_dim=2, classifier='XGB', EVALUATION_PARAMETER=''):

    rcl = 0
    acc = 0
    pre = 0
    f1  = 0

    g_z = pd.DataFrame(g_z)

    g_z.columns = x.columns

    REAL_CONCAT_GEN_SET = np.vstack([x, g_z])
    REAL_CONCAT_GEN_SET = pd.DataFrame(REAL_CONCAT_GEN_SET)
    REAL_CONCAT_GEN_SET.columns = x.columns

    REAL_CONCAT_GEN_SET_LABELS = np.hstack(
        [np.zeros(int(len(x))), np.ones(int(len(g_z)))])

    REAL_CONCAT_GEN_SET['Label'] = REAL_CONCAT_GEN_SET_LABELS

    # REAL_CONCAT_GEN_SET = REAL_CONCAT_GEN_SET.sample(frac=1).reset_index(drop=True)

    REAL_CONCAT_GEN_SET_LABELS = REAL_CONCAT_GEN_SET['Label'].values

    # print(pd.DataFrame(REAL_CONCAT_GEN_SET_LABELS))

    # REAL_CONCAT_GEN_SET.to_csv(str(DATA_SET_PATH) + 'GAN' + 'GAN_REAL_CONCAT.csv')
    # print('File: ' + 'GAN' + '_AUG_DATA_SET.csv saved to directory')
# =====================================================================

    if (classifier == 'XGB'):
        # clf = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
        clf = XGBClassifier(eval_metric='logloss')
    elif (classifier == 'DT'):
        clf = DecisionTreeClassifier()
    elif (classifier == 'NB'):
        clf = GaussianNB()
    elif (classifier == 'LGBM'):
        clf = LGBMClassifier()
    elif (classifier == 'RF'):
        # clf = RandomForestClassifier(n_estimators=100)
        clf = RandomForestClassifier()
    elif (classifier == 'LR'):
        # clf = LogisticRegression(max_iter=10000)
        clf = LogisticRegression(max_iter=1000)
    elif (classifier == 'KNN'):
        # clf = KNeighborsClassifier(n_neighbors=5)
        clf = KNeighborsClassifier()
        
    for i in range(10):  # 10-folds
        X_train, X_test, y_train, y_test = train_test_split(
            REAL_CONCAT_GEN_SET, REAL_CONCAT_GEN_SET_LABELS, test_size=0.3, random_state=i)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc_ = accuracy_score(y_test, y_pred)
        rcl_ = recall_score(y_test, y_pred)
        pre_ = precision_score(y_test, y_pred)
        f1_ = f1_score(y_test, y_pred)
        
        acc_ = round(acc_ * 100, 2)
        rcl_ = round(rcl_ * 100, 2)
        pre_ = round(pre_ * 100, 2)
        f1_ = round(f1_ * 100, 2)
        
        #print(i+1, '[Acc:', acc_,',Rcl:' ,rcl_, ']')

        acc += acc_
        rcl += rcl_
        pre += pre_
        f1 += f1_

    acc = round(acc/(i+1), 4)
    rcl = round(rcl/(i+1), 4)
    pre = round(pre/(i+1), 4)
    f1 = round(f1/(i+1), 4)

    # print(classifier, ': [Acc:', acc, ', Rcl:', rcl, ', Pre:', pre, ', F1:', f1, ']')

    return [acc, rcl, pre, f1]
# ==================================================================

def clsfr_train_test(X_train, y_train, X_test, y_test, accu_list=[], rcl_list=[], prec_list=[], f1_list=[], clf=0):
    clf.fit(X_train, y_train)
    y_pred = np.round(clf.predict(X_test))

    accu = accuracy_score(y_test, y_pred)
    rcl = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # TP, TN, FP, FN = perf_measure(y_pred, y_test)

    # accu = ( TP + TN ) / ( TP + TN + FP + FN )
    # rcl  = TP  / ( TP + FN )
    # print(TP, TP + FP)
    # prec = TP / ( TP + FP )
    # f1 = 2 * ( prec * rcl) / ( prec + rcl)

    accu = round(accu * 100, 2)
    rcl = round(rcl * 100, 2)
    prec = round(prec * 100, 2)
    f1 = round(f1 * 100, 2)

    accu_list.append(accu)
    rcl_list.append(rcl)
    prec_list.append(prec)
    f1_list.append(f1)

    ConfusionMatrix(y_pred, y_test)

    print('Accuracy: ' + str(accu_list) + str('%'))
    print('Recall: ' + str(rcl_list) + str('%'))
    print('Precision: ' + str(prec_list) + str('%'))
    print('F1: ' + str(f1_list) + str('%') + '\n\n')
    
    return accu_list, rcl_list, prec_list, f1_list

# ===============================================================================================================================
# ===============================================================================================================================
# ===============================================================================================================================

def LIME_explainer(explainer, model, data, actual_label, instance_idx, num_features, save_path, data_name):
    model_name = model.__class__.__name__
    
    print(f">>>>> LIME explainer - {model_name} at {instance_idx} <<<<<")
    exp = explainer.explain_instance(data.iloc[instance_idx], model.predict_proba, num_features=num_features)
    
    # exp.save_to_file(save_path + f"{model_name}_LIME_instance_{instance_idx}.html")
    
    exp.show_in_notebook(show_table=True)
    
    prediction_proba = model.predict_proba(data.iloc[instance_idx].values.reshape(1, -1))
    predict_label = 1 if float(prediction_proba[:, 1][0]) >= 0.5 else 0
    
    botnet_proba = "%.2f" % (float(prediction_proba[:, 1][0]) * 100)
    benign_proba = "%.2f" % (float(prediction_proba[:, 0][0]) * 100)
    
    explanation = {
        'benign_proba': benign_proba,
        'botnet_proba': botnet_proba,
        'predict_label': predict_label,
        'actual_label': actual_label[instance_idx],
        'exp_html': exp,
        'exp_list': exp.as_list(),
    }

    print(f"[+] Explaination: {model_name} at {instance_idx}", explanation)

    return explanation

def LIME_explanation(explainer, classifiers, data_list, actual_label_list, instance_idxs_list, num_features, save_path, data_names):
    output_tables = {}
    
    for i, classifier in enumerate(classifiers):
        model_name = classifier.__class__.__name__.replace("Classifier", "").replace("LogisticRegression", "LR")
        
        for j, data in enumerate(data_list):
            actual_label = actual_label_list[j]
            instance_idxs = instance_idxs_list[j]
            data_name = data_names[j]
            
            model_data_name = f"{model_name}_{data_name}"
            
            if model_data_name not in output_tables:
                output_tables[model_data_name] = pd.DataFrame(columns=['instance_idx', 'benign_proba', 'botnet_proba', 'predict_label', 'actual_label'])
            
            for instance_idx in instance_idxs:
                explain_result = LIME_explainer(explainer, classifier, data, actual_label, instance_idx, num_features, save_path, model_data_name)
                
                output_tables[model_data_name] = output_tables[model_data_name].append({
                    'instance_idx': instance_idx,
                    'benign_proba': explain_result['benign_proba'],
                    'botnet_proba': explain_result['botnet_proba'],
                    'predict_label': explain_result['predict_label'],
                    'actual_label': explain_result['actual_label'],
                }, ignore_index=True)
                
                explain_result['exp_html'].save_to_file(f"{save_path}/{model_data_name}_{instance_idx}_LIME_explanation_image.html")
                exp_save_path = f"{save_path}/{model_data_name}_{instance_idx}_LIME_explanation_list.csv"
                
                with open(exp_save_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Feature', 'Weight'])
                    
                    for feature, weight in explain_result['exp_list']:
                        writer.writerow([feature, weight])
            
            # Save to CSV
            output_tables[model_data_name].to_csv(f"{save_path}/{model_data_name}_LIME_explanation_table_results.csv", index=False)
            
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.axis('off')
            table = ax.table(cellText=output_tables[model_data_name].values,
                             colLabels=output_tables[model_data_name].columns,
                             cellLoc='center',
                             loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(14)
            table.scale(1.2, 1.2)
            plt.savefig(f"{save_path}/{model_data_name}_LIME_explanation_table.png", bbox_inches='tight')
            plt.title(f"{model_data_name} LIME explanation")
            plt.show()
            plt.close()
    
    # return output_tables

def SHAP_explanation(classifiers, X_train, data_list, data_names, save_path):
    output_tables = {}
    
    for classifier in classifiers:
        model_name = classifier.__class__.__name__.replace("Classifier", "").replace("LogisticRegression", "LR")

        for data, data_name in zip(data_list, data_names):
            model_data_name = f"{model_name}_{data_name}"
            print(f">>>>> Running SHAP explanation - {model_data_name} on {data_name} <<<<<")
            
            # because of LR is need to wrapper :((
            if model_name == "LR":
                explainer = shap.LinearExplainer(classifier, X_train, feature_dependence="independent")
            else:
                explainer = shap.Explainer(classifier)
            
            shap_values = explainer.shap_values(data)
            shap.summary_plot(shap_values, data, show=False)
            plt.title(f"SHAP on {model_name} - {data_name}")
            plt.savefig(f"{save_path}/{model_data_name}_SHAP_summary_plot.png")
            plt.show()
            plt.close()


def generate_gan_data(x, labels=[], weight_or_epoch_number=0, data_dim=0, FULL_CACHE_PATH='',  GAN_type='', TODAY='', DATA_SIZE=0):

    with_class = False
    NOISE_SIZE = 100

    print(GAN_type)

    if GAN_type == 'GAN':
        base_n_count = 256
        gen_model, disc_model, comb_model = define_models_GAN(
            100, data_dim, base_n_count)

    elif GAN_type == 'keras_GAN':
        gen_model = GAN(IMG_SHAPE=data_dim).generator

    elif GAN_type == 'CGAN':
        with_class = True
        base_n_count = 64
        gen_model, disc_model, comb_model = define_models_CGAN(
            100, data_dim, 1, base_n_count)

    elif GAN_type == 'WGAN':

        gen_model = WGAN(IMG_SHAPE=data_dim).generator

        # base_n_count = 128
        # gen_model, disc_model, comb_model = define_models_WGAN(100, data_dim, base_n_count)

    elif GAN_type == 'WCGAN':
        with_class = True
        base_n_count = 128

        gen_model, disc_model, comb_model = define_models_CGAN(
            NOISE_SIZE, data_dim, 1, base_n_count, type='Wasserstein')

    print('Generating ' + GAN_type + '-bots')

    gen_model.load_weights(FULL_CACHE_PATH + TODAY + '/' + GAN_type +
                           '_generator_model_weights_step_' + str(weight_or_epoch_number)+'.h5')

    np.random.seed(20)

    z = np.random.normal(size=(DATA_SIZE, 100))

    if USE_UNIFORM_NOISE:

        z = np.random.uniform(size=(DATA_SIZE, NOISE_SIZE))

    if with_class:

        g_z = gen_model.predict([z, labels])
    else:
        g_z = gen_model.predict(z)

    # g_z -= g_z.min()
    # g_z /= g_z.max()

    df = pd.DataFrame(g_z).copy()

    if GAN_type == 'GAN' or GAN_type == 'WGAN':

        df.columns = x.columns[:-1]

    elif GAN_type == 'CGAN' or GAN_type == 'WCGAN':

        df.columns = x.columns

    df['Label'] = 1  # Label = 1 (For Black box Attack)

    return df
# ===============================================================================================================================
# ===============================================================================================================================
# ===============================================================================================================================


def augment_bots(X_train, y_train, bots, cols, GAN_type='', DATA_SET_PATH='', classifier=''):
    df = pd.DataFrame(X_train)
    df.columns = cols[:-1]

    df['Label'] = y_train
    if DEBUG:

        BOT_COUNTS = df['Label'].value_counts()[1]
        BENIGN_COUNTS = df['Label'].value_counts()[0]

        print('Bots in dataset:')
        print(BOT_COUNTS)

        print('Normal in dataset:')
        print(BENIGN_COUNTS)

        print('Dataset before aug:')
        print(df.shape)

    # for i in range(10):

    df = pd.concat([df, bots]).reset_index(
        drop=True)  # Augmenting with real botnets

    # df.loc[df[df.columns] >0.5 ] = 1  # For Husnain Data

    gen_data_set = df
# ===============================================================================================================================
    gen_data_set.to_csv(str(DATA_SET_PATH) + classifier +
                        '_' + GAN_type + '_AUG_DATA_SET.csv')
    print('File: ' + GAN_type + '_AUG_DATA_SET.csv saved to directory')
# ===============================================================================================================================

    X_train = gen_data_set[cols[:-1]].values
    y_train = gen_data_set['Label'].values

    if DEBUG:

        BOT_COUNTS = gen_data_set['Label'].value_counts()[1]
        BENIGN_COUNTS = gen_data_set['Label'].value_counts()[0]

        print('Bots in dataset:')
        print(BOT_COUNTS)

        print('Normal in dataset:')
        print(BENIGN_COUNTS)

        print('Dataset after aug:')
        print(gen_data_set.shape)

    return X_train, y_train
# ===============================================================================================================================
# ===============================================================================================================================
# ===============================================================================================================================


def augment_bots_in_test_set(X_test, y_test, bots, cols):
    df = pd.DataFrame(bots)
    # df.columns = cols[:-1]

    # df['Label'] = y_test
    # if DEBUG:

    #     BOT_COUNTS = df['Label'].value_counts()[1]

    #     print('Bots in dataset:')
    #     print(BOT_COUNTS)

    #     print('Dataset before aug:')
    #     print(df.shape)

    # for i in range(10):

    # df = pd.concat([df, bots]).reset_index(drop=True) #Augmenting with real botnets

    gen_data_set = df

    X_test = gen_data_set[cols[:-1]].values
    y_test = gen_data_set['Label'].values

    # if DEBUG:

    #     BOT_COUNTS = gen_data_set['Label'].value_counts()[1]

    #     print('Bots in dataset:')
    #     print(BOT_COUNTS)

    #     print('Dataset after aug:')
    #     print(gen_data_set.shape)

    return X_test, y_test
# ===============================================================================================================================
# ===============================================================================================================================
# ===============================================================================================================================
# def collect_evasions():

#     SimpleMetrics(y_pred, y_test)
#     evasions = np.where((y_pred == 0) & (y_test == 1), 1, 0)
#     # print([i for i, x in enumerate(evasions) if x])

#     ev_list = [i for i, x in enumerate(evasions) if x]

#     evasions_list.extend(ev_list)
#     # evasions_list = list(dict.fromkeys(evasions_list))

#     print('Indices of Elements to be added: ' + str(ev_list))

#     # print('evasion_list--> unrepeated: ' + str(evasions_list))

#     print('evasion_list size --> : ' + str(len(evasions_list)) + '\n')

#     df = dfEvasions

#     for i in evasions_list:

#         # print('\n' + str(test_set[i]) + '\n')
#         # print('Length of this sample is: ' + str(len(test_set[i]))+ '\n')

#         df = df.append(dict(zip(df.columns, test_set[i])), ignore_index=True)
#         # dfEvasions = dfEvasions.append(dict(zip(dfEvasions.columns, test_set[i])), ignore_index=True)

#     # print('Df: \n' + str(df) + '\n\n')
#     # print('Evasions df: \n' + str(dfEvasions) + '\n\n')

#     dfEvasions = pd.concat([dfEvasions, df])

#     # dfEvasions = inverse_transform(dfEvasions)

#     # print('Evasions df After Concat: \n' + str(dfEvasions) + '\n\n')


#     # print(dfEvasions.describe(include = 'all'))
#     print('=======================================>>>>>>>>>>>>>>>>>>>>>>>>')

#         dfEvasions.to_csv(DATA_SET_PATH + str(classifier) +'_evasions.csv')


def predict_clf(G_Z, test_Normal, test_Bots, clf, ONLY_GZ=False):

    pred_G_Z_clf = clf.predict(G_Z)
    Ev_GZ_Bot_clf = round(
        sum(pred_G_Z_clf) / G_Z.shape[0], 4
    )

    if ONLY_GZ == False:
        pred_Normal_clf = clf.predict(test_Normal)
        pred_Bots_clf = clf.predict(test_Bots)

        N_acc_clf = round(
            sum(pred_Normal_clf) / test_Normal.shape[0], 4
        )

        Ev_Real_Bot_clf = round(
            sum(pred_Bots_clf) / test_Bots.shape[0], 4
        )

    if ONLY_GZ == True:
        return [Ev_GZ_Bot_clf]

    return [N_acc_clf, Ev_Real_Bot_clf]
