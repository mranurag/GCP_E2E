#!/usr/bin/env python3

import pandas as pd
import argparse
from scipy.stats import wasserstein_distance
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,confusion_matrix

parser = argparse.ArgumentParser()

parser.add_argument(
      '--project',
      help='project_id where service is deployed',
      required=True
  )


args = parser.parse_args()
PROJECT = args.project

def getAccuracyTableData(PROJECT):
    query = """ SELECT * FROM modelvalidation.executionsummary1  """
    df = pd.read_gbq(query,
                         project_id=PROJECT,
                         dialect='standard')
    validDict ={}
    tsList =list(df["TimeStamp"].unique())
    mdList = list(df["Model"].unique())* len(tsList)
    thList = [.80]* len(tsList)
    validDict["Accuracy"] =[];validDict["TruePositive"]  =[];validDict["FalseNegative"]  =[];validDict["FalsePositive"]  =[];validDict["TrueNegative"]  =[]
    validDict["TimeStamp"] =tsList
    validDict["Model"] =mdList
    validDict["Threshhold"] =thList
    group_df = df.groupby("TimeStamp")
    for ts in tsList:
        subDf = group_df.get_group(ts)
        ac =accuracy_score(subDf["Target"],subDf["Actual"])

        cm =confusion_matrix(subDf["Target"],subDf["Actual"])

        validDict["Accuracy"].append(ac)
        if len(cm)==1:
            validDict["TruePositive"].append(cm[0,0]);validDict["FalseNegative"].append(0)
            validDict["FalsePositive"].append(0);validDict["TrueNegative"].append(0)
        else:
            validDict["TruePositive"].append(cm[0,0]);validDict["FalseNegative"].append(cm[0,1])
            validDict["FalsePositive"].append(cm[1,0]);validDict["TrueNegative"].append(cm[1,1])


    metricsDF = pd.DataFrame(validDict)
    return metricsDF


def getDriftTableData(PROJECT):
    query = """ SELECT * FROM modelvalidation.executionsummary1  """
    data = pd.read_gbq(query,project_id=PROJECT,dialect='standard')
    tsList =list(data["TimeStamp"].unique())
    group_df = data.groupby("TimeStamp")
    driftDict ={}
    for ts in tsList:
        driftDict[ts]=[]
        

    for i in range(len(tsList)):
        scaler = preprocessing.StandardScaler() 
        subDf1 = group_df.get_group(tsList[0]).select_dtypes(include='float')
        subDf2 = group_df.get_group(tsList[i]).select_dtypes(include='float')
        if i==0:
            colList =list(subDf1.columns)
        subDf1 = scaler.fit_transform(subDf1)
        subDf2 = scaler.fit_transform(subDf2)
        
        subDf1 = pd.DataFrame(subDf1, columns=colList)
        subDf2 = pd.DataFrame(subDf2, columns=colList)
        
       
           
        for col in colList:
            dist = wasserstein_distance(subDf1[col],subDf2[col])
            driftDict[tsList[i]].append(dist)
    distDF =pd.DataFrame(driftDict).transpose().reset_index()
    distDF.columns = ["TimeStamp"] + colList 
    distDF["aveage_drift"] =list((distDF.mean(axis=1)).values)
    return distDF
    
       

    
resDf =   getAccuracyTableData(PROJECT)
driftData = getDriftTableData(PROJECT)


resDf.to_gbq(destination_table="modelvalidation.EfficiencyTablel1",project_id=PROJECT,if_exists='replace')     
driftData.to_gbq(destination_table="modelvalidation.DriftDetection1",project_id=PROJECT,if_exists='replace')     
    
    
    
    
    
