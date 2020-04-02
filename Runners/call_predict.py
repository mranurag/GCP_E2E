#!/usr/bin/env python3

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
from oauth2client.service_account import ServiceAccountCredentials
import argparse
import json
import pandas as pd

credFile = "customAuth.json"

scopes = ['https://www.googleapis.com/auth/cloud-platform']

credentials = ServiceAccountCredentials.from_json_keyfile_name(credFile, scopes=scopes)

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--project", required=True,
                    help="Project that heart service is deployed in")
                    
parser.add_argument("-md", "--model", required=True,
                    help="model to use")

parser.add_argument("-vs", "--version", required=True,
                    help="version")
args = parser.parse_args()
#credentials = GoogleCredentials.get_application_default()

api = discovery.build('ml', 'v1', credentials=credentials,
            discoveryServiceUrl='https://storage.googleapis.com/cloud-ml/discovery/ml_v1_discovery.json')

request_data = {'instances':
  [
      {
      "trestbps": .000001, 
      "chol": 0, 
      "thalach": 0
      }
      ,
      {
      "trestbps": 0, 
      "chol": .000001, 
      "thalach": .0000003
      }
      ,
      {
      "trestbps": .00000001, 
      "chol": 0, 
      "thalach": .03411111
      }
  ]
}

PROJECT = args.project
MODEL = args.model
VERSION = args.version
parent = 'projects/%s/models/%s/versions/%s' % (PROJECT, MODEL, VERSION)
response = api.projects().predict(body=request_data, name=parent).execute()
print("response={0}".format(response))

def getGBQSet(request_data,response):
    import datetime
    today = datetime.date.today()
   
    request_data =request_data["instances"]
    cols = list(request_data[0].keys())
    resDict ={}
    for col in cols:
        resDict[col]=[]
    for item in request_data:
        for key in item.keys():
            val = item[key]
            resDict[key].append(val)
    resDict["Target"]=[]
    resDict["TimeStamp"]=[]
    response =response["predictions"]
    for res in response:
        val = res["output_1"][0]
        #resDict["TimeStamp"].append("{:%b_%y}".format(today))
        resDict["TimeStamp"].append("Jan_20")
        
        if val>.40:
            resDict["Target"].append("yes")
        else:
            resDict["Target"].append("No")
    resDF = pd.DataFrame(resDict)
    resDF["Actual"] ="No"
    resDF[cols] =  resDF[cols].astype(float)
    return resDF
    
resDf =   getGBQSet(request_data,response)
resDf["Model"] =  MODEL 


resDf.to_gbq(destination_table="modelvalidation.executionsummary1",project_id="symmetric-flag-268717",if_exists='append')        
     
    
    
    
    
    
