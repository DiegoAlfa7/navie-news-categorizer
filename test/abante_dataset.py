import pymongo
import json


feeds = myclient = pymongo.MongoClient("mongodb://db:27017/")
db = myclient["db"]

articles = list(db.article.find(filter={"category": 'google' }, projection={"_id": False}))

with open(str('dataset_abante.json'), mode='w', encoding='UTF-8') as file_:
    file_.write(json.dumps(articles))
    file_.close()

