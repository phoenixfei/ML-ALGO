import numpy as np
import pandas as pd

#简单地了解数据情况
data_train_org = pd.read_csv('kedaxunfei/data/round1_iflyad_train.txt', sep='\t')
data_test_org = pd.read_csv('kedaxunfei/data/round1_iflyad_test_feature.txt', sep='\t')
data_all = pd.concat([data_train_org, data_test_org], ignore_index=True)
#print(data_all.info())#获取数据的简略信息

#查看某一列数据的具体组成，值+个数
#print(data_all['creative_is_js'].value_counts())

#查看多列数据的对应情况
#data_train_org.groupby(by=['province', 'click']).size()/.count()

columnNames = [
    'advert_industry_inner',  # 广告主行业
    'creative_type',  # 创意类型
    'app_cate_id',  # app分类
    'nnt',  # 网络
    'devtype',  # 设备类型
    'city',  # 城市
    'click',  # 是否点击
]

used_data = data_all[columnNames].fillna({'app_cate_id': 'NULL'}).copy()
