import numpy as np
import pandas as pd

#加载数据
data_train_org = pd.read_csv("data/round1_iflyad_train.txt",sep='\t')
data_test_org = pd.read_csv("data/round1_iflyad_test_feature.txt",sep='\t')
data_all = pd.concat([data_train_org, data_test_org],ignore_index=True)

def set_error_value(data):
    data['app_cate_id_full'] = data.app_cate_id
    data.loc[data.app_cate_id_full.isnull(), 'app_cate_id_full'] = "NULL"
    data.drop(['app_cate_id'], axis=1, inplace=True)


def attribute_to_number(data):
    columnNames = [
        'advert_industry_inner',  # 广告主行业
        'creative_type',  # 创意类型
        'app_cate_id',  # app分类
        'nnt',  # 网络
        'devtype',  # 设备类型
        'city',  # 城市
    ]

    for name in columnNames:
        data[name + "_factorize"] = pd.factorize(data[name].values, sort=True)[0] + 1
    data.drop(columnNames, axis=1, inplace=True)
    return data


def analysis():
    columnNames = [
        'advert_industry_inner',  # 广告主行业
        'creative_type',  # 创意类型
        'app_cate_id',  # app分类
        'nnt',  # 网络
        'devtype',  # 设备类型
        'city',  # 城市
        'click', #是否点击
    ]
    #filena函数有返回值，所以需要定义新的df对象接收
    #获取指定列的数据，并处理某列的异常值
    used_data = data_all[columnNames].fillna({'app_cate_id': 'NULL'}).copy()
    data_fac = attribute_to_number(used_data)

    data_fac.drop(['click'], axis=1, inplace=True)
    data_train = data_fac[0:data_train_org.shape[0]][:].copy()
    data_test = data_fac[data_train_org.shape[0]:][:].copy()
    y = data_train_org.click
    predictors = data_train.columns

    from sklearn import model_selection
    from sklearn.ensemble import RandomForestClassifier

    alg = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=1000, min_samples_leaf=50,
                                 n_jobs=-1)
    kf = model_selection.KFold(n_splits=10, shuffle=False, random_state=1)

    scores = model_selection.cross_val_score(alg, data_train[predictors], y, cv=kf)

    print(scores)
    print(scores.mean())

if __name__ == '__main__':
    analysis()



