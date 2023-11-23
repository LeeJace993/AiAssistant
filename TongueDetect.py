from collections import Counter
import pandas as pd


def detection(df, clf):
    # 复制测试数据
    dfTest = df

    # 从DataFrame中提取RGB信息，即红、绿、蓝三个通道的值
    X_test = dfTest.iloc[:, 2:5]  # pandas DataFrame

    # 从测试数据中去除重复的行
    X_test_short = X_test.drop_duplicates(subset=['red', 'green', 'blue'])

    # 使用分类器预测去除重复行后的数据的颜色
    prediction_test = clf.predict(X_test_short)

    # 将预测结果合并到DataFrame中
    X_test_short['preds'] = prediction_test

    # 将预测结果与原始数据合并
    df_out = pd.merge(X_test, X_test_short, how='left', on=['red', 'green', 'blue'])

    # 定义舌苔和舌体的颜色类别
    tonguecoat_name = ["danbai", "danhuang", "huang", "jiaohuang", "huihei", "jiaohei", "bobai"]
    tonguebody_name = ["danhong", "hong", "jiang", "qingzi", "bai"]

    # 将预测结果转换为列表
    predictionList = df_out['preds'].tolist()

    # 使用列表解析将颜色分为舌苔和舌体
    coatRows = [k for k in predictionList if k in tonguecoat_name]
    bodyRows = [k for k in predictionList if k in tonguebody_name]

    # 计算舌苔和舌体中最常见的颜色
    a = Counter(coatRows)
    b = Counter(bodyRows)
    # 返回舌苔和舌体中最常见的5种颜色
    color_mapping={
        "danbai":"淡白",
        "danhuang":"淡黄",
        "huang":"黄",
        "jiaohuang":"焦黄",
        "jiaohei":"焦黑",
        "bobai":"薄白",
        "danhong":"淡红",
        "hong":"红",
        "jiang":"姜",
        "qingzi":"青紫",
        "bai":"白",
        "huihei":"灰黑"
    }

    a_m=a.most_common(1)[0][0]
    b_m=b.most_common(1)[0][0]
    a_cn=color_mapping[a_m]
    b_cn=color_mapping[b_m]

    return a_cn, b_cn