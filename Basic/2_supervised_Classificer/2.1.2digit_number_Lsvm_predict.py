#-*- coding:utf-8 -*-
#!/usr/bin/python
'''
手写字体预测  SVM 支持向量机分类器 
'''
#######导入数据集######
# Scikit-learn内部集成了手写字体数字图片数据集
# 从sklearn.datasets里导入手写体数字加载器
from sklearn.datasets import load_digits
# 从通过数据加载器获得手写体数字的数码图像数据并储存在digits变量中
digits = load_digits()
# 检视数据规模和特征维度
print digits.data.shape

#####划分数据集#####
# 从sklearn.cross_validation中导入train_test_split用于数据分割
from sklearn.cross_validation import train_test_split #在新版本被替换为 model_selection
# 随机选取75%的数据作为训练样本；其余25%的数据作为测试样本
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)

#检视 训练数据和测试数据规模
print y_train.shape
print y_test.shape

#从sklearn.preprocessing里导入数据标准化模块
from sklearn.preprocessing import StandardScaler
#从sklearn.svm里导入基于线性假设的支持向量机分类器LinearSVC
from sklearn.svm import LinearSVC

# 从仍然需要对训练和测试的特征数据进行标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 初始化线性假设的支持向量机分类器LinearSVC
lsvc = LinearSVC()
#进行模型训练
lsvc.fit(X_train, y_train)
# 利用训练好的模型对测试样本的数字类别进行预测，预测结果储存在变量y_predict中
y_predict = lsvc.predict(X_test)

# 使用模型自带的评估函数进行准确性测评
print '线性支持向量机模型准确度： ', lsvc.score(X_test, y_test)

# 使用sklearn.metrics里面的classification_report模块对预测结果做更加详细的分析
from sklearn.metrics import classification_report
print classification_report(y_test, y_predict, target_names=digits.target_names.astype(str))









































