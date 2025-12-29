# 导入所需库
import numpy as np
import pandas as pd
import random
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, precision_score, recall_score, accuracy_score
import tensorflow as tf
from model1 import ourmodel_1mer, ourmodel_4mer, build_combined_model

# 设置随机种子以保证可重现性
my_seed = 42
np.random.seed(my_seed)
random.seed(my_seed)
tf.random.set_seed(my_seed)

# 定义用于计算度量标准的函数
def calculate_metrics(y_true, y_pred):
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_pred_prob = y_pred[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_true_labels, y_pred_labels).ravel()
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    mcc = matthews_corrcoef(y_true_labels, y_pred_labels)
    auc_score = roc_auc_score(y_true_labels, y_pred_prob)
    precision = precision_score(y_true_labels, y_pred_labels)
    recall = recall_score(y_true_labels, y_pred_labels)
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    return accuracy, mcc, auc_score, precision, recall, specificity, sensitivity

# 加载数据
data_4mer = np.load('/mnt/raid5/data2/zywei/wcross-attention/10.AVPs/xin/100_4mer.npz')
x_train_4mer = data_4mer['x_train']
x_test_4mer = data_4mer['x_test']

data_1mer = np.load('/mnt/raid5/data2/zywei/wcross-attention/10.AVPs/xin/100_1mer.npz')
x_train_1mer = data_1mer['x_train']
x_test_1mer = data_1mer['x_test']

# 加载并转换标签为分类格式
y_train = pd.read_csv('/mnt/raid5/data2/zywei/wcross-attention/10.AVPs/xin/train/y_train.csv').to_numpy()
y_train = to_categorical(y_train, dtype='int')

y_test = pd.read_csv('/mnt/raid5/data2/zywei/wcross-attention/10.AVPs/xin/test/y_test.csv').to_numpy()
y_test = to_categorical(y_test, dtype='int')

# 实例化两个单独的模型
model_1mer = ourmodel_1mer(input_shape=(100, 150))
model_4mer = ourmodel_4mer(input_shape=(97, 150))

# 构建并实例化组合模型
combined_model = build_combined_model(model_1mer, model_4mer)

# 编译模型
combined_model.compile(optimizer=Adam(learning_rate=0.0001),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

# 准备模型训练时的回调函数
#callbacks = [
#    EarlyStopping(monitor='val_loss', patience=40, verbose=1),
#    ModelCheckpoint('best_model.h5',monitor='val_loss', verbose=1, save_best_only=True)]

# 训练模型
history = combined_model.fit(
    [x_train_1mer, x_train_4mer], y_train,
    batch_size=32,
    epochs=16,
    verbose=1,
    validation_data=([x_test_1mer, x_test_4mer], y_test),
#    callbacks=callbacks
)

# 在测试集上评估模型
scores = combined_model.evaluate([x_test_1mer, x_test_4mer], y_test, verbose=1)
print(f'测试评分: {combined_model.metrics_names[0]} 为 {scores[0]}; {combined_model.metrics_names[1]} 为 {scores[1] * 100}%')

# 生成预测结果
y_pred = combined_model.predict([x_test_1mer, x_test_4mer])

# 计算并打印度量标准
accuracy, mcc, auc, precision, recall, specificity, sensitivity = calculate_metrics(y_test, y_pred)
print(f'AUC: {auc:.4f}')
print(f'准确率: {accuracy:.4f}')
print(f'敏感性: {sensitivity:.4f}')
print(f'特异性: {specificity:.4f}')
print(f'MCC: {mcc:.4f}')
print(f'精确度: {precision:.4f}')
print(f'召回率: {recall:.4f}')
