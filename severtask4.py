import pandas
import torch.nn as nn
import torch
import numpy as np
import threading
import socket
import datetime
from sklearn.preprocessing import MinMaxScaler
import json
# -----模型参数
input_dim = 7  # 数据的特征数
hidden_dim = 64  # 隐藏层的神经元个数
num_layers = 1  # LSTM的层数
output_dim = 1  # 预测值的特征数
pre_days = 7  # 以1周的数据为一组
# ----------

def socket_service():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 绑定端口为9001
        s.bind(('127.0.0.1', 9001))
        # 设置监听数
        s.listen(10)
    except socket.error as msg:
        print(msg)
    print('Waiting connection...')

    ## 一直开启，监听客户端
    while 1:
        # 等待请求并接受(程序会停留在这一旦收到连接请求即开启接受数据的线程)
        conn, addr = s.accept()
        # 接收数据
        t = threading.Thread(target=deal_data, args=(conn, addr))
        t.start()

## 和客户端的交互函数
def deal_data(conn, addr):
    print('Accept new connection from {0}'.format(addr))
    # conn.settimeout(500)
    # 收到请求后的回复
    conn.send('Hi, Welcome to the server!'.encode('utf-8'))

    ## 接收到训练样本、超参数等
    json_string, addr = conn.recvfrom(8192*2)
    print(json_string)
    mydict = json.loads(json_string)
    print(mydict)
    learning_rate =mydict['learning_rate']
    num_epochs = mydict['epoch']
    data_feat1= mydict['data_feat']
    fangshi=mydict['fangshi']
    seq=mydict['seq']
    scaler = MinMaxScaler(feature_range=(-1, 1))
    sel_col = ["power_consumption", "low_temp", "high_temp", "kind", "wind", "level", "holiday"]

    df_main={"power_consumption":data_feat1[0],"low_temp":data_feat1[1],"high_temp":data_feat1[2],"kind":data_feat1[3],"wind":data_feat1[4],"level":data_feat1[5],"holiday":data_feat1[6]}
    df_main=pandas.DataFrame(df_main)
    for col in sel_col:
        df_main[col] = scaler.fit_transform(df_main[col].values.reshape(-1, 1))
    df_main['target'] = df_main['power_consumption'].shift(-1)
    data_feat,data_target=[],[]
    for index in range(len(df_main) - seq):
        # 构建特征集
        data_feat.append((df_main[["power_consumption", "low_temp", "high_temp", "kind", "wind", "level", "holiday"]][
                         index: index + seq].values).tolist())
        # 构建target集
        data_target.append((df_main['target'][index:index + seq].values).tolist())
    # 将特征集和标签集整理成numpy数组
    data_feat = np.array(data_feat)
    data_target = np.array(data_target)
    # 这里按照8:2的比例划分训练集和测试集
    if fangshi == "8:2":
        test_set_size = 122  # np.round(1)是四舍五入，
        train_size = data_feat.shape[0] - (test_set_size)
    elif fangshi == "7:3":
        test_set_size = 183  # np.round(1)是四舍五入，
        train_size = data_feat.shape[0] - (test_set_size)
    elif fangshi == "6:4":
        test_set_size = 244  # np.round(1)是四舍五入，
        train_size = data_feat.shape[0] - (test_set_size)
    trainX = torch.from_numpy(data_feat[:train_size].reshape(-1, seq, 7)).type(torch.Tensor)
    testX = torch.from_numpy(data_feat[train_size:].reshape(-1, seq, 7)).type(torch.Tensor)
    trainY = torch.from_numpy(data_target[:train_size].reshape(-1, seq, 1)).type(torch.Tensor)
    testY = torch.from_numpy(data_target[train_size:].reshape(-1, seq, 1)).type(torch.Tensor)
    xieruwenjian = torch.from_numpy(data_feat[:10].reshape(-1, seq, 7)).type(torch.Tensor)
    torch.save(xieruwenjian, '任务四的测试')
    total = []
    class LSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
            super(LSTM, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5)
            # 全连接层
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
            out = self.fc(out)
            return out

    # 封装模型
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    # 定义优化器和损失函数
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 使用Adam优化算法
    # 模型评估条件
    loss_fn = torch.nn.MSELoss(size_average=True)  # 使用均方差作为损失函数
    hist = np.zeros(num_epochs)
    ls = []
    time=[]
    result={}
    starttime = datetime.datetime.now()
    for t in range(num_epochs):
        y_train_pred = model(trainX)
        loss = loss_fn(y_train_pred, trainY)
        ls.append(float(loss.item()))
        endtime = datetime.datetime.now()
        spendtime = (endtime - starttime).seconds
        time.append(spendtime)
        if t % 10 == 0 and t != 0:  # 每训练十次，打印一次均方差
            print("Epoch ", t, "MSE: ", loss.item())

        hist[t] = loss.item()
        # 梯度归零
        optimiser.zero_grad()
        # Backward
        loss.backward()
        # 更新参数
        optimiser.step()
    list1 = []
    list2 = []


    print(type(ls[0]))
    y_test_pred = model(testX)
    y_train_pred = model(trainX)
    loss1 = loss_fn(y_test_pred[:-pre_days], testY[:-pre_days]).item()
    print(loss1)
    pt = []
    test_pre = y_test_pred.detach().numpy()[:, -1, 0]
    test_ture = testY.detach().numpy()[:, -1, 0]
    listb=[]
    for i in range(len(test_pre)):
        a=[]
        a.append(float(test_pre[i]))
        a.append(float(test_ture[i]))
        listb.append(a)
    result['train_loss']=ls
    result['pre_true']=listb
    result['train_time']=time
    endtime = datetime.datetime.now()
    spendtime = (endtime - starttime).seconds
    result['spendtime']=spendtime
    print(result)
    json_string = json.dumps(result)
    conn.send(str(len(json_string.encode())).encode())
    conn.send(json_string.encode())
    conn.close()
    print('传输完成')
# #
if __name__ == "__main__":
    socket_service()