#!coding=utf-8

from torch.autograd import Variable
import torch.nn as nn
from pandas import DataFrame
import numpy as np
import torch
import threading
import socket
import datetime
import json

## 你们之前写的训练和预测函数，删去了画图的那一部分，添加了一个新的返回值ls，应该不需要改动
def predict(df1,learning_rate,epoch,look_back,dijige):
    # 一、数据准备
    # print(df1)
    data = df1.values
    # print(datas)
    # 归一化处理，这一步必不可少，不然后面训练数据误差会很大，模型没法用
    max_value = np.max(data)
    min_value = np.min(data)
    scalar = max_value - min_value
    datas = list(map(lambda x: x / scalar, data))
    # print(datas)
    # 数据集和目标值赋值，dataset为数据，look_back为以几行数据为特征维度数量
    def creat_dataset(dataset, look_back):
        data_x = []
        data_y = []
        for i in range(len(dataset) - look_back):
            data_x.append(dataset[i:i + look_back])
            data_y.append(dataset[i + look_back])
        data_x.append(dataset[len(dataset) - look_back:len(dataset) - look_back + look_back])
        return np.asarray(data_x), np.asarray(data_y)  # 转为ndarray数据

    # 以2为特征维度，得到数据集
    dataX, dataY = creat_dataset(datas, look_back)
    # print(dataX)
    # print(dataY)
    train_size = int(len(dataX) * 0.7)
    x_train = dataX[:train_size]  # 训练数据
    y_train = dataY[:train_size]  # 训练数据目标值
    x_train = x_train.reshape(-1, 1, 2)  # 将训练数据调整成pytorch中lstm算法的输入维度
    y_train = y_train.reshape(-1, 1, 1)  # 将目标值调整成pytorch中lstm算法的输出维度
    # 将ndarray数据转换为张量，因为pytorch用的数据类型是张量
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)

    # 二、创建LSTM模型
    class RNN(nn.Module):
        def __init__(self):
            super(RNN, self).__init__()  # 面向对象中的继承
            self.lstm = nn.LSTM(2, 6, 2)  # 输入数据2个特征维度，6个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果
            self.out = nn.Linear(6, 1)  # 线性拟合，接收数据的维度为6，输出数据的维度为1

        def forward(self, x):
            x1, _ = self.lstm(x)
            a, b, c = x1.shape
            out = self.out(x1.view(-1, c))  # 因为线性层输入的是个二维数据，所以此处应该将lstm输出的三维数据x1调整成二维数据，最后的特征维度不能变
            out1 = out.view(a, b, -1)  # 因为是循环神经网络，最后的时候要把二维的out调整成三维数据，下一次循环使用
            return out1

    rnn = RNN()

    # rnn = RNN()
    # 参数寻优，计算损失函数
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()

    # 三、训练模型
    ls = []
    time=[]
    starttime_id = datetime.datetime.now()
    for i in range(epoch):
        var_x = Variable(x_train).type(torch.FloatTensor)
        var_y = Variable(y_train).type(torch.FloatTensor)
        out = rnn(var_x)
        loss = loss_func(out, var_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print('Epoch:{}, Loss:{:.5f}'.format(i + 1, loss.item()))
        # ls.append(loss.item()*0.1)
        if (i+1)%10==0:
            ls.append(loss.item()*0.1)
            endtime = datetime.datetime.now()
            spendtime = (endtime - starttime_id).seconds
            time.append(spendtime)
    p_value = []
    for i in range(2):  # 预测用户后两次缴费金额
        dataX1 = dataX.reshape(-1, 1, 2)
        dataX2 = torch.from_numpy(dataX1)
        var_dataX = Variable(dataX2).type(torch.FloatTensor)
        pred = rnn(var_dataX)
        pred_test = pred.view(-1).data.numpy()  # 转换成一维的ndarray数据，这是预测值
        pred_value = list(map(lambda x: round(x * scalar), pred_test))
        # print(pred_value)
        p_value.append(pred_value[-1])
        p = list(map(lambda x: x / scalar, pred_value))
        dataX,y = creat_dataset(p, 2)
    # print("预测值:",p_value)
    return p_value,ls

## 服务器连接函数
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
    json_string, addr = conn.recvfrom(8192)
    mydict = json.loads(json_string)
    print(mydict)
    learning_rate =mydict['learning_rate']
    epoch = mydict['epoch']
    data = mydict['data']
    total = []
    pred = []
    ls = []
    look_back = 2

    result = {}
    result['train_loss']=[]
    starttime = datetime.datetime.now()
    for i in range(len(data)):
        df1 = DataFrame(data[i])
        ls = list(df1[0])
        if (len(df1) > 1):
            # learning_rate=0.0

            pred,loss= predict(df1, learning_rate, epoch, look_back, i)
            if i <10:
                result['train_loss'].append(loss)
        else:
            pred = [ls[0], ls[0], ls[0]]
        total.append(pred)
    result['pred']=total
    endtime = datetime.datetime.now()
    spendtime = (endtime - starttime).seconds
    result['spendtime']=spendtime

    json_string = json.dumps(result)
    conn.send(str(len(json_string.encode())).encode())
    conn.send(json_string.encode())
    conn.close()
    print('传输完成')
#
if __name__ == "__main__":
    socket_service()
