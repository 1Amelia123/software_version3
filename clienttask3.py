#!coding=utf-8
import socket
import sys
import json
import base64
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("TKAgg")
def socket_client(ip,epoch, learning_rate):
    ## 开始链接，127.0.0.1，9001表示本机的端口这里不用修改
    try:
        s = socket.socket()
        s.connect(ip)
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    ## 服务器第一次回信，会收到welcome代表连上
    print(s.recv(1024))
    ## 给边缘设备发送的内容：epoch数、learning_rate大小，以及从excel读取的data数据，存放在send_dic字典中（可添加新的内容：send_dic['xxx']=xxx）
    param = pd.read_excel('param.xlsx')
    param['迭代次数'][0] = epoch
    param['学习率'][0] = learning_rate
    df = pd.read_excel('任务三data.xlsx')
    id = 1000000001

    data = []
    true_data = []
    for i in range(len(df['用户编号'])):
        if i != len(df['用户编号']) - 1:
            if df['用户编号'][i] == id:
                true_data.append(int(df['缴费金额（元）'][i]))
            else:
                id = df['用户编号'][i]
                data.append(true_data)
                true_data = []
                true_data.append(int(df['缴费金额（元）'][i]))
        else:
            true_data.append(int(df['缴费金额（元）'][i]))
            data.append(true_data)
    print(data)
    send_dic = {}
    send_dic['data']=data
    send_dic['learning_rate']=learning_rate
    send_dic['epoch']=epoch
    ## 将字典打包为字符串发过去
    json_string = json.dumps(send_dic)
    s.send(json_string.encode())

    ## 接受边缘设备的输出结果：画图所需的10个用户的损失（result['loss']），训练时间(result['spend_time'])，以及预测值(result['pred'],这个就是你们之前写的存放在total数组中的数据)
    ## 8192表示一次可以接受到最大的byte数为8192，长度太长会被截断，可以调整
    result_length = int(s.recv(1024))
    print(result_length)
    result_byte = bytes("","utf-8")
    while True:
        result_byte += s.recv(1024)
        if len(result_byte) == result_length:
            break
    result = json.loads(result_byte)

    s.close()
    return result

# 更新用户信息后写入
def new_user(total):
    # 数据写入
    import pandas as pd
    path = r'users.xlsx'
    df = pd.read_excel(path, usecols=None)  # 直接使用 read_excel() 方法读取, 不读取列名
    lines = df.values.tolist()
    user = []
    i = 0
    for line in lines:
        ls1 = []
        c = line[1] + 1  # 缴费次数加1
        m = line[2] + total[i][0]  # 金额加预测值
        ls1.append(line[0])
        ls1.append(c)
        ls1.append(m)
        user.append(ls1)
        i += 1
    print(user)
    # 数据写入
    # -*- coding: utf-8 -*-
    import pandas as pd
    def pd_toExcel(data, fileName):  # pandas库储存数据到excel
        ids = []
        counts = []
        prices = []
        for i in range(len(data)):
            ids.append(data[i][0])
            counts.append(data[i][1])
            prices.append(data[i][2])
        dfData = {  # 用字典设置DataFrame所需数据
            '用户编号': ids,
            '缴费次数': counts,
            '缴费金额': prices
        }
        df = pd.DataFrame(dfData)  # 创建DataFrame
        df.to_excel(fileName, index=False)  # 存表，去除原始索引列（0,1,2...）

    fileName = 'user_pred.xlsx'
    pd_toExcel(user, fileName)
    return user
# -----------
def output(path):
    # df=data
    df = pd.read_excel(path, usecols=None)  # 直接使用 read_excel() 方法读取, 不读取列名
    lines = df.values.tolist()
    result = []
    avg_counts = 6.62
    avg_money = 702.31

    # 客户分类
    # 先除去已经是高价值型的客户
    df = pd.read_csv(r'居民客户的用电缴费习惯分析 2.csv',encoding='utf-8')
    high = []
    for i in range(100):
        if df['客户类型'][i] == "高价值型客户":
            high.append(df['用户编号'][i])
    # print(high)
    for line in lines:
        if line[0] not in high:
            ls = []
            if line[1] > avg_counts and line[2] > avg_money:
                type =  "高价值型客户"
                ls.append(str(line[0]))
                ls.append(line[2])
                ls.append(type)
                result.append(ls)

    # 按金额排序
    for i in range(len(result)):
        for j in range(i + 1, (len(result))):
            if result[i][1] < result[j][1]:
                result[i][1], result[j][1] = result[j][1], result[i][1]
                result[i][0], result[j][0] = result[j][0], result[i][0]
    # print(result)
    f_result = []
    for i in range(5):
        a = [i+1,result[i][0],result[i][2],result[i][1] ]
        f_result.append(a)
    print(f_result)
    # 数据写入
    import csv
    f = open(u'居民客户的用电缴费习惯分析 3.csv', 'w', encoding='utf-8-sig', newline='')
    csv_write = csv.writer(f)
    csv_write.writerow(['用户排名', '用户编号', '客户类型', '缴费金额'])
    for data in f_result:
        csv_write.writerow([data[0], data[1],data[2],data[3]])


# ----------一下是布局
# streamlit的页面布局顺序是与代码位置一致的，因此我先在最前面加一个大标题
# def task5():
st.sidebar.expander('')
st.sidebar.expander('')
st.sidebar.subheader('在下方调节你的参数')
wenjian = st.sidebar.radio('是否选择上传本地测试文件', ['是', '否'])
number = st.sidebar.number_input('请输入迭代次数:')
epoch=int(number)
learning=st.sidebar.number_input('请输入学习率:')
learning_rate=learning
device=st.sidebar.selectbox('请选择运行的设备',['本地','jetson-4G','jetson-8G','树莓派2G','树莓派4G'])
if device=='本地':
    ip=('127.0.0.1', 9001)
elif device=='jetson-4G':
    ip=('127.0.0.1', 9001)
elif device=='jetson-8G':
    ip=('127.0.0.1', 9001)
elif device=='树莓派2G':
    ip=('127.0.0.1', 9001)
elif device == '树莓派4G':
    ip = ('127.0.0.1', 9001)

look_back=st.sidebar.radio('请输入考虑之前数据的组数',list(range(2,5)))

st.markdown('''## <b style="color:white;"><center>1.任务简介</center></b>''', unsafe_allow_html=True)
st.markdown('''### <b style="color:white;">需求分析</b>''', unsafe_allow_html=True)
st.write('''<b style="color:white;">由测试数据集，采用时间序列分析方法，训练得到用户价值预测模型，预测出最有可能成为高价值客户的前五人，将结果以csv格式保存。</b>''', unsafe_allow_html=True)
st.markdown('''### <b style="color:white;">方案设计</b>''', unsafe_allow_html=True)
st.write('''<b style="color:white;">首先利用LSTM模型预对每位用户下一次缴费的金额进行时间序列预测，之后在结合任务1、2中所得的数据，确定出最有可能成为高价值类型客户的top5。</b>''', unsafe_allow_html=True)
st.markdown('''## <b style="color:white;"><center>2.原始数据</center></b>''',unsafe_allow_html=True)
st.write('''<b style="color:white;">本地数据是每位用户在2018和2019年期间的购电日期和每次购买金额的数据,您也可以选择上传自己的本地文件。展示如下：</b>''', unsafe_allow_html=True)


def main():
    # st.dataframe(df)
    st.markdown('''## <b style="color:white;"><center>3.预测过程</center> </b>''', unsafe_allow_html=True)
    param = pd.read_excel('param.xlsx')
    param['迭代次数'][0] = epoch
    param['学习率'][0] = learning_rate
    st.markdown('''#### <b style="color:white;">LSTM模型的具体参数设置</b>''', unsafe_allow_html=True)
    if epoch != 0:
        st.dataframe(param)
        st.markdown('''#### <b style="color:white;"><center>训练过程损失曲线</center></b>''', unsafe_allow_html=True)
        result=socket_client(ip,epoch,learning_rate)
        train_loss=result['train_loss']
        pred=result['pred']

        for i in range(0,10):
            st.write(f'''<b style="color: white;">用户{i+1}训练过程的损失曲线</b>''' , unsafe_allow_html=True)
            # result=socket_client()
            ls=train_loss[i]
            loss_data = pd.DataFrame(
                ls,
                columns=["loss值"]
            )
            st.line_chart(loss_data)
        st.markdown('''#### <b style="color:white;"><center>用户缴费金额实际值与预测值对比曲线</center></b>''', unsafe_allow_html=True)
        st.image('预测值与真实值.png')

        st.write('''<b style="color:white;"> 在对用户下一次缴费金额预测完毕后，更新客户的数据信息，将预测的数据添加到原有的客户信息中，得到新的用户数据文件。</b>''',
                 unsafe_allow_html=True)
        new = new_user(pred)
        st.write('''<b style="color:white;">\"user_pred.xlsx\"</b>''')
        st.dataframe(new)
        st.markdown('''## <b style="color:white;"><center>4.预测结果</center></b>''', unsafe_allow_html=True)
        path = r'user_pred.xlsx'
        output(path)
        result = pd.read_csv('居民客户的用电缴费习惯分析 3.csv')
        st.dataframe(result)
        st.write('''<b style="color:white;">点击链接可以下载表格</b>''', unsafe_allow_html=True)
        data = open('居民客户的用电缴费习惯分析 3.csv', 'rb').read()  # 以只读模式读取且读取为二进制文件
        b64 = base64.b64encode(data).decode('UTF-8')  # 解码并加密为base64
        href = f'<a href="data:file/data;base64,{b64}" download = "居民客户的用电缴费习惯分析 3.csv"> 下载 “居民客户的用电缴费习惯分析 3.csv” </a>'  # 定义下载链接，默认的下载文件名是myresults.xlsx
        st.markdown(href, unsafe_allow_html=True)  # 输出到浏览器
if wenjian=='是':
    st.write(''' <b style="color:white;"><center>请上传本地文件</center></b>''', unsafe_allow_html=True)
    file=st.file_uploader("", type=None, accept_multiple_files=False, key=None, \
                     help=None, on_change=None)
    if file is not None:
        df= pd.read_excel(file)
        st.dataframe(df)
        main()
else:
    df = pd.read_excel('任务三data.xlsx')
    st.dataframe(df)
    main()

# 修改背景样式
img_url = 'https://img.zcool.cn/community/0156cb59439764a8012193a324fdaa.gif'
st.markdown('''<style>.css-fg4pbf{background-image:url(''' + img_url + ''');
background-size:100% 100%;background-attachment:fixed;}</style>
''', unsafe_allow_html=True)




