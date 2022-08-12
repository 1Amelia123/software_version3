import streamlit as st
import base64
import pandas as pd
path = r'任务一data.xlsx'
df = pd.read_excel(path,usecols=None)   # 直接使用 read_excel() 方法读取, 不读取列名
lines = df.values.tolist()
print(len(lines))
result = []
username = '1000000001'
count = 0
money = 0
i = 0
ls = []
# 数据预处理
for line in lines:
    i += 1
    if i != len(lines)-1:
        if str(line[0]) == username:
            count = count + 1
            money += int(line[2])
        else:
            ls.append(username)
            ls.append(count)
            ls.append(money)
            result.append(ls)
            username = str(line[0])
            ls = []
            count = 1
            money = int(line[2])
    else:
        ls.append(username)
        ls.append(count)
        ls.append(money)
        result.append(ls)


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

fileName = 'users.xlsx'
pd_toExcel(result, fileName)


st.markdown('''## <b style="color:white;"><center>1.任务简介</center></b>''',unsafe_allow_html=True)
st.markdown('''### <b style="color:white;">需求分析</b>''',unsafe_allow_html=True)
st.markdown('''<b style="color:white;">由测试数据集中的数据，计算出平均缴费金额和平均缴费次数，并以csv格式输出结果保存。</b>''',unsafe_allow_html=True)
st.markdown('''### <b style="color:white;">方案设计</b>''',unsafe_allow_html=True)
st.markdown('''<b style="color:white;">利用python的panda和numpy库，先统计每位用户的缴费次数和缴费总金额，生成“user.xlsx”文件。之后由“user.xlsx”文件中的数据可以计算得到用户的平均缴费次数和平均缴费金额。</b>''',unsafe_allow_html=True)
st.markdown('''#### <b style="color:white;">users.xlsx</b>''',unsafe_allow_html=True)
df = pd.read_excel('users.xlsx')
st.dataframe(df)

st.markdown('''## <b style="color:white;"><center>2.输出结果</center></b>''',unsafe_allow_html=True)

lines = df.values.tolist()
print(len(lines))
result = []
counts = 0
money = 0
i = 0
ls = []
for line in lines:
    counts += int(line[1])
    money += int(line[2])
avg_counts = counts/100
avg_money = money/100
money=[]
money.append(avg_money)
counts=[]
counts.append(avg_counts)
dic = {"平均缴费金额":money,"平均缴费次数":counts}
df=pd.DataFrame(dic,index=None)
st.dataframe(df)

product_list = ["平均缴费金额","平均缴费次数"]
product_type = st.selectbox(
    "请选择：",
    product_list
)
# 数据写入
import csv
#  创建csv文件对象，u是写入中文，encoding='utf-8'是设置编码格式，newline=''为了防止空行
f = open(u'居民客户的用电缴费习惯分析 1.csv', 'w', encoding='utf-8-sig', newline='')
csv_write = csv.writer(f)
csv_write.writerow(['平均缴费金额', '平均缴费次数'])
csv_write.writerow([avg_money, avg_counts])

st.write(f'''<b style="color: white;">结果为:{dic[product_type]}</b>''', unsafe_allow_html=True)
st.write('''<b style="color:white;">点击链接可以下载表格</b>''', unsafe_allow_html=True)
data = open('居民客户的用电缴费习惯分析 1.csv','rb').read()#以只读模式读取且读取为二进制文件
b64 = base64.b64encode(data).decode('UTF-8') # 解码并加密为base64
href = f'<a href="data:file/data;base64,{b64}" download = "居民客户的用电缴费习惯分析 1.csv"> 下载 “居民客户的用电缴费习惯分析 1.csv” </a>' #定义下载链接，默认的下载文件名是myresults.xlsx
st.markdown(href, unsafe_allow_html=True)
# 修改背景样式
img_url = 'https://img.zcool.cn/community/0156cb59439764a8012193a324fdaa.gif'
st.markdown('''<style>.css-fg4pbf{background-image:url(''' + img_url + ''');
background-size:100% 100%;background-attachment:fixed;}</style>
''', unsafe_allow_html=True)
