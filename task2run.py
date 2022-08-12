import streamlit as st
import pandas as pd
import base64
df = pd.read_excel('users.xlsx')
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
# st.markdown('''<b style="color:white;"><center>任务2</center></b>''',unsafe_allow_html=True)
st.markdown('''## <b style="color:white;"><center>1.任务简介</center></b>''',unsafe_allow_html=True)
st.markdown('''### <b style="color:white;">需求分析</b>''',unsafe_allow_html=True)
st.write('''<b style="color:white;">根据\"客户类型区分原则\"，由每位客户缴费次数和缴费金额，对每个居民客户的用电缴费情况进行归类，并以csv格式输出结果保存。</b>''',unsafe_allow_html=True)
st.markdown('''### <b style="color:white;">方案设计</b>''',unsafe_allow_html=True)
st.write('''<b style="color:white;">根据\"客户类型区分原则\"，利用python程序，将每一位客户的缴费次数与金额和平均值进行对比，最终可以得到各用户的客户类型，并输出为csv格式文件。</b>''',unsafe_allow_html=True)
st.markdown('''## <b style="color:white;"><center>2.输出结果</center></b>''',unsafe_allow_html=True)
type = ""
for line in lines:
    ls = []
    if line[1] <= avg_counts:
        if line[2] <= avg_money:
            type = "低价值型客户"
            ls.append(str(line[0]))
            ls.append(type)
        else:
            type = "潜力型客户"
            ls.append(str(line[0]))
            ls.append(type)
    else:
        if line[2] <= avg_money:
            type = "大众型客户"
            ls.append(str(line[0]))
            ls.append(type)
        else:
            type = "高价值型客户"
            ls.append(str(line[0]))
            ls.append(type)
    result.append(ls)
# 数据写入
# import csv
# #  1.创建csv文件对象，encoding='utf-8'是设置编码格式，newline=''为了防止空行
# f = open(u'居民客户的用电缴费习惯分析 2.csv', 'w', encoding='utf-8-sig', newline='') #居民客户的用电缴费习惯分析 2
# #  2.基于文件对象构建csv写入对象
# csv_write = csv.writer(f)
# #  3.构建列表头
# csv_write.writerow(['用户编号', '客户类型'])
# for data in result:
#     #  4.写入csv文件
#     csv_write.writerow([str(data[0]), data[1]])
r2 = pd.read_csv('居民客户的用电缴费习惯分析 2.csv')
st.markdown('''### <b style="color:white;">用户对应类型表</b>''',unsafe_allow_html=True)
st.dataframe(r2)
st.write('''<b style="color:white;">点击链接可以下载表格</b>''', unsafe_allow_html=True)
data=open('居民客户的用电缴费习惯分析 2.csv','rb').read() # 以只读模式读取且读取为二进制文件
b64 = base64.b64encode(data).decode('UTF-8') # 解码并加密为base64
href = f'<a href="data:file/data;base64,{b64}" download = "居民客户的用电缴费习惯分析 2.csv"> 下载 “居民客户的用电缴费习惯分析 2.csv” </a>' #定义下载链接，默认的下载文件名是myresults.xlsx
st.markdown(href, unsafe_allow_html=True)  # 输出到浏览器

# 筛选客户类型
product_list = r2["用户编号"].unique()
st.markdown('''<b style="color:white;"><center>您可以在选择用户编号,并得到其对应的类型：</center></b>''',unsafe_allow_html=True)
product_type = st.selectbox("",product_list)

part_df = str(r2[(r2["用户编号"] == product_type)])
r = part_df.split(" ")
# s="该"
# st.write(f'该用户客户类型为<b style="white color;">：{r[-1]}</b>',unsafe_allow_html=True)
st.write(f'''<b style="color: white;">该用户客户类型为:{r[-1]}</b>''', unsafe_allow_html=True)


# 修改背景样式
img_url = 'https://img.zcool.cn/community/0156cb59439764a8012193a324fdaa.gif'
st.markdown('''<style>.css-fg4pbf{background-image:url(''' + img_url + ''');
background-size:100% 100%;background-attachment:fixed;}</style>
''', unsafe_allow_html=True)
