import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("TKAgg")
from PIL import Image
st.sidebar.expander('')
st.markdown('''## <b style="color:white;"><center>1.任务简介</center></b>''', unsafe_allow_html=True)  # 展示任务
st.markdown('''### <b style="color:white;">需求分析</b>''', unsafe_allow_html=True)
st.write('''<b style="color:white;text-align:justify; text-justify:inter-ideograph;">根据不同的标准对用户进行集群划分，如某一用户的行为特征、用户基本属性、电器设备使用、用电曲线形态等，保存为\"电力用户集群分析模型.mdl\"。</b>''',unsafe_allow_html=True)
st.markdown('''### <b style="color:white;">方案设计</b>''', unsafe_allow_html=True)
# st.header("")
st.write('''<b style="color:white;text-align:justify; text-justify:inter-ideograph;">选取江苏省扬中市高新区1000多家企业每日用电量的数据，数据集中包含用户编号，日期，以及用电量。我们使用用户的用电曲线形态来对电力用户进行集群分析。首先对数据进行预处理，将用户编号一致的放在一起作为一个样本数据，每月用电量的均值作为样本数据并对数据进行归一化处理，然后使用K-means算法进行聚类分析。选取合适的k值，对用户进行分类，绘制电力用户曲线，并保存模型。</b>''',
         unsafe_allow_html=True)
# st.write("选取江苏省扬中市高新区1000多家企业每日用电量的数据，数据集中包含用户编号，日期，以及用电量。我们使用用户的用电曲线形态来对电力用户进行集群分析。首先对数据进行预处理，将用户编号一致的放在一起作为一个样本数据，每月用电量的均值作为样本数据并对数据进行归一化处理，然后使用K-means算法进行聚类分析。选取合适的k值，对用户进行分类，绘制电力用户曲线，并保存模型。")
st.markdown('''## <b style="color:white;"><center>2.详细实现</center></b>''', unsafe_allow_html=True)
# st.header("3.详细实现")
st.markdown('''### <b style="color:white;">数据预处理</b>''', unsafe_allow_html=True)
# st.subheader("3.1数据预处理")
st.write('''<b style="color:white;text-align:justify; text-justify:inter-ideograph;">将用户编号一致的放在一起作为一个样本数据，由于2016年的数据不全，因此这里选择使用2015年的数据，如果选择每天的数据，数据量过大，因此考虑将其按月进行考虑。即把每月每天的用电量相加然后除以天数得到每月用电量的均值，之后对数据进行归一化处理。</b>''',unsafe_allow_html=True)
# st.write("将用户编号一致的放在一起作为一个样本数据，由于2016年的数据不全，因此这里选择使用2015年的数据，如果选择每天的数据，数据量过大，因此考虑将其按月进行考虑。即把每月每天的用电量相加然后除以天数得到每月用电量的均值，之后对数据进行归一化处理。")
# st.subheader("3.2模型结构")
st.markdown('''### <b style="color:white;">模型结构</b>''', unsafe_allow_html=True)
image = Image.open('任务五的模型结构.jpg')
st.image(image, caption='任务五的模型结构',use_column_width=True)
# 设置侧边栏     Tips:所有侧边栏的元素都必须在前面加上 sidebar，不然会在主页显示
st.sidebar.expander('')  # expander必须接受一个 label参数，我这里留了一个空白
st.sidebar.subheader('在下方调节你的参数')  # 副标题
# st.selectbox:创造一个下拉选择框的单选题，接收参数: (题目名称， 题目选项)
moxing=st.sidebar.radio('请选择是否需要使用已经训练好的模型:',['否','是'])
fangshi = st.sidebar.selectbox('请选择聚类方式:', ['对一年的数据进行聚类', '对第一季度的数据聚类', '对第二季度的数据聚类', '对第三季度的数据聚类', '对第四季度的数据聚类'])
shouzhou = st.sidebar.radio('是否需要通过手肘法选取k值', ['是', '否'])
cluster_class = st.sidebar.selectbox('请设置聚类数量:', list(range(1, 10)))  # 选择聚类中心，并赋值
minmaxscaler = st.sidebar.radio('请选择是否进行归一化:', ['是', '否'])
disply = st.sidebar.radio('请选择是否需要展示聚类中心:', ['是', '否'])

# ---------这里是训练数据
# 聚类实现
from sklearn.cluster import KMeans
Shu=np.load('year_data.npy')
markersize=3
if fangshi=='对一年的数据进行聚类':
    Shu=np.load('year_data.npy')
    markersize=3
elif fangshi=='对第一季度的数据聚类':
    Shu=np.load('jidu1_data.npy')
    markersize=0.5
elif fangshi=='对第二季度的数据聚类':
    Shu=np.load('jidu2_data.npy')
    markersize=0.5
elif fangshi=='对第三季度的数据聚类':
    Shu=np.load('jidu3_data.npy')
    markersize=0.5
elif fangshi=='对第四季度的数据聚类':
    Shu=np.load('jidu4_data.npy')
    markersize=0.5

# --------下面这里是为了画图的好看
def trans(select):
    if select == '黑色':
        return 'black'
    elif select == '银色':
        return 'silver'
    elif select == '亮红色':
        return 'lightcoral'
    elif select == '棕色':
        return 'brown'
    elif select == '橙色':
        return 'orange'
    elif select == '金黄色':
        return 'gold'
    elif select == '黄色':
        return 'yellow'
    elif select == '绿色':
        return 'lawngreen'
    elif select == '天蓝色':
        return 'cyan'
    elif select == '紫色':
        return 'purple'
    elif select=='粉色':
        return 'pink'
    elif select == '圆形':
        return 'o'
    elif select == '朝下三角':
        return 'v'
    elif select == '朝上三角形':
        return '^'
    elif select == '正方形':
        return 's'
    elif select == '五边形':
        return 'p'
    elif select == '星型':
        return '*'
    elif select == '六角形':
        return 'h'
    elif select == '+号':
        return '+'
    elif select == 'x号':
        return 'x'
    elif select == '小型菱形':
        return 'd'

if shouzhou == '是':
    # st.subheader('3.3k值选取')
    st.markdown('''### <b style="color:white;">k值选取</b>''', unsafe_allow_html=True)
    st.markdown('''#### <b style="color:white;"><center>不同k-值对应距离平方和关系图</center></b>''', unsafe_allow_html=True)
    scope = range(1, 15)
    sse = []
    # 对数据进行归一化处理
    s = StandardScaler()
    Shu = s.fit_transform(Shu)
    for k in scope:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(Shu)
        sse.append(kmeans.inertia_)

    loss_data = pd.DataFrame(
        sse,
        columns=["平方值"]
    )
    st.line_chart(loss_data)

if minmaxscaler=='是':
    s = StandardScaler()
    Shu = s.fit_transform(Shu)
estimator = KMeans(n_clusters=cluster_class)
estimator.fit(Shu)
label_pred=estimator.predict(Shu)
# ----------
choice = pd.DataFrame([])
for i in range(1, cluster_class + 1):
    col1, col2 = st.sidebar.columns(2)
    with col1:
        choice.loc[i, 'color'] = trans(st.selectbox(f'第{i}类颜色', ['黑色', '亮红色', '棕色', '橙色', '粉色', '黄色',
                                                                 '天蓝色', '紫色', '银色']))
        with col2:  # 第二列的东西
            choice.loc[i, 'shape'] = trans(st.selectbox(f'第{i}类形状', ['圆形', '朝下三角', '朝上三角形', '正方形', '五边形', '星型'
                , '六角形', '+号', 'x号', '小型菱形']))
model = KMeans(n_clusters=cluster_class).fit(Shu)
data_done = np.c_[Shu, model.labels_]
# --------
l1,l2,l3,l4,l5,l6,l7,l8=0,0,0,0,0,0,0,0
fig, ax = plt.subplots()
for i in range(len(label_pred)):#遍历每一个用户
    if label_pred[i] == 0:
        l1=l1+1
        color=choice.loc[0+1,'color']
        shape=choice.loc[0+1,'shape']
        plt.plot(range(len(Shu[i])), Shu[i], color=color,marker=shape,markersize=markersize)
        ax.plot(range(len(Shu[i])), Shu[i], color=color,marker=shape,markersize=markersize)
    elif label_pred[i] == 1:
        l2=l2+1
        color = choice.loc[1 + 1, 'color']
        shape = choice.loc[1 + 1, 'shape']
        plt.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape,markersize=markersize)
        ax.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape,markersize=markersize)
    elif label_pred[i] == 2:
        l3=l3+1
        color = choice.loc[2 + 1, 'color']
        shape = choice.loc[2 + 1, 'shape']
        plt.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape,markersize=markersize)
        ax.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape,markersize=markersize)
    elif label_pred[i]==3:
        l4=l4+1
        color = choice.loc[3 + 1, 'color']
        shape = choice.loc[3 + 1, 'shape']
        plt.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape,markersize=markersize)
        ax.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape,markersize=markersize)
    elif label_pred[i]==4:
        l5=l5+1
        color = choice.loc[4 + 1, 'color']
        shape = choice.loc[4 + 1, 'shape']
        plt.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape,markersize=markersize)
        ax.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape,markersize=markersize)
    elif label_pred[i]==5:
        l6=l6+1
        color = choice.loc[5 + 1, 'color']
        shape = choice.loc[5 + 1, 'shape']
        plt.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape,markersize=markersize)
        ax.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape,markersize=markersize)
    elif label_pred[i]==6:
        l7=l7+1
        color = choice.loc[6 + 1, 'color']
        shape = choice.loc[6 + 1, 'shape']
        plt.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape,markersize=markersize)
        ax.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape,markersize=markersize)
    elif label_pred[i]==7:
        l8=l8+1
        color = choice.loc[7 + 1, 'color']
        shape = choice.loc[7 + 1, 'shape']
        plt.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape,markersize=markersize)
        ax.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape,markersize=markersize)

st.markdown('''## <b style="color:white;"><center>3.结果分析</center></b>''', unsafe_allow_html=True)
st.markdown('''#### <b style="color:white;"><center>用户负荷曲线聚类示意图</center></b>''', unsafe_allow_html=True)

st.pyplot(fig)


if disply=='是':
    st.markdown('''### <b style="color:white;">聚类中心</b>''', unsafe_allow_html=True)
    st.write(model.cluster_centers_)
if fangshi=='对一年的数据进行聚类':
    # st.subheader("4.3用户类型分类")
    st.markdown('''### <b style="color:white;">用户类型分类</b>''', unsafe_allow_html=True)

    st.write(''' <b style="color:white;text-align:justify; text-justify:inter-ideograph;">(1)第一类用户用电曲线，这类企业用电量在一年内的变化不明显，同时整体用电量较低，可以看到其应该为小型生产企业类型。</b>''',unsafe_allow_html=True)
    st.write(''' <b style="color:white;text-align:justify; text-justify:inter-ideograph;">(2)第二类用户用电曲线，这类企业在1月份的时候用电量出现了明显下降，因此这类曲线对应的用户可能是在冬季需要减少相关生产的企业类型。</b>''',unsafe_allow_html=True)
    st.write(''' <b style="color:white;text-align:justify; text-justify:inter-ideograph;">(3)第三类用户用电曲线，这类企业在一年内的变化也不明显，但是其整体用电量要比红色曲线的高。因此可以认为其是类似煤矿企业等。</b>''',unsafe_allow_html=True)
    st.write(''' <b style="color:white;text-align:justify; text-justify:inter-ideograph;">(4)第四类用户用电曲线，这类企业在一年内处于一个波动状态，同时其整体用电量较高，因此可以认为其是耗电量特大的如电石、电介铝等企业。</b>''',unsafe_allow_html=True)

st.markdown('''#### <b style="color:white;"><center>上传需要聚类的数据,可以得到其对应类别:</center></b>''', unsafe_allow_html=True)
file=st.file_uploader("", type=None, accept_multiple_files=False, key=None, \
                     help=None, on_change=None)
if moxing=='是':
    if fangshi == '对一年的数据进行聚类':
        estimator = joblib.load("电力用户集群模型.mkl")
        markersize = 3
    elif fangshi == '对第一季度的数据聚类':
        estimator=joblib.load("电力用户集群模型_1.mkl")
        markersize = 0.5
    elif fangshi == '对第二季度的数据聚类':
        estimator=joblib.load("电力用户集群模型_2.mkl")
        markersize = 0.5
    elif fangshi == '对第三季度的数据聚类':
        estimator=joblib.load("电力用户集群模型_3.mkl")
        markersize = 0.5
    elif fangshi == '对第四季度的数据聚类':
        estimator=joblib.load("电力用户集群模型_4.mkl")
        markersize = 0.5
if file is not None:
    data=np.load(file)
    label_pred=estimator.predict(data)
    data=pd.DataFrame(data)
    st.write('''<b style="color:white;"><center>上传的数据</center></b>''', unsafe_allow_html=True)
    st.write(data)
    chart_data_yu = pd.DataFrame(
        label_pred,
        columns=["类别"]
    )
    st.write('''<b style="color:white;"><center>聚类结果展示</center></b>''', unsafe_allow_html=True)
    fig, ax = plt.subplots()
    for i in range(len(label_pred)):  # 遍历每一个用户
        if label_pred[i] == 0:
            l1 = l1 + 1
            color = choice.loc[0 + 1, 'color']
            shape = choice.loc[0 + 1, 'shape']
            plt.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape, markersize=markersize)
            ax.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape, markersize=markersize)
        elif label_pred[i] == 1:
            l2 = l2 + 1
            color = choice.loc[1 + 1, 'color']
            shape = choice.loc[1 + 1, 'shape']
            plt.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape, markersize=markersize)
            ax.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape, markersize=markersize)
        elif label_pred[i] == 2:
            l3 = l3 + 1
            color = choice.loc[2 + 1, 'color']
            shape = choice.loc[2 + 1, 'shape']
            plt.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape, markersize=markersize)
            ax.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape, markersize=markersize)
        elif label_pred[i] == 3:
            l4 = l4 + 1
            color = choice.loc[3 + 1, 'color']
            shape = choice.loc[3 + 1, 'shape']
            plt.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape, markersize=markersize)
            ax.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape, markersize=markersize)
        elif label_pred[i] == 4:
            l5 = l5 + 1
            color = choice.loc[4 + 1, 'color']
            shape = choice.loc[4 + 1, 'shape']
            plt.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape, markersize=markersize)
            ax.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape, markersize=markersize)
        elif label_pred[i] == 5:
            l6 = l6 + 1
            color = choice.loc[5 + 1, 'color']
            shape = choice.loc[5 + 1, 'shape']
            plt.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape, markersize=markersize)
            ax.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape, markersize=markersize)
        elif label_pred[i] == 6:
            l7 = l7 + 1
            color = choice.loc[6 + 1, 'color']
            shape = choice.loc[6 + 1, 'shape']
            plt.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape, markersize=markersize)
            ax.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape, markersize=markersize)
        elif label_pred[i] == 7:
            l8 = l8 + 1
            color = choice.loc[7 + 1, 'color']
            shape = choice.loc[7 + 1, 'shape']
            plt.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape, markersize=markersize)
            ax.plot(range(len(Shu[i])), Shu[i], color=color, marker=shape, markersize=markersize)
    st.markdown('''## <b style="color:white;"><center>3.结果分析</center></b>''', unsafe_allow_html=True)
    st.markdown('''#### <b style="color:white;"><center>用户负荷曲线聚类示意图</center></b>''', unsafe_allow_html=True)

    st.pyplot(fig)

    st.write(chart_data_yu)
# 背景图片的网址
img_url = 'https://img.zcool.cn/community/0156cb59439764a8012193a324fdaa.gif'

# 修改背景样式
st.markdown('''<style>.css-fg4pbf{background-image:url(''' + img_url + ''');
background-size:100% 100%;background-attachment:fixed;}</style>
''', unsafe_allow_html=True)

# 侧边栏样式
st.markdown('''<style>#root > div:nth-child(1) > div > div > div > div >
section.css-1lcbmhc.e1fqkh3o3 > div.css-1adrfps.e1fqkh3o2
{background:rgba(255,255,255,0.5)}</style>''', unsafe_allow_html=True)
