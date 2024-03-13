
#%%%

import pandas as pd
import numpy as np
import japanize_matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
import pandas as pd
import os
import shutil
import zipfile
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from scipy.integrate import simps
from sklearn.metrics import confusion_matrix
import seaborn as sns
pd.set_option('display.max_columns', None)



season = input('season, 1year or summer')
df = pd.read_csv('./csv/' + season + '/val_result_10.csv')
df_test = pd.read_csv('./csv/' + season + '/test_result_10.csv')

print('df', df)

df_max = pd.DataFrame(df.max())
df_max.to_csv('./csv/' + season + '/val_' + season + '_max.csv')

df_min = pd.DataFrame(df.min())
df_min.to_csv('./csv/' + season + '/val_' + season + '_min.csv')

df_mean= pd.DataFrame(df.mean())
# df_mean.to_csv('./csv/' + season + '/val_' + season + '_mean.csv')


a = df['img_path'].str.split('.', expand=True)[0]
a = pd.DataFrame(a)
a.columns = ['img']

df_copy = df.copy()

df_copy = pd.concat([df_copy['img_path'], df_copy['datetime'], a['img']], axis=1)

df = df.drop('datetime', axis=1)
df = df.drop('img_path', axis=1)
df = (df - df.min()) / (df.max() - df.min())


df = pd.concat([df, a], axis=1)
df = pd.DataFrame(df)
df.describe()



def plot_kde(df, colname, xlabel, bins=50, xmin=-0.1, xmax=1.1):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.hist(
        df[colname], 
        label = 'normalized uncertainty',
        bins=bins,
        alpha=0.6, 
        ec='black',
        range = (0, 1)
        )
    
    ax1.set_ylabel('度数', fontsize=18)
    ax1.tick_params(labelsize=15)
   
    kde = KernelDensity(kernel='gaussian', bandwidth=0.001).fit(np.array(df[colname].dropna()).reshape(-1,1)) 
    x = np.linspace(-0.1, 1.1, bins)[:,None]
    y = np.exp(kde.score_samples(x))

    ax2.plot(x, y, color='black')
    
    ax1.set_xlabel(xlabel, fontsize=18)
    ax2.set_ylabel('P', fontsize=18)
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(0)
    ax2.set_ylim(0)
    plt.show()
    
    return x, y




df.loc[df['TF'] == df['pred_class'], 'isCorrect'] = 1
df.loc[df['TF'] != df['pred_class'], 'isCorrect'] = 0


correct_df = df.query('isCorrect == 1')
correct_df = correct_df.reset_index(drop=True)

iscorrect_df = df.query('isCorrect == 0')
iscorrect_df = iscorrect_df.reset_index(drop=True)

x_correct, y_correct = plot_kde(correct_df, 'raw_t', 'tur', bins=1000, xmax=0.2)
x_iscorrect, y_iscorrect = plot_kde(iscorrect_df, 'raw_t', 'tur', bins=1000, xmax=0.2)



# TF = input('T or F, T = correct, F = iscorrect:')

# カーネル密度推定した分布から積分
def calc_integral(target, value, x1, y1, x2, y2):
    # 正規化するために最大値、最小値、平均値を取得
    max_ = df_max.loc[target]
    min_ = df_min.loc[target]
    mean_ = df_mean.loc[target]
    
    # 対象データを正規化
    zscore = (value - int(min_)) / (int(max_) - int(min_))
    # print(str(value) + '度のzscore', zscore)
    
    # 対象データが確率分布のどこか判断
    # x1_df = pd.DataFrame(x1)
    # x1_df.to_csv('./csv/' + season + '/x_' + TF + '.csv')
    # y1_df = pd.DataFrame(y1)
    # y1_df.to_csv('./csv/' + season + '/y_' + TF + '.csv')

    # 対象データが確率分布のどこか判断
    x1_df = pd.DataFrame(x1)
    x1_df.to_csv('./csv/' + season + '/x_T.csv')
    y1_df = pd.DataFrame(y1)
    y1_df.to_csv('./csv/' + season + '/y_T.csv')

    x2_df = pd.DataFrame(x2)
    x2_df.to_csv('./csv/' + season + '/x_F.csv')
    y2_df = pd.DataFrame(y2)
    y2_df.to_csv('./csv/' + season + '/y_F.csv')
    for index, data in x1_df.iterrows():
        if zscore <= float(data):
            # print(index)
            break
    x1 = x1[index - 5 : index + 5].reshape(-1)
    y1 = y1[index - 5 : index + 5].reshape(-1)

    for index, data in x2_df.iterrows():
        if zscore <= float(data):
            # print(index)
            break
    x2 = x2[index - 5 : index + 5].reshape(-1)
    y2 = y2[index - 5 : index + 5].reshape(-1)

    s1 = simps(y1, x1)
    s2 = simps(y2, x2)
    # print(s)
    # print(str(value) + '度の発生確率', round(s*10000) / 10000)
    return s1, s2



p1 = []
p2 = []


for index, data in tqdm(df_test.iterrows(), total=len(df_test)):
    s1, s2 = calc_integral('raw_t', data['raw_t'], x_correct, y_correct, x_iscorrect, y_iscorrect)
    p1.append(s1)
    p2.append(s2)

p1_df = pd.DataFrame(p1)
p1_df.columns = ['occur_true_p']
p1_df = p1_df.reset_index(drop=True)

p2_df = pd.DataFrame(p2)
p2_df.columns = ['occur_false_p']
p2_df = p2_df.reset_index(drop=True)

df_1 = pd.concat([df, p1_df, p2_df], axis=1)
df2 = df_1.dropna()

df2 = pd.merge(df_copy, df2, on='img')

df2 = df2.dropna()
df2 = df2.sort_values('img')
df2.to_csv('./csv/' + season + '/test_' + season + '.csv')





