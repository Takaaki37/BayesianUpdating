from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import utils, plot


def get_max_min(df, colname):
    # 最小値と最大値の取得
    max_ = df[colname].max()
    min_ = df[colname].min()
    print(colname)
    print('max', max_)
    print('min', min_)
    return max_, min_


# def normalize_row(dfco, dfin, colname, max_, min_):
#     # 列の正規化
#     co_ = dfco.copy()
#     in_ = dfin.copy()
#     co_["normal_"+colname] = (co_[colname]-min_)/(max_-min_)
#     in_["normal_"+colname] = (in_[colname]-min_)/(max_-min_)
#     return co_, in_

def normalize_row(dfco, dfin, df, colname, max_, min_):
    # 列の正規化
    co_ = dfco.copy()
    in_ = dfin.copy()
    co_["normal_"+colname] = (co_[colname]-min_)/(max_-min_)
    in_["normal_"+colname] = (in_[colname]-min_)/(max_-min_)
    df["normal_"+colname] = (df[colname]-min_)/(max_-min_)
    
    print('正規化0.01の値：', 0.01 * (max_-min_) + min_)
    
    return co_, in_, df


# def plot_hist(dfco, dfin, colname, bins=50):
#     # ヒストグラムの可視化
#     plot.hist(
#         dfco["normal_"+colname],
#         "normalized uncertainty",
#         bins=bins,
#         alpha=0.5,
#         legend_label="correct"
#     )
#     plot.hist(
#         dfin["normal_"+colname],
#         "normalized uncertainty",
#         bins=bins,
#         alpha=0.5,
#         legend_label="error"
#     )
#     plt.legend(fontsize=15)
#     plt.xlim(-0.1, 1.1)
#     plt.show()


def plot_hist(dfco, dfin, colname, bins=50):
    # ヒストグラムの可視化
    plot.hist(
        dfco["normal_"+colname],
        "normalized uncertainty",
        bins=bins,
        # alpha=0.5,
        legend_label="correct",
        # range=(0, 1)
    )
    plot.hist(
        dfin["normal_"+colname],
        "normalized uncertainty",
        bins=bins,
        # alpha=0.5,
        legend_label="error",
        # range=(0, 1)
    )
    plt.legend(fontsize=15)
    plt.xlim(-0.1, 1.1)
    plt.subplots_adjust(left=0.2, right=0.85, bottom=0.15, top=0.95)
    plt.savefig('img2/' + colname + '_hist.jpg', dpi=300)
    plt.show()


def plot_kde_hist(dfco, dfin, df, colname, length, xlabel, bin=50, upper_num=30000):
    # ヒストグラムの可視化
    
    plt.rcParams["font.size"] = 26
    plt.figure(figsize=(10, 4))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_ylim(0, upper_num)
    ax2.set_ylim(0, 100)
    
    # ax1.hist(
    #     dfco["normal_"+colname],
    #     label="正解データ",
    #     bins=bin,
    #     # alpha=0.5,
    #     # legend_label="correct",
    #     # color='#1f77b4',
    #     histtype='bar',
    #     edgecolor='black',
    # )
    
    # ax1.hist(
    #     dfin["normal_"+colname],
    #     label="不正解データ",
    #     bins=bin,
    #     # alpha=0.5,
    #     # legend_label="error",
    #     # color='#ff7f0e',
    #     histtype='bar',
    #     edgecolor='black',
    # )
    
    ax1.hist(
        df["normal_"+colname],
        label="不正解データ",
        bins=bin,
        # alpha=0.5,
        # legend_label="error",
        color='gray',
        histtype='bar',
        edgecolor='black',
    )

    ax1.set_ylabel("データ数(度数)", fontsize=18)
    ax1.tick_params(labelsize=15)
    # ax1.legend()
    
    
    # カーネル密度推定の可視化
    # kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(
    #     np.array(df["normal_"+colname][:, None])
    # )
    # x = np.linspace(-0.1, 1.1, 1200)
    # y_lst = np.exp(kde.score_samples(x[:, None]))
    # ax2.plot(x, y_lst, color='black', lw=2, label='全データの確率分布')
   
    # kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(
    #     np.array(dfco["normal_"+colname][:, None])
    # )
    # x = np.linspace(-0.1, 1.1, 1200)
    # y_lst = np.exp(kde.score_samples(x[:, None]))
    # ax2.plot(x, y_lst, color='#1f77b4', lw=2, label='正解データの確率分布')
    
    # kde = KernelDen２in["normal_"+colname][:, None])
    # )
    # x = np.linspace(-0.1, 1.1, 1200)
    # y_lst = np.exp(kde.score_samples(x[:, None]))
    # ax2.plot(x, y_lst, color='#ff7f0e', lw=2, label='不正解データの確率分布')


    Y = []
    Z = []
    length = length
    length2 = length / 2
    
    for i in tqdm(range(1001)):
        i = i * length
        x = df.loc[(df["normal_"+colname] >= i - length2) & (df["normal_"+colname] < i + length2)]
        Y.append(x['TF'].mean())
        Z.append(i)

    Y = pd.DataFrame(Y)* 100
    Y.columns = ['正答率']
    Z = pd.DataFrame(Z)
    Z.columns = [colname]
    dd = pd.concat([Z, Y], axis=1)
    # dd.loc[22, :]['正答率'] = 87.301082-4
    # dd.loc[23, :]['正答率'] = 90.805417-5
    # dd.loc[24, :]['正答率'] = 87.301082-6
    # dd.loc[25, :]['正答率'] = 100.000000-13

    ax2.plot(dd[colname], 
             dd['正答率'], 
             color='black', 
            #  marker="D", 
            #  markersize=3, 
             linewidth=2, 
             label="正答率")
    
    ax1.set_xlabel(xlabel, fontsize=18)
    ax2.set_ylabel("正答率", fontsize=18)
    ax2.tick_params(labelsize=15)
    ax1.set_xlim(-0.1, 1.1)
    # ax1.set_ylim(0)
    ax2.set_ylim(0)
    plt.subplots_adjust(left=0.2, right=0.9, bottom=0.15, top=0.85)
    plt.savefig('img2/' + colname + '.jpg', dpi=300)
    plt.show()
    return dd



def plot_kde(df, colname, bandwidth, xlabel, bins=50, TF=1):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.hist(
        df["normal_"+colname],
        label="normalized uncertainty",
        bins=bins,
        alpha=0.5,
        ec='black',
        range=(0, 1)
    )
    ax1.set_ylabel("データ数(度数)", fontsize=18)
    ax1.tick_params(labelsize=15)
    # カーネル密度推定の可視化
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(
        np.array(df["normal_"+colname][:, None])
    )
    x = np.linspace(-0.1, 1.1, 1200)
    y_lst = np.exp(kde.score_samples(x[:, None]))
    ax2.plot(x, y_lst)
    ax1.set_xlabel(xlabel, fontsize=18)
    ax2.set_ylabel("P", fontsize=18)
    ax2.tick_params(labelsize=15)
    ax1.set_xlim(-0.1, 1.1)
    plt.subplots_adjust(left=0.2, right=0.85, bottom=0.15, top=0.95)
    plt.savefig('img2/' + colname + '_' + str(TF) + '.jpg', dpi=300)
    plt.show()
    return x, y_lst


def integral_df(df, colname, max_, min_, x_lst, co_y_lst, in_y_lst):
    # 積分した列の作成
    index = list(df.columns.values).index(colname) + 1
    m1_lst = []
    m3_lst = []
    for row in tqdm(df.itertuples(), total=len(df)):
        value = (row[index] - min_) / (max_-min_)
        m1 = utils.integral(co_y_lst, x_lst, value)
        m3 = utils.integral(in_y_lst, x_lst, value)
        m1_lst.append(m1)
        m3_lst.append(m3)
    ret = df.copy()
    ret["normal_"+colname+"_m1"] = m1_lst
    ret["normal_"+colname+"_m3"] = m3_lst
    return ret


def update_baysian(p_before, m1_colname, m3_colname, new_colname, df):
    # ベイズ更新
    p1_before = 1 - p_before
    ret = df.copy()
    ret[new_colname] = (df[m1_colname]*p_before) / \
        (df[m1_colname]*p_before+df[m3_colname]*p1_before)
    return ret


def plot_prep_p(df, colname, tanni=100):
    # プロットで可視化
    sort_df = df.sort_values(colname)
    x_lst = []
    y_lst = []
    lines = 0

    sort_df = sort_df[sort_df[colname] >= 0.0]
    while lines < len(sort_df):
        target_df = sort_df.iloc[lines:lines+tanni]
        x = target_df[colname].mean()
        y = target_df['isCorrect'].mean()
        lines += tanni
        x_lst.append(x)
        y_lst.append(y)

    plot.line(
        np.array(x_lst),
        np.array(y_lst),
        x_label="predicted accuracy",
        y_label="accuracy",
        set_range=True,
        x_range=(0, 1),
        y_range=(0, 1),
        isFill=False,
        isSquare=True
    )
    _, _, _ = utils.linear_metrics(x_lst, y_lst, False)


def visu(df2, colname, bins=100):
    xs, ys = utils.bining_by_count(df2, 100, colname)
    _ = utils.aECE(xs, ys)

    plot.xreliabi_yacc(xs, ys)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    plot.roc(df2['isCorrect'], df2[colname], colname)
    plt.show()

    plot.xreliabi_yacchist(df2, colname, bins)
