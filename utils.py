import pandas as pd
# import git
import os
import shutil
import zipfile
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from scipy.integrate import simps
# import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns

import plot


def aECE(xs, ys):
    aece = 0.0
    r = len(xs)
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        aece += abs(x - y)
    aece /= r
    print("aECE: {:4f}".format(aece))
    return aece


def temp_scaling_softmax(input: np.array, t: float):
    '''
    input: それぞれのクラスの出力
    t: 温度
    '''
    exp_input = np.exp(input / t)
    exp_sum_input = np.sum(exp_input, axis=0)
    return exp_input / exp_sum_input


def integral(y_lst, x_lst, input_x):
    '''
    y_lst: カーネル密度関数のyの値
    x_lst: xの値(np.linspace(-2,8,1000))
    input_x: 積分したい値
    '''
    x_max = max(x_lst)
    x_min = min(x_lst)
    if input_x <= x_min:
        index_up = 11
        index_down = 0
    elif input_x >= x_max:
        index_up = 1199
        index_down = 1188
    else:
        start = input_x - (x_min)
        rate = start / (x_max - (x_min))
        index = rate * 1200
        index_up = int(index) + 5
        index_down = max(0, int(index) - 4)
        
    y_ = y_lst[index_down:index_up]
    x_ = x_lst[index_down:index_up]
    p = simps(y_, x_)
    return p


def bining_by_count(
    df: pd.DataFrame,
    bins,
    x_colname,
    y_colname='isCorrect'
):

    df = df.sort_values(x_colname)

    xs = []
    ys = []

    itr = 0
    while itr < len(df):
        filterd = df[itr: itr+bins]
        xs.append(filterd[x_colname].mean())
        ys.append(filterd[y_colname].mean())
        itr += bins

    return xs, ys


def bining_by_value(
    df: pd.DataFrame(),
    range,
    unit,
    x_colname,
    y_colname="isCorrect"
):
    border = range[0]
    xs = []
    ys = []
    while border <= range[1]:
        filterd = df[df[x_colname] > border][df[x_colname] <= border + unit]
        xs.append((border + border + unit) / 2)
        ys.append(filterd[y_colname].mean())
        border += unit
    return xs, ys


def get_x_y_lst(df: pd.DataFrame(), border_start, border_end, tanni):
    border = border_start
    x_lst = []
    y_lst = []
    while border <= border_end:
        tmp_df = df[df['p_after'] > border][df['p_after'] <= border + tanni]
        y_lst.append((border + border + tanni) / 2)
        x_lst.append(tmp_df['isCorrect'].mean())
        border += tanni
    return x_lst, y_lst


def baysian_update(p_before, df, target_colname, tanni=0.05, correct_colname="isCorrect"):
    correct_df = df.copy()[df[correct_colname] == 1]
    incorrect_df = df.copy()[df[correct_colname] == 0]

    p_after_lst = []
    for i in tqdm(range(len(df))):
        tmp_df = df.iloc[i, :]
        addjust_treated_t = tmp_df[target_colname]
        range_t = round(addjust_treated_t * 100) / 100
        range_t_up = range_t + tanni
        range_t_down = range_t - tanni

        correct_target_df = correct_df[correct_df[target_colname]
                                       > range_t_down]
        correct_target_df = correct_target_df[correct_target_df[target_colname] < range_t_up]

        m1 = len(correct_target_df) / len(correct_df)
        m2 = 1 - m1

        incorrect_target_df = incorrect_df[incorrect_df[target_colname] > range_t_down]
        incorrect_target_df = incorrect_target_df[incorrect_target_df[target_colname] < range_t_up]

        m3 = len(incorrect_target_df) / len(incorrect_df)
        m4 = 1 - m3

        p1 = m1 * p_before
        p2 = m2 * p_before
        p3 = m3 * (1-p_before)
        p4 = m4 * (1-p_before)

        p_after = p1 / (p1+p3)
        p_after_lst.append(p_after)

    df['p_after'] = p_after_lst
    return df


def linear_metrics(y_true, y_pred, plotshow=True):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print("mse: ", mse)
    print("rmse: ", rmse)
    print("r2: ", r2)
    if plotshow:
        plot.yyplot(y_true, y_pred)
    return mse, rmse, r2


def classification_metrics(y_true, y_pred):
    print("== マトリクス (縦軸:実際のクラス, 横軸:予測クラス) ==")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True)
    plt.show()

    print("\n== 正解率 ==")
    cnt = 0
    for y in range(cm.shape[0]):
        for x in range(cm.shape[1]):
            if y == x:
                cnt += cm[y][x]
    print("{}/{} = {:4f}".format(cnt, cm.sum(), cnt/cm.sum()))

    print("\n== 真のクラスの条件下 ==")
    for y in range(cm.shape[0]):
        print("[{}]: {}/{} = {:4f}".format(y, cm[y][y],
              cm[y].sum(), cm[y][y] / cm[y].sum()))

    print("\n== 予測クラスの条件下 ==")
    for x in range(cm.shape[1]):
        sum = 0
        for y in range(cm.shape[0]):
            sum += cm[y][x]
        print("[{}]: {}/{}  = {:4f}".format(x, cm[x][x], sum, cm[x][x] / sum))


def unzip_in_colab(zip_path: str, unzip_path: str):
    os.makedirs(unzip_path, exist_ok=True)
    data = zipfile.ZipFile(zip_path)
    data.namelist()
    data.extractall(path=unzip_path)


def show_class_frequency(df, class_num, colname='class'):
    for i in range(class_num+1):
        frequency = len(df[df[colname] == i])
        if frequency:
            print("class{}: {}".format(i, frequency))


def under_sampling_binary(df, colname='class'):
    class0 = df[df[colname] == 0]
    class1 = df[df[colname] == 1]
    num = min(len(class0[colname]), len(class1[colname]))
    class0 = class0.sample(num)
    class1 = class1.sample(num)
    ret = pd.concat([class0, class1])
    return ret


def gitclone(url):
    repo_name = url.split("/")[1].split(".")[0]
    try:
        git.Repo.clone_from(
            url,
            repo_name)
    except:
        git.GitCommandError


def split_df(df: pd.DataFrame(), n: int):
    '''
    DataFrameをn個に等分に切り分けてDataFrameの配列で返す関数
    行数がほぼn等分になるように切り分けるからDataFrameのサイズがでかくなりすぎたときに便利
    df: 等分したいDataFrame
    n: 何個のdfに分けるか
    '''
    index_lst = []
    start = 0
    long = len(df)
    tanni = int(float(long) / float(n)) + 1
    while start < long:
        if start+tanni > long:
            index_lst.append((start, long))
        else:
            index_lst.append((start, start+tanni))
        start += tanni
    df_lst = []
    for indexs in index_lst:
        tmp_df = df.copy().iloc[indexs[0]:indexs[1], :]
        df_lst.append(tmp_df)
    return df_lst


def sort_img_date(df: pd.DataFrame(), img_path_colname='img_path'):
    '''
    作成したデータフレームを日付順にソートする関数
    データセットにzipファイルを用いた時に使用する
    '''
    date_lst = []
    for i in range(len(df[img_path_colname])):
        value = df[img_path_colname].iloc[i]
        date_lst.append(int(value.split(".")[0]))
    df['date'] = date_lst
    df = df.sort_values('date')
    return df


def make_zip_from_df(df: pd.DataFrame(), img_dir: str, dest_path: str, img_path_colname="img_path"):
    '''
    作成したデータフレームに含まれるimg_pathのimage.jpgをまとめてzipファイルに圧縮する関数
    df: 使用するデータフレーム
    img_dir: 画像がたくさん入っているディレクトリ
    dest_path: zipファイルに圧縮するフォルダ名(.zipはつけない)
    '''
    os.makedirs(dest_path, exist_ok=True)
    for img_path in list(df[img_path_colname]):
        path = os.path.join(img_dir, img_path)
        shutil.copy2(path, dest_path)
    shutil.make_archive(dest_path, format="zip", root_dir=dest_path)
