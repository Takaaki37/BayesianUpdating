import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import os
import cv2
from IPython.display import display


def yyplot(
    y_act,
    y_pre,
    x_label="y_observed",
    y_label="y_predicted",
    labelsize=18,
    ticksize=15,
    set_range=False,
    set_max=0,
    set_min=0,
    s=1
):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(y_act, y_pre, marker='.', s=s)
    if set_range == True:
        max_ = set_max
        min_ = set_min
    else:
        max_ = max(y_pre.max(), y_act.max())
        min_ = min(y_pre.min(), y_act.min())
    line_x = np.arange(min_, max_+0.1, 0.1)
    line_y = line_x
    plt.plot(line_x, line_y, c="black")
    plt.xlim(min_, max_)
    plt.ylim(min_, max_)
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel(x_label, fontsize=labelsize)
    plt.ylabel(y_label, fontsize=labelsize)
    plt.tick_params(labelsize=ticksize)


def line(
    y_act,
    y_pre,
    x_label="y_observed",
    y_label="y_predicted",
    labelsize=18,
    ticksize=15,
    set_range=False,
    x_range=(0, 1),
    y_range=(0, 1),
    isFill=False,
    isSquare=False
):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if isFill == True:
        ax.fill_between(y_act, y_pre)
    plt.plot(y_act, y_pre, lw=0.4)
    if isSquare == True:
        line_x = np.arange(x_range[0], x_range[1]+0.1, 0.1)
        line_y = line_x
        plt.plot(line_x, line_y, c="black")
        ax.set_aspect('equal', adjustable='box')
        if set_range == False:
            x_max_ = max(y_pre.max(), y_act.max())
            y_max_ = max(y_pre.max(), y_act.max())
            x_min_ = min(y_pre.min(), y_act.min())
            y_min_ = min(y_pre.min(), y_act.min())
        else:
            x_max_ = max(x_range[1], y_range[1])
            y_max_ = max(x_range[1], y_range[1])
            x_min_ = min(x_range[0], y_range[0])
            y_min_ = min(x_range[0], y_range[0])
        plt.xlim(x_min_, x_max_)
        plt.ylim(y_min_, y_max_)
    else:
        if set_range == True:
            x_max_ = x_range[1]
            y_max_ = y_range[1]
            x_min_ = x_range[0]
            y_min_ = y_range[0]
            plt.xlim(x_min_, x_max_)
            plt.ylim(y_min_, y_max_)
    plt.xlabel(x_label, fontsize=labelsize)
    plt.ylabel(y_label, fontsize=labelsize)
    plt.tick_params(labelsize=ticksize)


def bar(
    x,
    y,
    x_label="",
    y_label="",
    labelsize=18,
    ticksize=15,
    set_range=False,
    set_max=0,
    set_min=0,
    width=1,
    isSquare=False
):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.bar(x, y, width=width)
    if isSquare == True:
        ax.set_aspect('equal', adjustable='box')
    if set_range == True:
        max_ = set_max
        min_ = set_min
    else:
        max_ = max(x.max(), y.max())
        min_ = min(x.min(), y.min())
    line_x = np.arange(min_, max_+0.1, 0.1)
    line_y = line_x
    plt.plot(line_x, line_y, c="black")
    plt.xlim(min_, max_)
    plt.ylim(min_, max_)
    plt.xlabel(x_label, fontsize=labelsize)
    plt.ylabel(y_label, fontsize=labelsize)
    plt.tick_params(labelsize=ticksize)


def plot(x_lst, y_lst, x_label, y_label, labelsize=18, ticksize=15, legend_label=""):
    plt.plot(x_lst, y_lst, label=legend_label)
    plt.xlabel(x_label, fontsize=labelsize)
    plt.ylabel(y_label, fontsize=labelsize)
    plt.tick_params(labelsize=ticksize)


def scatter(x_lst, y_lst, x_label, y_label, labelsize=18, ticksize=15, s=1, square=False, yyline=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if square == True:
        ax.set_aspect('equal', adjustable='box')
    if yyline == True:
        line_x = np.arange(0, 1+0.1, 0.1)
        line_y = line_x
        plt.plot(line_x, line_y, c="black")
    plt.scatter(x_lst, y_lst, s=s)
    plt.xlabel(x_label, fontsize=labelsize)
    plt.ylabel(y_label, fontsize=labelsize)
    plt.tick_params(labelsize=ticksize)
    # 日付用
    # plt.gcf().autofmt_xdate()


def plot_acc(epoch_lst, train_acc_lst, val_acc_lst):
    plt.plot(epoch_lst, train_acc_lst, label="train")
    plt.plot(epoch_lst, val_acc_lst, label="val")
    plt.xlabel("epoch", fontsize=18)
    plt.ylabel("accuracy", fontsize=18)
    plt.tick_params(labelsize=15)
    plt.legend()


def plot_loss(epoch_lst, train_loss_lst, val_loss_lst):
    plt.plot(epoch_lst, train_loss_lst, label="train")
    plt.plot(epoch_lst, val_loss_lst, label="val")
    plt.xlabel("epoch", fontsize=18)
    plt.ylabel("loss", fontsize=18)
    plt.tick_params(labelsize=15)
    plt.legend()


def hist(lst, x_label, labelsize=18, ticksize=15, bins=-1, alpha=1, legend_label=""):
    if bins == -1:
        plt.hist(lst, alpha=alpha, ec='black', label=legend_label)
    else:
        plt.hist(lst, bins=bins, alpha=alpha, ec='black', label=legend_label)
    plt.xlabel(x_label, fontsize=labelsize)
    plt.ylabel("Frequency", fontsize=labelsize)
    plt.tick_params(labelsize=ticksize)


def font_set(
    x_label,
    y_label,
    labelsize=18,
    ticksize=18,
    islegend=False
):
    if islegend:
        plt.legend(fontsize=labelsize)
    plt.xlabel(x_label, fontsize=labelsize)
    plt.ylabel(y_label, fontsize=labelsize)
    plt.tick_params(labelsize=ticksize)


def xreliabi_yacc(xs, ys):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    plt.plot(
        xs, ys, marker="D", markersize=1, linewidth=0.2, label="正解率"
    )
    line_x = np.arange(0, 1.1, 0.1)
    line_y = line_x
    plt.plot(line_x, line_y, c="black")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    font_set("信頼度", "正解率")


def roc(outcome_lst, uncertainty_lst, label):
    fpr, tpr, _ = roc_curve(outcome_lst, uncertainty_lst)
    plt.plot(fpr, tpr, label=label)
    plt.fill_between(fpr, tpr, 0, alpha=0.3)
    font_set("1-特異度", "感度")
    print(
        f'{label}: AUC: {roc_auc_score(outcome_lst, uncertainty_lst):.4f}'
    )


def xreliabi_yacchist(df, colname, bins):
    max_ = 1
    min_ = 0
    co_ = df.query('isCorrect == 1')
    in_ = df.query('isCorrect == 0')

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    cret = ax1.hist(
        co_[colname],
        alpha=0.5,
        ec='black',
        label="正解データ",
        bins=bins,
        range=(min_, max_)
    )

    iret = ax1.hist(
        in_[colname],
        alpha=0.5,
        ec='black',
        label="不正解データ",
        bins=bins,
        range=(min_, max_),
    )
    labelsize = 20
    ticksize = 18
    x_label = "信頼度"

    ax1.set_xlabel(x_label, fontsize=labelsize)
    ax1.set_ylabel("データ数 (度数)", fontsize=labelsize)
    ax1.tick_params(labelsize=ticksize)

    xs = []
    for i in range(len(cret[1])):
        if i == 0:
            continue
        xs.append((cret[1][i]+cret[1][i-1])/2)
    ys = cret[0] / (cret[0] + iret[0])
    ax2.plot(
        xs, ys,
        c='black', marker="D",
        markersize=3, linewidth=0.8,
        label="正解率"
    )
    ax2.set_ylabel("正解率", fontsize=labelsize)
    ax2.tick_params(labelsize=ticksize)
    ax2.set_ylim(0, 1)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(
        h1+h2, l1+l2, fontsize=labelsize, bbox_to_anchor=(0.4, 1.2), loc='upper center', borderaxespad=0, ncol=3
    )
    plt.show()


def x_yacchist(df, colname, bins, xrange, xlabel):
    max_ = xrange[1]
    min_ = xrange[0]
    co_ = df.query('isCorrect == 1')
    in_ = df.query('isCorrect == 0')

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    cret = ax1.hist(
        co_[colname],
        alpha=0.5,
        ec='black',
        label="正解データ",
        bins=bins,
        range=(min_, max_)
    )

    iret = ax1.hist(
        in_[colname],
        alpha=0.5,
        ec='black',
        label="不正解データ",
        bins=bins,
        range=(min_, max_),
    )
    labelsize = 20
    ticksize = 18

    ax1.set_xlabel(xlabel, fontsize=labelsize)
    ax1.set_ylabel("データ数 (度数)", fontsize=labelsize)
    ax1.tick_params(labelsize=ticksize)

    xs = []
    for i in range(len(cret[1])):
        if i == 0:
            continue
        xs.append((cret[1][i]+cret[1][i-1])/2)
    ys = cret[0] / (cret[0] + iret[0])
    ax2.plot(
        xs, ys,
        c='black', marker="D",
        markersize=3, linewidth=0.8,
        label="正解率"
    )
    ax2.set_ylabel("正解率", fontsize=labelsize)
    ax2.tick_params(labelsize=ticksize)
    ax2.set_ylim(0, 1)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(
        h1+h2, l1+l2, fontsize=labelsize, bbox_to_anchor=(0.4, 1.2), loc='upper center', borderaxespad=0, ncol=3
    )
    plt.show()


def img(basename, dir_path):
    path = os.path.join(dir_path, basename)
    im = cv2.imread(path)
    plt.imshow(im)
    plt.title(basename)
    plt.tick_params(labelbottom=False, labelleft=False,
                    labelright=False, labeltop=False)
    return im


def plot_six_images(df, dir_path):
    _, ax = plt.subplots(nrows=1, ncols=6, figsize=(20, 20))
    for itr, row in enumerate(df.itertuples()):
        _c = itr % 6

        path = os.path.join(dir_path, row.img_path)
        img = cv2.imread(path)
        title = "[{}]".format(row.img_path)
        ax[_c].set_title(title)
        ax[_c].axes.xaxis.set_visible(False)
        ax[_c].axes.yaxis.set_visible(False)
        ax[_c].tick_params(
            labelbottom=False, labelleft=False,
            labelright=False, labeltop=False
        )
        ax[_c].imshow(img, cmap='Greys')
    plt.show()


def plt_random_imgs(df, dir_path):
    '''
    dfからランダムに10行抽出して表示する
    '''
    row = 1
    col = 6

    random_filterd = df.sample(min(len(df), 6))
    plot_six_images(random_filterd, dir_path)
    return random_filterd
