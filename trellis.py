"""
trellis: a python module for batch automation of charts and reports

Copyright (C) 2017  Allan Campopiano

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sqlite3
from subprocess import call
import re
import csv
import glob
from IPython.display import Markdown
from scipy import stats


def saveSwapFile(destPath, destFname, srcPath, srcFname, col_name, show=True):

# load csv into df, reduce to one column, find unique values, save to csv
# csv is used for string swapping in jupyter notebooks.

    df = pd.read_csv(srcPath + srcFname)
    str_list = list(pd.Series.unique(df[col_name]))

    with open(destPath+destFname, 'w') as myfile:
        wr = csv.writer(myfile, lineterminator='\n')
        for val in str_list:
            wr.writerow([val])

    if show:
        print(str_list)

def nb2pdf(proj_path, nb):

    # 1 export notebook as md
    # 2 remove py code from mark down
    # 3 pandocs to convert md to latex
    # 4 remove captions from latex
    # 5 pandocs to convert latex to pdf

   # get into proper directory
   os.chdir(proj_path)

   # saves notebook with same name in cwd
   call('jupyter nbconvert --to markdown ' + nb)

   # little script for finding and replacing and reading/writing with re

   # open md file
   f = open(nb[0:-6] + '.md', encoding='utf-8')
   text = f.read()
   f.close()

   clean = re.findall('```python(.*?)```', text, re.DOTALL)

   for chunk in clean:
      text=text.replace('```python' + chunk + '```' ," ")

   f = open(nb[0:-6] + '.md', 'w', encoding='utf-8')
   f.write(text)
   f.close()

   # remove captions for figures

   os.system('pandoc -f markdown -t latex ' + nb[0:-6] + '.md -o ' + nb[0:-6] + '.tex')

   # open md file
   f = open(nb[0:-6] + '.tex')
   text = f.read()
   f.close()

   # remove captions and choose PDFs except for header
   text=text.replace('\caption{png}', ' ')
   text=text.replace('png', 'pdf')
   text = text.replace('pdf', 'png', 1)

   f = open(nb[0:-6] + '.tex', 'w')
   f.write(text)
   f.close()

   # pandocs to convert latex to pdf
   call('pandoc ' + nb[0:-6] + '.tex -o ' + nb[0:-6] + '.pdf')

def batchConvert(toStrings_fname, nb_fname, fpath, append_ind, toPDF=False, toHTML=False):

    """
    Search batch compatible notebook for {{}}=type strings, when found, replace them with strings from a dataframe (CSV). Column headers in df are given without {{}},
    and each row corresponds to a new notebook once batching is complete. Append ind dictates which column header is used to distinguish the notebooks (thus, primary key)

    :param toStrings_fname: csv file containing the mapping of find and replace pairs. Column header is the string to find in batch file, rows are the replaced string
    :param nb_fname: name of parent nb
    :param fpath: path to save batch files and where parent batch file exists
    :param append_ind: numerical index of the column in main df (toStrings_fname.csv) to use for getting strings to append to output file names
    :param toPDF: True or False
    :param toHTML: True or False
    :return:
    """

    # multiple columns can have different str swaps in them
    #if type(fromStrings) is not list:
    #    fromStrings=[fromStrings] # make a list if one string is given

    os.chdir(fpath)
    df=pd.read_csv(toStrings_fname)

    # get column headers
    ks=list(df.keys())

    # build {{}}-type strings from the column headers in toStrings_fname.csv
    fromStrings=[]
    for k in ks:
        fromStrings.append('{{' + k + '}}')

    for i in range(len(df)):

        # open batch notebook
        f = open(nb_fname)
        text = f.read()
        f.close()

        for ix, s in enumerate(fromStrings):

            # do the find and replace
            toStr=df[ks[ix]][i]

            # remove captions
            text = text.replace(s, toStr)

        # formating file names
        newStr=df[ks[append_ind]][i]
        newStr = newStr.strip()
        #newStr = toStr.strip()
        newStr = newStr.replace(" ", "_")
        newStr = newStr.replace(".", "_")
        newStr = newStr.replace("__", "_")

        f = open(nb_fname[0:-6] + '_' + newStr + '.ipynb', 'w')
        f.write(text)
        f.close()

        os.system('jupyter nbconvert --to notebook --execute ' + nb_fname[0:-6] + '_' + newStr + '.ipynb --output ' + nb_fname[0:-6] + '_' + newStr + '.ipynb')

        if toPDF:
            nb2pdf(fpath, nb_fname[0:-6] + '_' + newStr + '.ipynb')

        if toHTML:
            os.system('jupyter nbconvert --to html ' + nb_fname[0:-6] + '_' + newStr + '.ipynb')

def fillDatabase(fname_path, db_path, db_name, if_exists='replace'):

    """
    Add table(s) to database
    Can add to existing database or spin up brand new one

    :param fname_path: cleaned files ready for db eg: '/home/allan/research/databases/EQAO/cleaned/'
    :param db_path: eg: '/home/allan/research/databases/EQAO/'
    :param db_name: eg: mydatabase.db
    :param if_exists: 'replace', 'fail', replace table if exists, or fail to replace if exists
    """

    # globing fnames
    os.chdir(fname_path)
    fnames=glob.glob('*.csv')

    # preparing to place the db
    os.chdir(db_path)
    con=sqlite3.connect(db_name)

    # add table
    if if_exists=='replace':

        for f in fnames:
            df = pd.read_csv(fname_path+f)
            df.to_sql(os.path.splitext(f)[0] , con, if_exists=if_exists, index=False)

    # add table
    elif if_exists == 'fail':

        for f in fnames:
            try:
                df = pd.read_csv(fname_path + f)
                df.to_sql(os.path.splitext(f)[0], con, if_exists=if_exists, index=False)
            except Exception as e:
                print('ERROR', e)

def runQuery(query, connect, axis=0, show=False, join_on=False):

    """

    :param query:
    :param connect:
    :param axis:
    :param show:
    :param join_on: join variable (variable to do join on) for outer join, or False for straight concat
    :return:
    """

    if type(query) is list:
        df=None

        for sql in query:

            if df is None:
                df = pd.read_sql_query(sql, connect)
            else:
                tmp = pd.read_sql_query(sql, connect)

                if join_on is False:
                    df = pd.concat([df, tmp], axis=axis, ignore_index=False)

                else:

                    # testing
                    # df1=pd.DataFrame({'scores': [82,71,89,91,93,88], 'schools': ['Mother Teresa S','Pope John Paul II', 'St Bernadette Sep S', 'St Joan of Arc Catholic ES', 'St Matthews S', 'St Mary Elementary School'],})
                    # df2=pd.DataFrame({'scores': [86,80,89,88,95,73, 88], 'schools': ['Mother Teresa S','Pope John Paul II', 'St Bernadette Sep S', 'St Joan of Arc Catholic ES', 'St Matthews S', 'St Gregory the Great Elementary Schoo', 'St Mary Elementary School'],})
                    # df3=pd.merge(df1, df2, on='schools', how='outer')
                    df= pd.merge(df, tmp, on=join_on, how='outer')
                    df=df.sort_values(by=join_on, ascending=1)


    else:

        # sorting if index is being used (through 'join on' variable
        if join_on is False:
            df = pd.read_sql_query(query, connect)
        else:
            df = pd.read_sql_query(query, connect)
            df = df.sort_values(by=join_on, ascending=1)

    if join_on:
        # del df[join_on]
        df.set_index([join_on], inplace=True)

    if show:
        print(df.to_string(index=True))


    return df

def makeFig(df, conf, fig=None, defaults=None, xtext=False, legtext=False, pos=111):

    # merge with possible python dict
    if defaults:
        conf = {**defaults, **conf}

    y_text_obj = []

    if not fig:
        fig=plt.figure(figsize=conf['figsize'])

    if conf['zeronan']:
        df=df.fillna(0, inplace=False)

    # control rounding
    if conf['round_data'] is not False:
        df=roundFrame(conf,df)

    # figure options
    if conf['chart'] == 'bar':
        plotBar(df, conf, fig, xtext=xtext)

    if conf['chart'] == 'line':
        y_text_obj=plotLine(df, conf, fig, pos=pos, xtext=xtext)

    if conf['chart'] == 'scatter':
        plotScatter(df, conf, fig, pos=pos)

    if conf['chart'] == 'clust_bar':
        plotClustBar(df, conf, fig, pos=pos, legtext=legtext)
        #fig = plotClustBar(df, conf, fig, pos=pos, legtext=legtext)

    if conf['chart'] == 'stacked_bar':
        y_text_obj=plotStackedBar(df, conf, fig, pos=pos, xtext=xtext)

    # return fig object
    return y_text_obj

def makeSubFig(df_list, dict_list, pos_list, defaults=None, xtext_list=False, legtext_list=False):


    """
    give lists of dataframes, yaml configs, subplot positions, etc...
    wraps around makeFig to produce subplots

    :param df_list:
    :param dict_list:
    :param pos_list:
    :param defaults:
    :param xtext_list:
    :param legtext_list:
    :param legtext_list:
    :return:
    """

    #print(yam_list[0])
    conf = dict_list[0]

    if defaults:
        conf = {**defaults, **conf}

    # build canvas and add super annotations (xlabel, etc)
    fig = plt.figure(figsize=conf['par_figsize'])
    ax=fig.add_subplot(111,frameon=False)

    # setting title and optional positions
    m=1000 # here so that title padding behaves similar to axis label padding
    fig.suptitle(conf['par_title'], fontsize=conf['par_title_fontsize'], y=.98+(conf['par_title_pad']/m))
    plt.xlabel(conf['par_xlabel'], fontsize=conf['par_xlabel_fontsize'], labelpad=conf['par_xlabel_pad'])
    plt.ylabel(conf['par_ylabel'], fontsize=conf['par_ylabel_fontsize'], labelpad=conf['par_ylabel_pad'])
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

    res=[]

    for df, conf_dict, pos in zip(df_list, dict_list, pos_list):
        res=makeFig(df, conf_dict, fig=fig, defaults=defaults, pos=pos, xtext=xtext_list, legtext=legtext_list)

    # add a super legend to the canvas
    legpatch=[]
    if conf['par_legend']:

        for labs, cols in zip(conf['par_legend'], conf['par_legend_color']):
            legpatch.append(mpatches.Patch(label=labs, color=cols))

        ax.legend(handles=legpatch,loc=(1.04, .736), fontsize=conf['par_legend_fontsize'])
        # plt.legend(patches_tuple, labels_tuple, loc=(1.04, .736), fontsize=parconf['legend_fontsize'])


    return res

def readConfig(conf, defaults=None):

    #conf = yaml.load(yam)

    if defaults:
        #opts = yaml.load(defaults)

        #ks = list(conf.keys())  # get new set of keys

        # merge defaults with user specified options
        # nconf={}
        conf = {**defaults, **conf}

        #if 'axis' not in ks:
        #    conf['axis'] = opts['axis']

    else:
        defaults = dict(

            axis=1,
            ylabel='Percentage of Students',
            xlabel='Year',
            xticks='',
            ytick_fontsize=15,
            xtick_fontsize=18,
            title='',
            color=[[.97, .59, .11], [0, .49, .76], [.55, .77, .24], [.7, .7, .7], 'purple', 'green', 'black'],
            label_fontsize=20,
            title_fontsize=20,
            width=.3,
            text_color='black',
            text_fontsize=20,
            bench_data=False,
            bench_text=False,
            bench_color='black',
            bench_fontsize=12,
            bench_vert_align='top',
            ytext=6,
            xsep=['n=', ''],
            legend_fontsize=20,
            legend=False,
            legendpos='outside_top_right',
            linestyle='-',
            marker_style='-o',
            text_vert_align='bottom',
            ylim=[0, 100],
            zeronan=False,
            text_destroy=False,
            text_adjust=10,
            round_data=0,
            ytext_suffix='%',
            auto_comment=False,
            comment_categories=False,
            marker_size=6,
            linewidth=2,
            grid='y',
            rotate_xticks=45,
            figsize=[16, 6],
            round_text=0,
            par_title='',
            par_figsize=[16, 12],
            par_title_fontsize=20,
            par_xlabel_fontsize=20,
            par_ylabel_fontsize=20,
            par_title_pad=40,
            par_xlabel_pad=50,
            par_ylabel_pad=40,
            par_xlabel='Year',
            par_ylabel='Percentage of Students',
            par_legend=['School', 'Board'],
            par_legend_color=[[.97, .59, .11], [0, .49, .76], 'darkred'],
            par_legend_fontsize=20,
            kill_label=False,
            bold_y_gridline=False,
            regline_color='red',
            ols=False,
            ols_text=True
        )

        # give configuration some standard defaults
        #opts = yaml.load(defaults)

        ks = list(conf.keys())  # get new set of keys

        # merge defaults with user specified options
        # nconf={}
        conf = {**defaults, **conf}

        #if 'axis' not in ks:
        #    conf['axis'] = defaults['axis']

    return conf

def plotBar(df, conf, fig, xtext=False, pos=111):

    # shape of dataframe
    nrow, ncol = df.shape

    # get y
    if nrow == 1:

        y=df.values.tolist()[0]

    # allow df to be n rows and 1 column
    elif nrow > 1:

        # rearrange row oriented df into a list
        y = list(np.array(df)[:, 0])
        #y=df.values.tolist()[0]

    x = range(len(y))

    # figure canvas
    # figure canvas
    ax = fig.add_subplot(pos)  # give access to axis properties
    ax.set_axisbelow(True)
    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1)  # give access to axis properties
    #ax.set_axisbelow(True)

    # figure options
    width = conf['width']
    plt.grid(axis=conf['grid'])
    plt.xlabel(conf['xlabel'], fontsize=conf['label_fontsize'])
    plt.ylabel(conf['ylabel'], fontsize=conf['label_fontsize'])
    plt.title(conf['title'], fontsize=conf['title_fontsize'])

    if conf['xsep']:
        xticks=addXText(conf, xtext)
        plt.xticks(x, xticks, fontsize=conf['xtick_fontsize'], rotation=conf['rotate_xticks'])
    else:
        plt.xticks(x, conf['xticks'], fontsize=conf['xtick_fontsize'], rotation=conf['rotate_xticks'])


    plt.bar(x, y, width, color=conf['color'])

    # add text to top of bars
    # ynums = [round(elem, 3) for elem in y] # rounded text on bars
    # ystrs = [str(elem) for elem in ynums] # convereted to list of strs

    # loop through points and add text
    if conf['ytext']:

        ytext(x, y, conf)
        # ytext(x, y, ystrs, conf)

    # add benchmark if specified
    if conf['bench_data']:
        benchMark(conf, x, y)

    #return fig

def plotClustBar(df, conf, fig, legtext=False, pos=111):

    y = df.values  # np array

    # rotate if axis is 0
    # meaning you want each line to correspond to a column (across rows)
    if conf['axis'] == 0:
        y = np.rot90(y, k=1)
        y = np.flipud(y)

    nrow, ncol = y.shape
    x = range(ncol)

    # legend
    if not conf['legend']:
        leg_labs=['']*nrow

    else:
        leg_labs = conf['legend']

    # add text stuff to legends if specified (normally added to enhance xtick lables, but
    # in the cluster bar case, the text is added to the legend
    if conf['xsep'] and conf['legend']:
        leg_labs=addLegText(conf, legtext)
        #print(leg_labs)


    # figure canvas
    ax = fig.add_subplot(pos)
    ax.set_axisbelow(True)

    # figure options
    plt.grid(axis=conf['grid'])
    plt.xlabel(conf['xlabel'], fontsize=conf['label_fontsize'])
    plt.ylabel(conf['ylabel'], fontsize=conf['label_fontsize'])
    plt.title(conf['title'], fontsize=conf['title_fontsize'])

    if conf['ylim']:
        plt.ylim(conf['ylim'][0], conf['ylim'][1])

    # handle single color strings
    if type(conf['color']) is str:
        conf['color'] = [conf['color']] * nrow

    # empty list in case y text objects are returned
    width=conf['width']
    space=width
    for ind in range(nrow):

        if  ind==0:
            plt.bar(x, y[ind], conf['width'], color=conf['color'][ind], label=leg_labs[ind])
            xtext=x
        else:
            plt.bar(np.array(x)+space, y[ind], conf['width'], color=conf['color'][ind], label=leg_labs[ind])
            xtext = np.array(x) + space
            space=space+width

        # loop through point on current line and add text to them
        if conf['ytext']:

            # get numbers and strings for y
            # ynums = [round(elem, 3) for elem in y[ind]]  # rounded text on bars
            # ystrs = [str(elem) for elem in ynums]  # convereted to list of strs
            # ytext(xtext, y[ind], ystrs, conf)
            ytext(xtext, y[ind], conf)

        # if a legend and a position were specified
        if conf['legend'] and conf['legendpos']:

            if conf['legendpos'] == 'outside_top_right':
                plt.legend(loc=(1.04, .736), fontsize=14)

            if conf['legendpos'] == 'outside_bottom_right':
                plt.legend(loc=(1.04, 0), fontsize=14)

        # if a legend but no position was specified
        elif conf['legend']:
            plt.legend(fontsize=14)

    # add ticks position so its kind of sensible
    # probably works for clusters <4, but who wants more than that anyway...
    if nrow>2:
        plt.xticks(np.array(x)+width, conf['xticks'], fontsize=conf['xtick_fontsize'], rotation=conf['rotate_xticks'])
    else:
        plt.xticks(x, conf['xticks'], fontsize=conf['xtick_fontsize'], rotation=conf['rotate_xticks'])

    #print('help')
    #print(fig)
    #return fig

def plotLine(df, conf, fig, xtext=False, pos=111):

    y = df.values # np array

    # rotate if axis is 0
    # meaning you want each line to correspond to a column (across rows)
    if conf['axis']==0:

        y = np.rot90(y, k=1)
        y = np.flipud(y)

    nrow, ncol = y.shape
    x = range(ncol)

    # legend
    if not conf['legend']:
        leg_labs=['']*nrow

    else:
        leg_labs = conf['legend']

    # figure canvas
    ax = fig.add_subplot(pos)  # give access to axis properties
    ax.set_axisbelow(True)
    ax.yaxis.set_tick_params(labelsize=conf['ytick_fontsize'])
    # ax.set_autoscale_on(True)

    # figure options
    #plt.grid(axis='y')
    plt.grid(axis=conf['grid'])

    if conf['bold_y_gridline'] is not False:

        # get grid line objects
        #grid_h = ax.get_ygridlines()
        # grid_ln=grid_h[conf['bold_y_gridline']]
        #grid_ln.set_linewidth(3)
        grid_ln = conf['bold_y_gridline']
        plt.axhline(y=grid_ln, linewidth=3, color=[.7, .7, .7])


    plt.xlabel(conf['xlabel'], fontsize=conf['label_fontsize'])
    plt.ylabel(conf['ylabel'], fontsize=conf['label_fontsize'])
    plt.title(conf['title'], fontsize=conf['title_fontsize'])
    plt.xlim([min(x)-.5,max(x)+.5]) # This forces the full xaxis, otherwise it gets scalled to min and max (so lead/trail nans get chopped off, changing the xaxis range)


    if conf['xsep']:
        xticks=addXText(conf, xtext)
        plt.xticks(x, xticks, fontsize=conf['xtick_fontsize'], rotation=conf['rotate_xticks'])
    else:
        plt.xticks(x, conf['xticks'], fontsize=conf['xtick_fontsize'], rotation=conf['rotate_xticks'])

    if conf['ylim']:
        plt.ylim(conf['ylim'][0], conf['ylim'][1])

    if type(conf['linestyle']) is str:
            conf['linestyle'] = [conf['linestyle']] * nrow

    if type(conf['linewidth']) is not list:
            conf['linewidth'] = [conf['linewidth']] * nrow

    if type(conf['marker_size']) is not list:
            conf['marker_size'] = [conf['marker_size']] * nrow

    # handle single color strings
    if type(conf['color']) is str:
        conf['color']=[conf['color']]*nrow

    # empty list in case y text objects are returned
    y_text_obj=[]
    for ind in range(nrow):
        plt.plot(x, y[ind], conf['marker_style'], linestyle=conf['linestyle'][ind],
                 color=conf['color'][ind], lw=conf['linewidth'][ind], ms=conf['marker_size'][ind], label=leg_labs[ind])

        # loop through point on current line and add text to them
        if conf['ytext'] is not False:

            # get numbers and strings for y
            # ynums = [round(elem, 3) for elem in y[ind]]  # rounded text on bars
            # ystrs = [str(elem) for elem in ynums]  # convereted to list of strs
            # text_obj=ytext(x, y[ind], ystrs, conf)
            text_obj = ytext(x, y[ind], conf)
            y_text_obj.append(text_obj)

        # if a legend and a position were specified
        if conf['legend'] and conf['legendpos']:

            if conf['legendpos']=='outside_top_right':

                plt.legend(loc=(1.04,.736), fontsize=conf['legend_fontsize'])

            if conf['legendpos']=='outside_bottom_right':

                plt.legend(loc=(1.04, 0), fontsize=conf['legend_fontsize'])


        # if a legend but no position was specified
        elif conf['legend']:

            plt.legend(fontsize=conf['legend_fontsize'])

    # adjust too close text labels
    if conf['text_adjust'] and conf['ytext']:
            textLabelAdjust(conf, y_text_obj, y)

    # destroy too close text labels
    if conf['text_destroy'] and conf['ytext']:
        textLabelDestroy(conf, y_text_obj, y)


    # text obj can be returned if maunal control over text label is needed
    return y_text_obj

def plotScatter(df, conf, fig, pos=111):

    # ATM this assumes that columns are the xy data and rows are observations

    # control text formating based on rounding option
    round_opt=str(conf['round_text'])
    if not round_opt == 'False':
        round_str="{0:." + round_opt + "f}"
    else:
        round_str="{0:.4f}"

    # handle single color strings
    if type(conf['color']) is str:
        conf['color']=[conf['color']]

    if type(conf['color']) is str:
        conf['color']=[conf['color']]

    xy=df.values # np array
    xy.astype(np.float) # convert from strings to numbers
    x=list(xy[:,0])
    y=list(xy[:,1])


    if type(pos) is list:
        ax = fig.add_subplot(pos[0],pos[1],pos[2])  # give access to axis properties
    else:
        ax = fig.add_subplot(pos)  # give access to axis properties

    ax.set_axisbelow(True)
    ax.yaxis.set_tick_params(labelsize=conf['ytick_fontsize'])

    plt.grid(axis=conf['grid'])

    # figure options
    plt.xlabel(conf['xlabel'], fontsize=conf['label_fontsize'])
    plt.ylabel(conf['ylabel'], fontsize=conf['label_fontsize'])
    #print(conf['title'])
    plt.title(conf['title'], fontsize=conf['title_fontsize'])

    plt.scatter(x ,y, s=conf['marker_size'], c=conf['color'], alpha=.5)

    # basic regression stats
    if conf['ols'] is True:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x ,y)

        slp = intercept + slope * np.array(x)
        plt.plot([min(x),max(x)], [min(slp), max(slp)], color=conf['regline_color'], lw=conf['linewidth'])
        # plt.plot(x,slp, color=conf['regline_color'], lw=conf['linewidth'])

        if conf['ols_text'] is True:

            str_to_plot='r2 = ' + round_str.format(r_value**2) + '\np value = ' + round_str.format(p_value)

            if p_value<=.01:
                cl='red'
            else:
                cl=conf['text_color']

            plt.text(min(x), max(y), str_to_plot, color=cl,
                          verticalalignment=conf['text_vert_align'], horizontalalignment='left',
                          size=conf['text_fontsize'], style='italic')

    #return fig

def plotStackedBar(df, conf, fig, xtext=False, pos=111):

    y = df.values  # np array

    # rotate if axis is 0
    # meaning you want each line to correspond to a column (across rows)
    if conf['axis'] == 0:
        print('axis must be 1 (within each bar, segments correspond to elements in each column of df')
        print('crashing or incorrect results should occur now')
        #y = np.rot90(y, k=1)
        #y = np.flipud(y)

    nrow, ncol = y.shape

    # legend
    if not conf['legend']:
        leg_labs=['']*nrow

    else:
        leg_labs = conf['legend']

    # figure canvas
    ax = fig.add_subplot(pos)  # give access to axis properties
    ax.set_axisbelow(True)
    ax.yaxis.set_tick_params(labelsize=conf['ytick_fontsize'])

    # figure options
    plt.grid(axis=conf['grid'])
    plt.xlabel(conf['xlabel'], fontsize=conf['label_fontsize'])
    plt.ylabel(conf['ylabel'], fontsize=conf['label_fontsize'])
    plt.title(conf['title'], fontsize=conf['title_fontsize'])

    if conf['xsep']:
        xticks=addXText(conf, xtext)
        plt.xticks(range(len(conf['xticks'])), xticks, fontsize=conf['xtick_fontsize'], rotation=conf['rotate_xticks'])
    else:
        plt.xticks(range(len(conf['xticks'])), conf['xticks'], fontsize=conf['xtick_fontsize'], rotation=conf['rotate_xticks'])

    if conf['ylim']:
        plt.ylim(conf['ylim'][0], conf['ylim'][1])

    # handle single color strings
    if type(conf['color']) is str:
        conf['color'] = [conf['color']] * nrow

    # create copy of y array so that I retain actual values used for text labels)
    y_text_obj = []

    fixedx=np.arange(ncol)
    yorig=y.copy()
    p = []
    for ind in range(nrow):

        if ind==0:
            pob=plt.bar(fixedx, y[ind], conf['width'], color=conf['color'][ind])

            # center lables on bar segment by taking average ((top+bottom)/2)
            ylocs=y[ind]/2

        else:

            pob = plt.bar(fixedx, y[ind], conf['width'], bottom=sum(y[0:ind]), color=conf['color'][ind])

            # center lables on bar segment by taking average ((top+bottom)/2)
            #ylocs = (y[ind] + y[ind - 1]) / 2
            # ylocs = (sum(y[0:ind+1]) + y[ind])/2
            ylocs = sum(y[0:ind]) + (y[ind] / 2)

        # loop through point on current line and add text to them
        if conf['ytext']:

            text_obj = ytext(fixedx, yorig[ind], conf, alternative_ylocs=ylocs)
            y_text_obj.append(text_obj)

        p.append(pob)

    # if a legend and a position were specified
    if conf['legend'] and conf['legendpos']:

        if conf['legendpos'] == 'outside_top_right':

            plt.legend(p, leg_labs, loc=(1.04, .736), fontsize=conf['legend_fontsize'])

            #for ix in range(nrow):
            #    leg.legendHandles[ix].set_color(conf['color'][ix])

        if conf['legendpos'] == 'outside_bottom_right':
            plt.legend(p,leg_labs, loc=(1.04, 0), fontsize=conf['legend_fontsize'])

            #for ix in range(nrow):
            #    leg.legendHandles[ix].set_color(conf['color'][ix])

    # if a legend but no position was specified
    elif conf['legend']:
        plt.legend(p,leg_labs, conf['color'],fontsize=conf['legend_fontsize'])

        #for ix in range(nrow):
        #    leg.legendHandles[ix].set_color(conf['color'][ix])

    # destroy too close text labels
    if conf['text_destroy'] and conf['ytext']:
        textLabelDestroy(conf, y_text_obj, yorig)

    # text obj can be returned if maunal control over text label is needed
    return y_text_obj

def benchMark(conf,x,y):

    # make benchmark an array if it is not
    if type(conf['bench_data']) is not list:
        conf['bench_data']=[conf['bench_data']]*len(y)

    plt.plot(x, conf['bench_data'], '-o', color=conf['bench_color'], lw=2)

    if 'bench_text' in conf:
        plt.text(x[-1], conf['bench_data'][-1], conf['bench_text'], color=conf['bench_color'],
                 verticalalignment=conf['bench_vert_align'], horizontalalignment='left', size=conf['bench_fontsize'], style='italic')

def ytext(xrow,yrow,conf, alternative_ylocs=None):

    """
    :param xrow: row of x indices
    :param yrow: row of numerical yvals
    :param conf: configuration yaml
    :param alternative_ylocs: positions for the labels that are different from the data array
    :return: A text object that can offer set and get

    Annotate the y markers with the yaxis value. Ignores Nans.
    """
    text_obj = []

    # kill labels at certain indexes
    if type(conf['ytext']) is int:
        conf['ytext']=[conf['ytext']]

    if type(conf['ytext']) is list:
        inds_to_plot = set(conf['ytext'])
        all_inds=set(range(len(yrow)))
        inds_to_kill=list(all_inds-inds_to_plot)
        yrow = yrow.astype(float)
        yrow[inds_to_kill]=np.nan

    # kill labels that are equal to some value
    kill_opt = str(conf['kill_label'])
    if not kill_opt == 'False':
        yrow=yrow.astype(float)
        l=yrow==conf['kill_label']
        yrow[l]=np.nan

    # control text formating based on rounding option
    round_opt=str(conf['round_text'])
    if not round_opt == 'False':
        round_str="{0:." + round_opt + "f}"
    else:
        round_str="{0:.4f}"

    #check if alternative locations for labels are needed
    if alternative_ylocs is not None:
        ylocs=alternative_ylocs
    else:
        ylocs=yrow


    if conf['ytext_suffix']:

        for i, lab in enumerate(xrow):

            if not np.isnan(yrow[i]):

                th = plt.text(lab, ylocs[i], round_str.format(yrow[i]) + conf['ytext_suffix'], color=conf['text_color'],
                              verticalalignment=conf['text_vert_align'], horizontalalignment='center',
                              size=conf['text_fontsize'], style='italic')

                text_obj.append(th)

            else:
                th=False
                text_obj.append(th)

    else:
        for i, lab in enumerate(xrow):

            if not np.isnan(yrow[i]):

                th=plt.text(lab, ylocs[i], round_str.format(yrow[i]), color=conf['text_color'],
                         verticalalignment=conf['text_vert_align'], horizontalalignment='center',
                         size=conf['text_fontsize'], style='italic')

                text_obj.append(th)

            else:
                th=False
                text_obj.append(th)

    return text_obj

# def RepStrList(yam, qry, show=False, opts={}):
#
#     """
#
#     :param yam: yaml configuration (becomes dict) that holds the replacement strings
#     :param qry: yaml configuration that gets duplicated, but with string replacements in it (from yam)
#     :param show: print final list of strings
#     :param opts: dictionary to merge with yaml - allowing pure python configuration on the way in
#     :return: final list of replaced strings
#     """
#
#     yam_struct = yaml.load(yam)
#
#     # merge with possible python dict
#     if opts:
#         yam_struct = {**opts, **yam_struct}
#
#     ks = list(yam_struct.keys())
#
#     # make single element lists into strings/numerical
#     for k in ks:
#         if len(yam_struct[k])==1 and type(yam_struct[k]) is list:
#             yam_struct[k]=yam_struct[k][0]
#
#     # find longest list as this determines block size (number of text blobs)
#     len_grab=[]
#     for k in ks:
#         if type(yam_struct[k]) is list:
#             l=len(yam_struct[k])
#             len_grab.append(l)
#
#     # if no list is present
#     if not len_grab:
#         max_yam=1
#     else:
#         max_yam = max(len_grab)
#
#     # if yaml field is not a list, replicate it as a list
#     for k in ks:
#         if type(yam_struct[k]) is not list:
#             yam_struct[k]=[yam_struct[k]]*max_yam
#
#     block_len=max_yam
#     qry_list = []
#
#     for i in range(block_len):
#
#         tmp=qry
#         for k in ks:
#
#             if '@'+k in tmp:
#
#                 if not yam_struct[k][i]:
#                     tmp=tmp.replace('@'+k, "''")
#                 else:
#                     tmp = tmp.replace('@' + k, str(yam_struct[k][i]))
#
#         # append to make larger list of queries
#         qry_list.append(tmp)
#
#     if show:
#         for q in qry_list:
#             print(q)
#
#     return qry_list

def MakeStrings(src, targ, show=False):

    """
    :param targ: SQL that gets duplicated, but with string replacements in it (from yam)
    :param show: print final list of strings
    :param src: dictionary of values that get put into the SQL query
    :return: final list of replaced strings
    """

    ks = list(src.keys())

    # find longest list as this determines block size (number of text blobs)
    len_grab=[]
    for k in ks:
        if type(src[k]) is list:
            l=len(src[k])
            len_grab.append(l)

    # if no list is present
    if not len_grab:
        max_opts=1
    else:
        max_opts = max(len_grab)

    block_len=max_opts
    qry_list = []

    for i in range(block_len):

        tmp=targ
        for k in ks:

            if '@'+k in tmp:

                if not src[k][i]:
                    tmp=tmp.replace('@'+k, "''")
                else:
                    tmp = tmp.replace('@' + k, str(src[k][i]))

        # append to make larger list of queries
        qry_list.append(tmp)

    if show:
        for q in qry_list:
            print(q)

    return qry_list

def MakeDicts(src, targ, show=False):

    """
    :param src: dict that will give birth to other dicts
    :param targ: dict that holds keys whose values are placed inside of src to create new dicts
    :param show: true/false
    :return: final list of dicts
    """

    ks = list(src.keys())

    # find longest list as this determines block size (number of text blobs)
    len_grab=[]
    for k in ks:
        l=len(src[k])
        len_grab.append(l)

    # if no list is present
    # if not len_grab:
    #     max_opts=1
    # else:
    max_opts = max(len_grab)

    block_len=max_opts
    dict_list = []

    for i in range(block_len):

        tmp=targ.copy()
        for k in ks:

            tmp[k]=src[k][i]
            #print(tmp[k])

        # append to make larger list of queries
        dict_list.append(tmp)

    if show:
        for q in dict_list:
            print(q)

    return dict_list

def runBatchQuery(src, targ, connect, axis=0, show_qry=False, show_res=False, join_on=False):

    # this wraps the function that builds big query strings (with str replaces)
    # it then runs the func that returns the results from the database based on big query
    #query=RepStrList(yam, qry, show=show_qry)
    query = MakeStrings(src, targ, show=show_qry)
    res=runQuery(query, connect, axis=axis, show=show_res, join_on=join_on)

    return res

def textLabelDestroy(conf, yobj, yarray):

    thresh=conf['text_destroy']

    # transposing happens so that lables are compared across rows, this is done if array was previously transposed (stacked bar)
    if conf['axis']==0:
        yobj=np.array(yobj).transpose()
        yarray=yarray.transpose()

    nrows, ncols = yarray.shape

    # move through columns
    for c in range(ncols):

        # set inner iter number
        k = 1

        # set row starting point
        q = 0

        # move through n-1 rows
        for nn in range(nrows-1):

            # set the addition value to control comparison with q
            j=1

            # do pairwise comparisons
            for n in range(nrows-k):

                if yobj[q][c] and yobj[q+j][c]: # check for False elements from ytext func

                    # kill text if needed
                    if abs(yobj[q][c].get_position()[1]-yobj[q+j][c].get_position()[1])<thresh: # index tuple
                        yobj[q][c].set_text('')
                        break

                j=j+1

            # set counter back one to control innermost loop
            k=k+1
            q=q+1

def textLabelAdjust(conf, yobj, yarray):

    """
    Adjust the vertical alignment of text labels so that they dont collide with other labels
    Works well with two lines on a line chart. More than two lines will probably still lead to collisions until code is improved
    When labels are equal, one is removed.

    In the future right and left horizontal alignments might be used to create a robust adjust function for more than two lines/labels

    :param conf: yaml configureation
    :param yobj: text objects to access gets/sets
    :param yarray: ydata just used to get the dimensions
    :return:
    """

    thresh=conf['text_adjust']

    # transposing happens so that lables are compared across rows, this is done if array was previously transposed (stacked bar)
    if conf['axis']==0:
        yobj=np.array(yobj).transpose()
        yarray=yarray.transpose()

    nrows, ncols = yarray.shape

    # move through columns
    for c in range(ncols):

        # set inner iter number
        k = 1

        # set row starting point
        q = 0

        # move through n-1 rows
        for nn in range(nrows-1):

            # set the addition value to control comparison with q
            j=1

            # do pairwise comparisons
            for n in range(nrows-k):

                if yobj[q][c] and yobj[q+j][c]: # check for False elements from ytext func

                    # kill text if needed
                    if abs(yobj[q][c].get_position()[1]-yobj[q+j][c].get_position()[1])<thresh: # index tuple

                        a=yobj[q][c].get_position()[1]
                        b=yobj[q + j][c].get_position()[1]

                        if a>b:
                            yobj[q][c].set_verticalalignment('bottom')
                            yobj[q + j][c].set_verticalalignment('top')
                        elif b>a:
                            yobj[q][c].set_verticalalignment('top')
                            yobj[q + j][c].set_verticalalignment('bottom')
                        elif a==b:
                            yobj[q][c].set_text('')
                        break

                j=j+1

            # set counter back one to control innermost loop
            k=k+1
            q=q+1

def addXText(conf, xtext):

    # concatenate xText to xticks
    # this would be used if you wanted to add an additional label to each item on the x axis

    # if the df is one column, then slicing to [0] returns only one element in "values" array
    # if df has > 1 column, even if it has one row, slicing to [0] returns vector in "values array

    if xtext is not False:

        # convert to dataframe
        if not isinstance(xtext, pd.core.frame.DataFrame):
            xtext = pd.DataFrame(xtext)

        tmp=xtext.values
        r, c=tmp.shape
        if c==1:
            xtext=xtext.values
        else:
            xtext = xtext.values[0]

        # make everythiong strings
        xticks=[str(i) for i in conf['xticks']]
        xtext = [str(i) for i in list(xtext)]

        sep_left=conf['xsep'][0]
        sep_right=conf['xsep'][1]

        newTicks=[]
        for x in zip(xticks, xtext):
            newTicks.append(x[0] + '\n' + sep_left + x[1] + sep_right)

    else:
        newTicks=conf['xticks']

    return newTicks

def addLegText(conf, legtext):

    # this would be used if you wanted to add an additional label to each item on the legend

    #print(legtext)
    # convert to array if data to overlay is not df
    if isinstance(legtext, pd.core.frame.DataFrame):

        vals=legtext.values
        nrw=vals.shape[0]
        if nrw>1:
            legtext=legtext.transpose()
            legtext = legtext.values[0]

    # unnest array
    elif isinstance(legtext, np.ndarray):
        legtext=legtext[0]

    # do nothing, this needs to be refactored as func doesn't need to be called
    else:
        return conf['legend']

    # make everythiong strings
    xticks=[str(i) for i in conf['legend']]
    legtext = [str(i) for i in list(legtext)]

    sep_left=conf['xsep'][0]
    sep_right=conf['xsep'][1]

    newTicks=[]
    for x in zip(xticks, legtext):
        #print(x)
        newTicks.append(x[0] + '\n' + sep_left + x[1] + sep_right)


    #print(newTicks)
    return newTicks

def roundFrame(conf, df):

    # get nan indices
    inds=df.isnull()
    df=df.fillna(0)

    # control rounding
    #df=df.round(int(conf['round_data']))

    if conf['round_data'] == 0:
        df = df.round(int(conf['round_data'])).astype(int)
    else:
        df = df.round(conf['round_data'])

    df[inds]=np.nan
    # df=df.astype(object)

    return df

def autoComment(yam, df, defaults=False):

    # annotate some figure captions automatically

    conf=readConfig(yam, defaults=defaults)

    # control text formating based on rounding option
    round_opt=str(conf['round'])
    if not round_opt == 'False':
        round_str="{0:." + round_opt + "f}"
    else:
        round_str="{0:.4f}"

    # for building markdown strings
    intro = conf['auto_comment'][0] + ' '
    end = ' ' + conf['auto_comment'][1]

    cats=conf['comment_categories']
    if type(cats) is str:
        cats=[cats]

    # grab a unit to display if one was given for y text
    if type(conf['ytext_suffix']) is str:
        unit=conf['ytext_suffix']
    else:
        unit=''

    # deal with nans
    if conf['zeronan']:
        df=df.fillna(0, inplace=False)

    df=roundFrame(conf,df)
    arr=df.values

    if conf['type']=='line':

        # get slice of second last to last columns in array
        vals = arr[:, -2:]
        vals = vals[:, 0] - vals[:, 1]

        # build direction strings
        dirs=[]
        for v in vals:
            if v>0:
                dirs.append(' decreased by ')
            elif v<0:
                dirs.append(' increased by ')
            else:
                dirs.append(' did not change ')


        # # concatenate and build markdown strings
        # intro=conf['auto_comment'][0] + ' '
        # end = ' ' + conf['auto_comment'][1]

        nvals=len(vals)
        md=''

        for i in range(nvals):
            cat=cats[i]
            val=abs(vals[i])
            val=round_str.format(val)
            dr=dirs[i]

            if dr == ' did not change ':
                dr=dr[:-1]

                # reverse indents are important here
                md=md+'''
- {intro}{cat}{dr}{end}'''.format(intro=intro, cat=cat, dr=dr, end=end)

            else:

                md = md + '''
- {intro}{cat}{dr}{val}{unit}{end}'''.format(intro=intro, cat=cat, dr=dr, val=val, unit=unit, end=end)


    elif conf['type']=='clust_bar' and arr.shape[0]==2: # only clusters of 2 allowed

        # get difference of clusters
        vals=arr[0]-arr[1]

        # get index of larges value
        ix = np.where(abs(vals) == max(abs(vals)))[0][0]
        # max_val = max(abs(vals))

        cat1=cats[0]
        cat2=cats[1]
        xcat=conf['xticks'][ix]

        md='''
- The largest difference between {cat1} and {cat2} was for the "{xcat}" category'''.format(cat1=cat1, cat2=cat2, xcat=xcat)


    elif conf['type']=='bar':
        pass

    md=Markdown(md)

    return md

def displayCaption(src, targ, df, defaults=False, direction_flags=None, direction_map=None):

    """
    annotate some figure captions automatically. Give 1D dataframe in any orientation. Numbers get subbed into
    sentences contructed by RepStrList. Bulleted Markdown is created when using display(displayCaption)

    :param src: dict that holds the replacement strings
    :param targ: string block that gets duplicated, but with string replacements in it (from dict)
    :param df: dataframe, series, or array to be used to populate numbers
    :param defaults: default yaml configuration
    :param direction_flags: binary flags in df format. Mapped to yaml configuration direction_map
    :param direction_map: yaml config dict to map binary values to english staements (eg., True: Higher than)
    :return: markdown object to display list of replaced strings
    """

    # deal with main df
    # convert to dataframe
    if not isinstance(df, pd.core.frame.DataFrame):
        df=pd.DataFrame(df)

    #print(src)
    conf = readConfig(src, defaults=defaults)
    #print(conf)

    # deal with nans
    if conf['zeronan']:
        df=df.fillna(0, inplace=False)

    if conf['round_data'] is not False:
        df=roundFrame(conf,df)

    arr=df.values

    # deal with df containing flags if it exists
    # using_direction=False
    if direction_flags is not None and direction_map is not None:

        using_direction=True

        # convert to dataframe
        if not isinstance(direction_flags, pd.core.frame.DataFrame):
            direction_flags=pd.DataFrame(direction_flags)

        # same for flags
        rows, cols= direction_flags.shape
        if rows==1:
            direction_flags=direction_flags.transpose()

        dir_vals = direction_flags.values


        list_block = MakeStrings(src, targ)
        md=''
        for ind, cur_string in enumerate(list_block):

            if using_direction:

                # check for '@direction' to make values absolute and add directional text before value
                cur_num = abs(arr[ind][0])
                cur_dir=dir_vals[ind][0]
                map_text=direction_map[cur_dir]

                #if cur_num !=0:
                cur_string=cur_string.replace('@number', str(cur_num))
                cur_string = cur_string.replace('@direction', map_text)
                #else:

                    # say noting if value is zero
                #    pass
            else:
                # check for '@direction' to make values absolute and add directional text before value
                cur_num = arr[ind][0]
                cur_string=cur_string.replace('@number', str(cur_num))

            tmpmd='''
- {cur_string}'''.format(cur_string=cur_string)
            md=md+tmpmd

        md = Markdown(md)
        return md