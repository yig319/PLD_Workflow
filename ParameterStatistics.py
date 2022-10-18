import ast
import re
import pandas as pd
import hvplot.pandas

def PlotParameter(df_api, ls_resp, parameter):
    '''
    This function is designed to plot the parameter change with time.

    :param ls_resp: collectionItemsList
    :type ls_resp: collectionItemsList

    :param parameter: PLD experiment parameter to track and plot 
    :type parameter: str
    
    :param plot_type: specify which type of plot to show, default: 'normal', option: 'hover'
    :type plot_type: str
    '''
    
    ID_list, time_stamp_list, metric_list = [], [], []

    for i in range(len(ls_resp[0].item)):

        id = ls_resp[0].item[i].id
        labels = ast.literal_eval(df_api.dataView(id)[0].data[0].metadata)
        
        if 'target_1' not in labels.keys(): continue
            
        if parameter in labels['target_1'].keys():
            metric = labels['target_1'][parameter]
        else:
            metric = 0

        if metric != 0:
            ID_list.append(labels['header']['Growth ID'])
#             print(labels['header']['Growth ID'])

            date = labels['header']['Date']    
            time_stamp = date[-4:]+'-'+date[:2]+'-'+date[3:5]
            if 'time' in labels['header'].keys():
                time = labels['header']['time']
                time_stamp = date + ' ' + time
            time_stamp_list.append(time_stamp)
#             print(time_stamp)

            metric = float(re.split('/', metric)[-1])
            metric_list.append(metric)

    df = pd.DataFrame({
                   'ID': ID_list,
                   'Date': time_stamp_list,
                    parameter: metric_list
                  })
    
    df_hover = df.set_index(pd.DatetimeIndex(df.iloc[:,1]))
    df_hover.drop('Date', 1, inplace=True)
    ax = df_hover.hvplot.scatter(hover_cols=['ID'])
    display(ax)
        
    return df