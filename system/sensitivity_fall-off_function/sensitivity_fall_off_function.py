"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    Sensitivity fall-off function
    
    使用此程式的注意事項
    
    1. 在跑此程式時，不可以打開輸出的 excel 檔案，不然會無法將跑完的結果存入
    2. 主要修改的地方包含 檔案路徑(base_path)、輸出檔名(outputFileName)、
       檔案數(number_of_files)、A-line的 pixel 數(pixel_number)、
       A-line的中心位置(middle_observe)
       
    3. Sensitivity fall-off function 的檔名預設為 1、2、3、... 以此類推，
       如需修改，請到第 37 行的 file = os.path.join(base_path, '%d.xlsx' % num)
       作修改

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import os
import numpy as np
import pandas as pd

base_path = r'D:\OCT working\sensitivity fall-off function'
outputFileName = 'Sensitivity fall-off function.xlsx'
number_of_files = 114
pixel_number = 1024

middle_observe = 512


#%% build a complete excel file containing everything
os.chdir(base_path)
outputFile = pd.DataFrame()
zerofill_number = len(str(number_of_files))

max_index = []
max_value = []


for num in range(1, number_of_files+1):
    file = os.path.join(base_path, '%d.xlsx' % num)
    
    data = pd.read_excel(file)
    
    # amplitude to dB
    power = np.array(data.loc[:,data.columns[1]])
    power = 20 * np.log10(power)
    data[data.columns[1]] = pd.Series(power)
 
    
    max_index.append((data.loc[middle_observe:, data.columns[1]]).idxmax())
    max_value.append((data.loc[middle_observe:, data.columns[1]]).max())
        
    max_index.append((data.loc[:middle_observe, data.columns[1]]).idxmax())
    max_value.append((data.loc[:middle_observe, data.columns[1]]).max())
                               
        
    file_name = str(num).zfill(zerofill_number)
            
    data = data.rename(columns={data.columns[0]:'pixel', data.columns[1]:file_name+'_Power'})
            

    outputFile = pd.concat([outputFile, data.loc[:, data.columns[1]]], axis=1)    


outputFile = outputFile.sort_index(axis=1)

outputFile = pd.concat([outputFile, pd.DataFrame(np.array(max_index), columns=['max value index'])], axis=1)
outputFile = pd.concat([outputFile, pd.DataFrame(np.array(max_value), columns=['max value'])], axis=1)



#%% This part for interpolation
from scipy.interpolate import UnivariateSpline, CubicSpline

zipped = zip(outputFile['max value index'].iloc[:number_of_files*2], 
             outputFile['max value'].iloc[:number_of_files*2])

zipped = sorted(zipped, key=lambda x: x[0])

max_index_sorted, max_value_sorted = zip(*zipped)

spline = CubicSpline(max_index_sorted, max_value_sorted)


outputFile = pd.concat([outputFile, pd.DataFrame(np.linspace(0, pixel_number-1, pixel_number), 
                                                 columns=['interpolation index'])], axis=1)

outputFile = pd.concat([outputFile, pd.DataFrame(spline(np.linspace(0, pixel_number-1, pixel_number)), 
                                                 columns=['interpolation value'])], axis=1)


#%% Output and plot on excel file

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter(outputFileName, engine='xlsxwriter')

sheet_name = 'Sheet1'
outputFile.to_excel(writer, sheet_name=sheet_name)


# Access the XlsxWriter workbook and worksheet objects from the dataframe.
workbook = writer.book
worksheet = writer.sheets[sheet_name]

chart1 = workbook.add_chart({'type': 'line'})
chart2 = workbook.add_chart({'type': 'line'})

for num in range(1, number_of_files+1):
    name = str(num).zfill(zerofill_number) + '_Power'
    
    chart1.add_series({
            'name': name, 
            'categories': [sheet_name, 1, 0, pixel_number, 0],
            'values':     [sheet_name, 1, num, pixel_number, num],
            })
    
    if num % 5 == 1:
        chart2.add_series({
            'name': name, 
            'categories': [sheet_name, 1, 0, pixel_number, 0],
            'values':     [sheet_name, 1, num, pixel_number, num],
            })



chart1.set_x_axis({'name': 'pixel', 'date_axis': True, 'position_axis': 'on_tick', 'min': 0, 'max': pixel_number-1})
chart1.set_y_axis({'name': 'Power', 'major_gridlines': {'visible': False}, 'min': 90, 'max': 140})

chart1.set_title({'name': 'sensitivity fall-off function'})
chart1.set_legend({'position': 'right'})
chart1.set_size({'width': 1000, 'height': 700})

# Insert the chart into the worksheet.
worksheet.insert_chart('C10', chart1)



chart2.set_x_axis({'name': 'pixel', 'date_axis': True, 'position_axis': 'on_tick', 'min': middle_observe, 'max': pixel_number-1})
chart2.set_y_axis({'name': 'Power', 'major_gridlines': {'visible': False}, 'min': 80, 'max': 140})

chart2.set_title({'name': 'sensitivity fall-off function'})
chart2.set_legend({'position': 'right'})
chart2.set_size({'width': 1000, 'height': 700})

worksheet.insert_chart('T10', chart2)



chart3_datapoint = workbook.add_chart({'type': 'scatter'})

chart3_datapoint.add_series({
        'name': 'data point',
        'categories': [sheet_name, 1, number_of_files+1, number_of_files*2, number_of_files+1],                            
        'values':     [sheet_name, 1, number_of_files+2, number_of_files*2, number_of_files+2],                             
        'marker': {'type': 'diamond', 
                   'size': 8, 
                   'border': {'color': 'black'}, 
                   'fill': {'color': 'blue'},
                   },
        })




chart3_interpolation = workbook.add_chart({'type': 'line'})

chart3_interpolation.add_series({
        'name': 'interpolation value',
        'categories': [sheet_name, 1, number_of_files+3, pixel_number, number_of_files+3],
        'values':     [sheet_name, 1, number_of_files+4, pixel_number, number_of_files+4],
        'line': {'color': 'red',
                 'width': 5,
                 }
        })



chart3_interpolation.combine(chart3_datapoint)


chart3_interpolation.set_x_axis({'name': 'pixel', 'date_axis': True, 'position_axis': 'on_tick', 
                                 'min': max_index_sorted[0], 'max': max_index_sorted[-1]})

chart3_interpolation.set_y_axis({'name': 'Power', 'major_gridlines': {'visible': False}, 'min': 90, 'max': 140})
chart3_interpolation.set_title({'name': 'peak and interpolation of sensitivity fall-off function'})
chart3_interpolation.set_legend({'position': 'bottom'})
chart3_interpolation.set_size({'width': 1000, 'height': 700})


worksheet.insert_chart('C50', chart3_interpolation)


# Close the Pandas Excel writer and output the Excel file.
writer.save()




