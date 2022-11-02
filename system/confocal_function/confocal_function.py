"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    Confocal function
    
    使用此程式的注意事項
    
    1. 在跑此程式時，不可以打開輸出的 excel 檔案，不然會無法將跑完的結果存入
    2. 主要修改的地方包含 檔案路徑(base_path)、Sensitivity fall-off function 的路徑(sensitivity_file)、
       輸出檔名(outputFileName)、檔案數(number_of_files)、A-line的 pixel 數(pixel_number)、
       A-line的中心位置(middle_observe)、Rayleigh length(rayleigh_real)
       
    3. Confocal function 的檔名預設為 1、2、3、... 以此類推，如需修改，請到第 42 行的 
       file = os.path.join(base_path, '%d.xlsx' % num)作修改

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import os
import numpy as np
import pandas as pd
import copy

base_path = r'D:\OCT working\confocal function'
sensitivity_file = os.path.join(base_path, 'Sensitivity fall-off function.xlsx')

outputFileName = 'Confocal function.xlsx'

number_of_files = 98

pixel_number = 1024

middle_observe = 512

rayleigh_real = 0.000532


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

#outputFile = pd.concat([outputFile, pd.DataFrame(np.array(max_index), columns=['max value index'])], axis=1)
#outputFile = pd.concat([outputFile, pd.DataFrame(np.array(max_value), columns=['max value'])], axis=1)


#%% This part for interpolation
from scipy.interpolate import UnivariateSpline, CubicSpline

zipped = zip(np.array(max_index), np.array(max_value))

zipped = sorted(zipped, key=lambda x: x[0])

max_index_sorted, max_value_sorted = zip(*zipped)



outputFile = pd.concat([outputFile, pd.DataFrame(np.array(max_index_sorted), columns=['max value index sorted'])], axis=1)
outputFile = pd.concat([outputFile, pd.DataFrame(np.array(max_value_sorted), columns=['max value sorted'])], axis=1)


spline = CubicSpline(max_index_sorted, max_value_sorted)


outputFile = pd.concat([outputFile, pd.DataFrame(np.linspace(0, pixel_number-1, pixel_number),
                                                 columns=['interpolation index'])], axis=1)
    
outputFile = pd.concat([outputFile, pd.DataFrame(spline(np.linspace(0, pixel_number-1, pixel_number)), 
                                                 columns=['interpolation value'])], axis=1)

    
#%% Remove the effect of sensitivity fall-off function from the measured confocal function
data = pd.read_excel(sensitivity_file)

confocal_value = outputFile['interpolation value'] - data['interpolation value']    # log(A/B) = log(A) - log(B)

confocal_value = 10**(confocal_value / 20)    # unit dB turns into unit power

confocal_value = confocal_value - confocal_value[int(max_index_sorted[0]):middle_observe+1].min()
confocal_value = confocal_value / confocal_value[int(max_index_sorted[0]):middle_observe+1].max()
#confocal_value = confocal_value - confocal_value[middle_observe:int(max_index_sorted[-1])+1].min()
#confocal_value = confocal_value / confocal_value[middle_observe:int(max_index_sorted[-1])+1].max()


confocal_value_to_fit = copy.copy(confocal_value)
left_value = 0
right_value = 0

if confocal_value[int(max_index_sorted[0]):middle_observe+1].idxmin() < \
    confocal_value[int(max_index_sorted[0]):middle_observe+1].idxmax():
    
    confocal_value_to_fit[:confocal_value[int(max_index_sorted[0]):middle_observe+1].idxmin()] = 0
    left_value = copy.copy(confocal_value[int(max_index_sorted[0]):middle_observe+1].idxmin())
    
    confocal_value_to_fit[middle_observe+1:] = 0
    right_value = copy.copy(middle_observe)
   
    
else:
    confocal_value_to_fit[:int(max_index_sorted[0])] = 0
    left_value = copy.copy(int(max_index_sorted[0]))
    
    confocal_value_to_fit[confocal_value[int(max_index_sorted[0]):middle_observe+1].idxmin()+1:] = 0
    right_value = copy.copy(confocal_value[int(max_index_sorted[0]):middle_observe+1].idxmin())
    
# if confocal_value[middle_observe:int(max_index_sorted[-1])+1].idxmin() < \
#     confocal_value[middle_observe:int(max_index_sorted[-1])+1].idxmax():
    
#     confocal_value_to_fit[:confocal_value[middle_observe:int(max_index_sorted[-1])+1].idxmin()] = 0
#     left_value = copy.copy(confocal_value[middle_observe:int(max_index_sorted[-1])+1].idxmin())
    
#     confocal_value_to_fit[int(max_index_sorted[-1])+1:] = 0
#     right_value = copy.copy(int(max_index_sorted[-1]))
  
    
# else:
#     confocal_value_to_fit[:middle_observe] = 0
#     left_value = copy.copy(middle_observe)
    
#     confocal_value_to_fit[confocal_value[middle_observe:int(max_index_sorted[-1])+1].idxmin()+1:] = 0
#     right_value = copy.copy(confocal_value[middle_observe:int(max_index_sorted[-1])+1].idxmin())



outputFile = pd.concat([outputFile, pd.DataFrame(np.array(confocal_value), columns=['confocal value'])], axis=1)


#%% fit confocal function
from scipy.optimize import curve_fit

n = 1   # air
alpha = 1   # specular reflection
pixel_size = 0.011 * 10**(-3)


x_fit = np.linspace(left_value, 
                    right_value, 
                    right_value-left_value+1) * pixel_size
                    

y_fit = np.array(confocal_value[left_value:right_value+1])

def confocal(x, rayleigh, z0):
    apparent_rayleigh = alpha * n * rayleigh
    temp = ((x-z0) / apparent_rayleigh)**2    
    return (temp + 1)**(-1)


params = curve_fit(confocal, x_fit, y_fit, p0=[rayleigh_real, x_fit[np.argmax(y_fit)]], 
                                                               bounds=([0,x_fit[0]], [np.inf, x_fit[-1]]))

print('Rayleigh range: ', float(params[0][0]))
print('focal plane: ', float(params[0][1]))


x = np.linspace(0, pixel_number-1, pixel_number) * pixel_size


y_theoretical = confocal(x, params[0][0], params[0][1])

outputFile = pd.concat([outputFile, pd.DataFrame(np.array(y_theoretical), columns=['theoretical confocal value'])], axis=1)
outputFile = pd.concat([outputFile, pd.DataFrame(np.array(params[0]), columns=['Rayleigh range'])], axis=1)


#%% Output and plot on excel file
outputFile = outputFile.replace([np.inf, -np.inf], np.nan)

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter(outputFileName, engine='xlsxwriter')

sheet_name = 'Sheet1'
outputFile.to_excel(writer, sheet_name=sheet_name)


# Access the XlsxWriter workbook and worksheet objects from the dataframe.
workbook = writer.book
worksheet = writer.sheets[sheet_name]

chart1 = workbook.add_chart({'type': 'line'})

for num in range(1, number_of_files+1):
    name = str(num).zfill(zerofill_number) + '_Power'
    
    chart1.add_series({
            'name': name, 
            'categories': [sheet_name, 1, 0, pixel_number, 0],
            'values':     [sheet_name, 1, num, pixel_number, num],
            })


chart1.set_x_axis({'name': 'pixel', 'date_axis': True, 'position_axis': 'on_tick', 'min': 0, 'max': pixel_number-1})
chart1.set_y_axis({'name': 'Power', 'major_gridlines': {'visible': False}, 'min': 90, 'max': 140})

chart1.set_title({'name': 'confocal function'})
chart1.set_legend({'position': 'right'})
chart1.set_size({'width': 1000, 'height': 700})

# Insert the chart into the worksheet.
worksheet.insert_chart('C10', chart1)




chart2_datapoint = workbook.add_chart({'type': 'scatter'})

chart2_datapoint.add_series({
        'name': 'data point',
        'categories': [sheet_name, 1, number_of_files+1, number_of_files*2, number_of_files+1],                              
        'values':     [sheet_name, 1, number_of_files+2, number_of_files*2, number_of_files+2],                             
        'marker': {'type': 'diamond', 
                   'size': 8, 
                   'border': {'color': 'black'}, 
                   'fill': {'color': 'blue'},
                   },
        })


chart2_interpolation = workbook.add_chart({'type': 'line'})

chart2_interpolation.add_series({
        'name': 'interpolation value',
        'categories': [sheet_name, 1, number_of_files+3, pixel_number, number_of_files+3],
        'values':     [sheet_name, 1, number_of_files+4, pixel_number, number_of_files+4],
        'line': {'color': 'red',
                 'width': 5,
                 }
        })


chart2_interpolation.combine(chart2_datapoint)

chart2_interpolation.set_x_axis({'name': 'pixel', 'date_axis': True, 'position_axis': 'on_tick', 
                                 'min': max_index_sorted[0], 'max': max_index_sorted[-1]})

chart2_interpolation.set_y_axis({'name': 'Power_normalized', 'major_gridlines': {'visible': False}, 'min':90, 'max': 140})
chart2_interpolation.set_title({'name': 'peak and interpolation of confocal function'})
chart2_interpolation.set_legend({'position': 'bottom'})
chart2_interpolation.set_size({'width': 1000, 'height': 700})


worksheet.insert_chart('T10', chart2_interpolation)


chart3_confocal_measured = workbook.add_chart({'type': 'line'})

chart3_confocal_measured.add_series({
        'name': 'measured',
        'categories': [sheet_name, 1, number_of_files+3, pixel_number, number_of_files+3],
        'values':     [sheet_name, 1, number_of_files+5, pixel_number, number_of_files+5],
        'line': {'color': 'blue',
                 'width': 3.25,
                 }
        })



   
chart3_confocal_theoretical = workbook.add_chart({'type': 'line'})

chart3_confocal_theoretical.add_series({
        'name': 'theoretical',
        'categories': [sheet_name, 1, number_of_files+3, pixel_number, number_of_files+3],
        'values':     [sheet_name, 1, number_of_files+6, pixel_number, number_of_files+6],
        'line': {'color': 'red',
                 'width': 3.25,
                 }
        })    
    
    
chart3_confocal_measured.combine(chart3_confocal_theoretical)    

chart3_confocal_measured.set_x_axis({'name': 'pixel', 'date_axis': True, 'position_axis': 'on_tick', 
                                 'min': max_index_sorted[0], 'max': max_index_sorted[-1]})

chart3_confocal_measured.set_y_axis({'name': 'Power', 'major_gridlines': {'visible': False}, 'min': 0, 'max': 1.0})
chart3_confocal_measured.set_title({'name': 'peak and interpolation of confocal function'})
chart3_confocal_measured.set_legend({'position': 'bottom'})
chart3_confocal_measured.set_size({'width': 1000, 'height': 700})


worksheet.insert_chart('C50', chart3_confocal_measured)


# Close the Pandas Excel writer and output the Excel file.
writer.save()




