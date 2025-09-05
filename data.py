##套件#######
import pandas as pd
import numpy as np
import chardet
### import data ####
def read_csv_auto_encoding(path):
    with open(path, 'rb') as f:
        raw_data = f.read(20000)
    detect_result = chardet.detect(raw_data)
    encoding = detect_result['encoding']
    print(f"{path} 偵測到編碼：{encoding}")
    return pd.read_csv(path, encoding=encoding)

path_data1  = r"D:/台新/pin/FTQ5_intraday_data_818.csv"
path_data2  = r"D:/台新/pin/FTQ5_intraday_data_819.csv"
path_data3  = r"D:/台新/pin/FTQ5_intraday_data_820.csv"
path_data4  = r"D:/台新/pin/FTQ5_intraday_data_813.csv"
path_data5  = r"D:/台新/pin/FTQ5_intraday_data_815.csv"
path_data6  = r"D:/台新/pin/FTQ5_intraday_data_814.csv"
path_data7  = r"D:/台新/pin/FTQ5_intraday_data_813.csv"
path_data8  = r"D:/台新/pin/FTQ5_intraday_data_811.csv"
path_data9  = r"D:/台新/pin/FTQ5_intraday_data_808.csv"
path_data10 = r"D:/台新/pin/FTQ5_intraday_data_807.csv"
path_data11 = r"D:/台新/pin/FTQ5_intraday_data_805.csv"
path_data12 = r"D:/台新/pin/FTQ5_intraday_data_804.csv"
path_data13 = r"D:/台新/pin/FTQ5_intraday_data_801.csv"
path_data14 = r"D:/台新/pin/FTQ5_intraday_data_829.csv"
path_data15 = r"D:/台新/pin/FTQ5_intraday_data_828.csv"
path_data16 = r"D:/台新/pin/FTQ5_intraday_data_827.csv"
path_data17 = r"D:/台新/pin/FTQ5_intraday_data_826.csv"
path_data18 = r"D:/台新/pin/FTQ5_intraday_data_822.csv"
path_data19 = r"D:/台新/pin/FTQ5_intraday_data_825.csv"
path_data20 = r"D:/台新/pin/FTQ5_intraday_data_812.csv"
path_data21 = r"D:/台新/pin/FTQ5_intraday_data_821.csv"
path_data22 = r"D:/台新/pin/FTQ5_intraday_data_807.csv"

data_818 = read_csv_auto_encoding(path_data1)
data_819 = read_csv_auto_encoding(path_data2)
data_820 = read_csv_auto_encoding(path_data3)
data_813 = read_csv_auto_encoding(path_data4)
data_815 = read_csv_auto_encoding(path_data5)
data_814 = read_csv_auto_encoding(path_data6)
data_813 = read_csv_auto_encoding(path_data7)
data_811 = read_csv_auto_encoding(path_data8)
data_808 = read_csv_auto_encoding(path_data9)
data_807 = read_csv_auto_encoding(path_data10)
data_805 = read_csv_auto_encoding(path_data11)
data_804 = read_csv_auto_encoding(path_data12)
data_801 = read_csv_auto_encoding(path_data13)
data_829 = read_csv_auto_encoding(path_data14)
data_828 = read_csv_auto_encoding(path_data15)
data_827 = read_csv_auto_encoding(path_data16)
data_826 = read_csv_auto_encoding(path_data17)
data_822 = read_csv_auto_encoding(path_data18)
data_825 = read_csv_auto_encoding(path_data19)
data_812 = read_csv_auto_encoding(path_data20)
data_821 = read_csv_auto_encoding(path_data21)
data_807 = read_csv_auto_encoding(path_data22)

####處理基本資料######
#(1) 2025/08/18
data_818 = data_818[data_818['Dates'] != 'Dates'].copy()
data_818['Dates'] = pd.to_datetime(
    data_818['Dates'],
    format='%Y/%m/%d %p %I:%M:%S'
)
data_818['date'] = data_818['Dates'].dt.date
data_818['time'] = data_818['Dates'].dt.time
data_818 = data_818.drop(columns=['Dates'])

#(2) 2025/08/19
data_819 = data_819[data_819['Dates'] != 'Dates'].copy()
data_819['Dates'] = pd.to_datetime(
    data_819['Dates'],
    format='%Y/%m/%d %p %I:%M:%S'
)
data_819['date'] = data_819['Dates'].dt.date
data_819['time'] = data_819['Dates'].dt.time
data_819 = data_819.drop(columns=['Dates'])

#(3) 2025/08/20
data_820 = data_820[data_820['Dates'] != 'Dates'].copy()
data_820['Dates'] = pd.to_datetime(
    data_820['Dates'],
    format='%Y/%m/%d %p %I:%M:%S'
)
data_820['date'] = data_820['Dates'].dt.date
data_820['time'] = data_820['Dates'].dt.time
data_820 = data_820.drop(columns=['Dates'])

#(5) 2025/08/15
data_815 = data_815[data_815['Dates'] != 'Dates'].copy()
data_815['Dates'] = pd.to_datetime(
    data_815['Dates'],
    format='%Y/%m/%d %p %I:%M:%S'
)
data_815['date'] = data_815['Dates'].dt.date
data_815['time'] = data_815['Dates'].dt.time
data_815 = data_815.drop(columns=['Dates'])


#(6) 2025/08/14
data_814 = data_814[data_814['Dates'] != 'Dates'].copy()
data_814['Dates'] = pd.to_datetime(
    data_814['Dates'],
    format='%Y/%m/%d %p %I:%M:%S'
)
data_814['date'] = data_814['Dates'].dt.date
data_814['time'] = data_814['Dates'].dt.time
data_814 = data_814.drop(columns=['Dates'])

#(7) 2025/08/13
data_813 = data_813[data_813['Dates'] != 'Dates'].copy()
data_813['Dates'] = pd.to_datetime(
    data_813['Dates'],
    format='%Y/%m/%d %p %I:%M:%S'
)
data_813['date'] = data_813['Dates'].dt.date
data_813['time'] = data_813['Dates'].dt.time
data_813 = data_813.drop(columns=['Dates'])

#(9) 2025/08/08
data_808 = data_808[data_808['Dates'] != 'Dates'].copy()
data_808['Dates'] = pd.to_datetime(
    data_808['Dates'],
    format='%Y/%m/%d %p %I:%M:%S'
)
data_808['date'] = data_808['Dates'].dt.date
data_808['time'] = data_808['Dates'].dt.time
data_808 = data_808.drop(columns=['Dates'])

#(11) 2025/08/05 
data_805 = data_805[data_805['Dates'] != 'Dates'].copy()
data_805['Dates'] = pd.to_datetime(
    data_805['Dates'],
    format='%Y/%m/%d %p %I:%M:%S'
)   
data_805['date'] = data_805['Dates'].dt.date
data_805['time'] = data_805['Dates'].dt.time
data_805 = data_805.drop(columns=['Dates'])

# (14) 2025/08/29
data_829 = data_829[data_829['Dates'] != 'Dates'].copy()
data_829['Dates'] = pd.to_datetime(data_829['Dates'], format='%Y/%m/%d %p %I:%M:%S')
data_829['date'] = data_829['Dates'].dt.date
data_829['time'] = data_829['Dates'].dt.time
data_829 = data_829.drop(columns=['Dates'])

# (15) 2025/08/28
data_828 = data_828[data_828['Dates'] != 'Dates'].copy()
data_828['Dates'] = pd.to_datetime(data_828['Dates'], format='%Y/%m/%d %p %I:%M:%S')
data_828['date'] = data_828['Dates'].dt.date
data_828['time'] = data_828['Dates'].dt.time
data_828 = data_828.drop(columns=['Dates'])

# (16) 2025/08/27
data_827 = data_827[data_827['Dates'] != 'Dates'].copy()
data_827['Dates'] = pd.to_datetime(data_827['Dates'], format='%Y/%m/%d %p %I:%M:%S')
data_827['date'] = data_827['Dates'].dt.date
data_827['time'] = data_827['Dates'].dt.time
data_827 = data_827.drop(columns=['Dates'])

# (17) 2025/08/26
data_826 = data_826[data_826['Dates'] != 'Dates'].copy()
data_826['Dates'] = pd.to_datetime(data_826['Dates'], format='%Y/%m/%d %p %I:%M:%S')
data_826['date'] = data_826['Dates'].dt.date
data_826['time'] = data_826['Dates'].dt.time
data_826 = data_826.drop(columns=['Dates'])

# (18) 2025/08/22
data_822 = data_822[data_822['Dates'] != 'Dates'].copy()
data_822['Dates'] = pd.to_datetime(data_822['Dates'], format='%Y/%m/%d %p %I:%M:%S')
data_822['date'] = data_822['Dates'].dt.date
data_822['time'] = data_822['Dates'].dt.time
data_822 = data_822.drop(columns=['Dates'])

# (19) 2025/08/25
data_825 = data_825[data_825['Dates'] != 'Dates'].copy()
data_825['Dates'] = pd.to_datetime(data_825['Dates'], format='%Y/%m/%d %p %I:%M:%S')
data_825['date'] = data_825['Dates'].dt.date
data_825['time'] = data_825['Dates'].dt.time
data_825 = data_825.drop(columns=['Dates'])

# (20) 2025/08/12
data_812 = data_812[data_812['Dates'] != 'Dates'].copy()
data_812['Dates'] = pd.to_datetime(data_812['Dates'], format='%Y/%m/%d %p %I:%M:%S')
data_812['date'] = data_812['Dates'].dt.date
data_812['time'] = data_812['Dates'].dt.time
data_812 = data_812.drop(columns=['Dates'])

# (21) 2025/08/21
data_821 = data_821[data_821['Dates'] != 'Dates'].copy()
data_821['Dates'] = pd.to_datetime(data_821['Dates'], format='%Y/%m/%d %p %I:%M:%S')
data_821['date'] = data_821['Dates'].dt.date
data_821['time'] = data_821['Dates'].dt.time
data_821 = data_821.drop(columns=['Dates'])

# (22) 2025/08/07
data_807 = data_807[data_807['Dates'] != 'Dates'].copy()
data_807['Dates'] = pd.to_datetime(data_807['Dates'], format='%Y/%m/%d %p %I:%M:%S')
data_807['date'] = data_807['Dates'].dt.date
data_807['time'] = data_807['Dates'].dt.time
data_807 = data_807.drop(columns=['Dates'])

# (23) 2025/08/01
data_801 = data_801[data_801['Dates'] != 'Dates'].copy()
data_801['Dates'] = pd.to_datetime(data_801['Dates'], format='%Y/%m/%d %p %I:%M:%S')
data_801['date'] = data_801['Dates'].dt.date
data_801['time'] = data_801['Dates'].dt.time
data_801 = data_801.drop(columns=['Dates'])

#print(data_820,data_818,data_819.head())

combine_data = pd.concat([data_801, data_804, data_805, data_807, data_808, data_811, data_812, data_813, data_814, data_815, data_818, data_819, data_820, data_821, data_822, data_825, data_826, data_827, data_828, data_829], ignore_index=True)

combine_data = combine_data.sort_values(by=['date', 'time']).reset_index(drop=True) 
