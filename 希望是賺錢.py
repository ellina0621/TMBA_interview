##套件#######
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import chardet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import statsmodels.api as sm
from numpy.linalg import lstsq
from dateutil.relativedelta import relativedelta
import mplfinance as mpf
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import ttest_1samp
from arch import arch_model
import statsmodels.formula.api as smf
from datetime import time

#資料處理
df = pd.read_csv("D:/NSYSU FIN/pinpinpinpinpinpin/TXF_R1_1min_data_combined.csv")

#把時間與日期分開
df["datetime"] = pd.to_datetime(df["datetime"])
df["date"] = df["datetime"].dt.date
df["hour"] = df["datetime"].dt.hour
df["minute"] = df["datetime"].dt.minute

########交易策略想法########(皆為當沖)
#過去文獻顯示隔夜正報酬後，隔日會容易出現日內反轉，因noise trader會在往上推生樂觀情緒，導致價格錯價
#開高後，後續快速套利者(HFT、Market maker...)與慢速套利者(fund manager等)會進場將價格拉回，形成拉鋸戰
#而在noise多且流動性佳的情況下，能撈到的報酬應該會更多。若我們能利用某指標知道台灣普遍在有隔夜正報酬後，期貨開盤(8:45-9:00)的資訊價值多少
#隨著現貨開盤(9:00~9:10)、盤中(9:11~13:20)、現貨尾盤(13:21~13:30)及期貨尾盤(13:31~13:45)期貨市場的資訊價值又是多少，做出敘述統計後，
#即使我們不知道任何資訊，我們可以透過統計得出台灣期貨市場在隔夜正報酬後的資訊價值分布情況，我想將其作為權重，
# 若某段時間的資訊價值高，這段期間變動的價格對我來說就是有意義的，假設開盤五分鐘的資訊價值高，這五分鐘內的盤勢為開高往下走，
# 我就假設他之後會往下，我就會在某點位放空，等後續回補。
#那何時要回補平倉? 我會使用另一個指標－資訊不對稱指標OFI，利用掛單的變動，去預測未來市上漲或下跌

df["date"] = df["date"].astype(str)
df = df[df["date"] >= "2017-06-01"]

######處理隔夜正報酬資料######
df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["hour"].astype(str) + ":" + df["minute"].astype(str))
open_846 = df[(df["hour"] == 8) & (df["minute"] == 46)][["date", "Open"]].rename(columns={"Open": "open_846"})
close_1345 = df[(df["hour"] == 13) & (df["minute"] == 45)][["date", "Close"]].rename(columns={"Close": "close_1345"})

daily_oc = pd.merge(open_846, close_1345, on="date", how="inner")

daily_oc["yesterday_close"] = daily_oc["close_1345"].shift(1)
daily_oc["overnight_positive"] = daily_oc["open_846"] > daily_oc["yesterday_close"]

print(daily_oc.head(10))

#抓隔夜正報酬的資料
valid_dates = daily_oc["date"].unique()
overnight_data = df[df["date"].isin(valid_dates)]
print(overnight_data.head(20))


#######計算資訊價值#######
#var / lambda
# 計算 log return (每分鐘)，跟paper same
overnight_data["log_return"] = np.log(overnight_data["Close"]) - np.log(df["Open"])
#print(overnight_data[["date", "hour", "minute", "Open", "Close", "log_return"]].head(20))
overnight_data["r2"] = overnight_data["log_return"]**2
overnight_data["timecode"] = overnight_data["hour"]*100 + overnight_data["minute"]
bins = [846, 900, 910, 1320, 1330, 1345]   # 區間邊界
labels = ["08:46-09:00", "09:01-09:10", "09:11-13:20", "13:21-13:30", "13:31-13:45"]

overnight_data["period"] = pd.cut(overnight_data["timecode"], bins=bins, labels=labels)

intra_var = overnight_data.groupby(["date","period"])["r2"].sum().unstack()

#計算overnight的var，認為此變數會影響8:45-9:00跟9:00-9:10，因此overnight波動度會加入這兩區段的intraday計算
#為了單位要一致，這邊的overnight波動度假設為美股開盤期間的台指期夜盤，以這段期間每分鐘r2累加得出overnight var
#夏令時間（3月中-11月中）：台灣時間21：30~04：00，假設 3/15 ~ 11/15
#冬令時間（11月中-3月中）：台灣時間22：30~05：00，假設 11/16 ~ 3/14
# 3) 逐日抓夜盤資料（跨日區間），每分鐘算 log return 與 r^2
valid_dates = pd.to_datetime(daily_oc["date"]).dt.normalize().unique()
minutes_list = []

for d in valid_dates:
    # 夏令
    if ((d.month > 3 and d.month < 11) or
        (d.month == 3 and d.day >= 15) or
        (d.month == 11 and d.day <= 15)):
        start1 = d - pd.Timedelta(days=1) + pd.Timedelta(hours=21, minutes=30)
        end1   = d
        start2 = d
        end2   = d + pd.Timedelta(hours=4)
    else:
        # 冬令
        start1 = d - pd.Timedelta(days=1) + pd.Timedelta(hours=22, minutes=30)
        end1   = d
        start2 = d
        end2   = d + pd.Timedelta(hours=5)

    # 抓夜盤每分鐘資料（跨日兩段併起來）
    night_data = df[((df["datetime"] >= start1) & (df["datetime"] < end1)) |
                    ((df["datetime"] >= start2) & (df["datetime"] < end2))].copy()


    night_data["log_return"] = np.log(night_data["Close"]) - np.log(night_data["Open"])
    night_data["r2"] = night_data["log_return"]**2

    # 標記隔天是哪一天
    night_data["session_day"] = d.date()

    minutes_list.append(night_data)

overnight_minutes = pd.concat(minutes_list, ignore_index=True)

# 5) 依每個 session_day，加總 r^2 得到 overnight 變異數
overnight_r2 = (
    overnight_minutes.groupby("session_day")["r2"]
    .sum()
    .reset_index()
    .rename(columns={"session_day": "date", "r2": "overnight_r2"})
)

print(overnight_minutes.head())  
print(overnight_r2.head())   

overnight_r2["date"] = pd.to_datetime(overnight_r2["date"])
intra_var2 = intra_var.reset_index()
intra_var2["date"] = pd.to_datetime(intra_var2["date"])
merged = intra_var2.merge(overnight_r2, on="date", how="left")
print(merged.head())
#print(intra_var.head())

#merged.to_excel("merged_intra_overnight_var.xlsx", sheet_name="merged", index=True)
#合併後發現會有一些問題：周一的overnight有跨假日，因此剔除樣本(這樣不太好 但為了方便計算)

#####計算波動度######
#1. overnight + 期貨開盤
var_data= merged.dropna(subset=["overnight_r2"]).copy()
var_data["overnight_plus_open_fx"] = var_data["overnight_r2"] + var_data["08:46-09:00"]

#2. overnight + 股市開盤
var_data["overnight_plus_open_stk"] = var_data["overnight_r2"] + var_data["09:01-09:10"]

print(var_data.head())

#####計算lambda#####
overnight_data["volume_diff"] = overnight_data.groupby("date")["Volume"].diff()

lambda_records = []

for (day, period), group in overnight_data.groupby(["date", "period"]):
    if group["log_return"].isnull().any() or group["volume_diff"].isnull().any():
        continue
    if len(group) < 5:
        continue 

    X = group["volume_diff"]
    y = group["log_return"]

    X = sm.add_constant(X)  
    model = sm.OLS(y, X).fit()

    lambda_hat = model.params[1]  
    lambda_records.append({
        "date": day,
        "period": period,
        "lambda_hat": lambda_hat
    })

lambda_df = pd.DataFrame(lambda_records)#可轉寬格式
lambda_wide = lambda_df.pivot(index="date", columns="period", values="lambda_hat")

#print(lambda_wide.head())

#####合併並計算資訊價值#####
var_data1 = var_data.copy()

var_data1 = var_data1.drop(columns=["08:46-09:00", "09:01-09:10"])

var_data1 = var_data1.rename(columns={
    "overnight_plus_open_fx": "08:46-09:00",
    "overnight_plus_open_stk": "09:01-09:10"
})

ordered_columns = ["date","08:46-09:00","09:01-09:10","09:11-13:20","13:21-13:30","13:31-13:45"]
var_data1 = var_data1[ordered_columns]

#轉成 long format
var_long = var_data1.melt(id_vars="date", var_name="period", value_name="var")
lambda_long = lambda_wide.reset_index().melt(id_vars="date", var_name="period", value_name="lambda")

var_long["date"] = pd.to_datetime(var_long["date"])
lambda_long["date"] = pd.to_datetime(lambda_long["date"])

merged_long = pd.merge(var_long, lambda_long, on=["date", "period"], how="inner")

merged_long['inform_value'] = merged_long['var'] / merged_long['lambda']
#print(merged_long.head())

####資訊價值敘述統計####
inform_stats = merged_long.groupby("period")["inform_value"].describe()
print(inform_stats)

###資訊價值forg乘上price###
overnight_data["datetime"] = pd.to_datetime(overnight_data["datetime"])
overnight_data["timecode"] = overnight_data["hour"] * 100 + overnight_data["minute"]
overnight_data["date"] = pd.to_datetime(overnight_data["date"])

target_timecodes = [500, 900, 910, 1320, 1330]


pt1_df = overnight_data[overnight_data["timecode"].isin(target_timecodes)][["date", "timecode", "Close"]].copy()

timecode_to_col = {
    500: "Pt_0846_0900",
    900: "Pt_0901_0910",
    910: "Pt_0911_1320",
    1320: "Pt_1321_1330",
    1330: "Pt_1331_1345"
}

pt1_df["Pt_label"] = pt1_df["timecode"].map(timecode_to_col)

pt1_wide = pt1_df.pivot(index="date", columns="Pt_label", values="Close").reset_index()

period_to_col = {
    "08:46-09:00": "Pt_0846_0900",
    "09:01-09:10": "Pt_0901_0910",
    "09:11-13:20": "Pt_0911_1320",
    "13:21-13:30": "Pt_1321_1330",
    "13:31-13:45": "Pt_1331_1345"
}
merged_long["pt_col"] = merged_long["period"].map(period_to_col)

merged_long = merged_long.merge(pt1_wide, on="date", how="left")

merged_long["Pt-1"] = merged_long.apply(lambda row: row[row["pt_col"]], axis=1)

merged_long = merged_long.drop(columns=["pt_col"])

merged_long['inform_price'] = merged_long['Pt-1'] * merged_long['inform_value']
#print(merged_long.head())
##統計##
inform_stats = merged_long.groupby("period")["inform_price"].describe()
print(inform_stats) #期貨開盤~現貨開盤前 & 現貨收盤前的資訊價值較佳


###基礎設定###
#單邊成本：$131(fee+tax)
#滑價成本：1tick
#本金：$1,000,000
#1 tick價值：$200
#只選擇當天有隔夜正報酬的日期trade

###IMPORT 2025/08逐筆資料###
from data import combine_data
combined_data = combine_data.copy()
combined_data = combined_data.sort_values(['date','time']).reset_index(drop=True)
combined_data['Price'] = pd.to_numeric(combined_data['Price'], errors='coerce')

combined_data["time"] = pd.to_datetime(combined_data["time"], format="%H:%M:%S")
combined_data["hour"] = combined_data["time"].dt.hour
combined_data["minute"] = combined_data["time"].dt.minute
combined_data["second"] = combined_data["time"].dt.second
combined_data = combined_data.drop(columns=["time"])
combined_data = combined_data.dropna(how="all").reset_index(drop=True)

#####################建立OBI###################################
ofi_trade = combined_data.copy()
ofi_trade = ofi_trade.sort_values(["date","hour","minute","second"]).reset_index(drop=True)

ofi_trade["Pb"] = np.where(ofi_trade["Type"]=="BID", ofi_trade["Price"], np.nan)
ofi_trade["Qb"] = np.where(ofi_trade["Type"]=="BID", ofi_trade["Size"] , np.nan)
ofi_trade["Pa"] = np.where(ofi_trade["Type"]=="ASK", ofi_trade["Price"], np.nan)
ofi_trade["Qa"] = np.where(ofi_trade["Type"]=="ASK", ofi_trade["Size"] , np.nan)


ofi_trade["bid_price_prev"] = ofi_trade["Pb"].replace(0, np.nan).ffill().shift(1)
ofi_trade["bid_size_prev"]  = ofi_trade["Qb"].replace(0, np.nan).ffill().shift(1)
ofi_trade["ask_price_prev"] = ofi_trade["Pa"].replace(0, np.nan).ffill().shift(1)
ofi_trade["ask_size_prev"]  = ofi_trade["Qa"].replace(0, np.nan).ffill().shift(1)
print(ofi_trade)

cols = ["Pb","Qb","Pa","Qa",
        "bid_price_prev","bid_size_prev",
        "ask_price_prev","ask_size_prev"]

ofi_trade[cols] = ofi_trade[cols].fillna(0).astype(float)

ofi_trade["e_n"] = 0.0
ofi_trade.loc[(ofi_trade["Pb"] != 0) & (ofi_trade["Pb"] >= ofi_trade["bid_price_prev"]),
              "e_n"] += ofi_trade["Qb"]

ofi_trade.loc[(ofi_trade["Pb"] != 0) & (ofi_trade["Pb"] <= ofi_trade["bid_price_prev"]),
              "e_n"] -= ofi_trade["bid_size_prev"]

ofi_trade.loc[(ofi_trade["Pa"] != 0) & (ofi_trade["Pa"] <= ofi_trade["ask_price_prev"]),
              "e_n"] -= ofi_trade["Qa"]

ofi_trade.loc[(ofi_trade["Pa"] != 0) & (ofi_trade["Pa"] >= ofi_trade["ask_price_prev"]),
              "e_n"] += ofi_trade["ask_size_prev"]
ofi_per_min = (
    ofi_trade
    .groupby(["date", "hour", "minute"], as_index=False)["e_n"]
    .sum()
    .rename(columns={"e_n": "OFI"})
)

print(ofi_per_min.head())
ofi_per_min.rename(columns={"e_n": "OFI"}, inplace=True)