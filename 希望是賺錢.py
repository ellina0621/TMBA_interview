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
import random
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

    
    night_data = df[((df["datetime"] >= start1) & (df["datetime"] < end1)) |
                    ((df["datetime"] >= start2) & (df["datetime"] < end2))].copy()


    night_data["log_return"] = np.log(night_data["Close"]) - np.log(night_data["Open"])
    night_data["r2"] = night_data["log_return"]**2

   
    night_data["session_day"] = d.date()

    minutes_list.append(night_data)

overnight_minutes = pd.concat(minutes_list, ignore_index=True)

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


ofi_per_min.rename(columns={"e_n": "OFI"}, inplace=True)

print(ofi_per_min.head())


####回測+ML####
trade_tape = combined_data[combined_data["Type"]=="TRADE"].copy()
trade_tape["date"] = pd.to_datetime(trade_tape["date"])
trade_tape["dt"]   = pd.to_datetime(
    trade_tape["date"].dt.strftime("%Y-%m-%d") + " " +
    trade_tape["hour"].astype(str)+":"+
    trade_tape["minute"].astype(str)+":"+
    trade_tape["second"].astype(str)
)
trade_tape = trade_tape.sort_values("dt").reset_index(drop=True)
trade_tape = trade_tape[["dt","Price","Size"]]  


# ===================== 1) 只在時間窗產訊號 =====================
def make_signals_window(ofi_min, long_thres, short_thres,
                        date_start, date_end,
                        start_h=8, start_m=45, end_h=10, end_m=0):
    df = ofi_min.copy()

   
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["dt_min"] = (pd.to_datetime(df["date"].astype(str))
                    + pd.to_timedelta(df["hour"].astype(int),   unit="h")
                    + pd.to_timedelta(df["minute"].astype(int), unit="m"))

   
    d0 = pd.to_datetime(date_start).date()
    d1 = pd.to_datetime(date_end).date()
    df = df[(df["date"]>=d0) & (df["date"]<=d1)].copy()

    from datetime import time as _time
    t0 = _time(start_h, start_m)
    t1 = _time(end_h,   end_m)
    df["t_in_day"] = df["dt_min"].dt.time
    df = df[(df["t_in_day"]>=t0) & (df["t_in_day"]<=t1)].copy()

    
    df["signal"] = 0
    df.loc[df["OFI"] >  long_thres, "signal"] = 1
    df.loc[df["OFI"] <  short_thres,"signal"] = -1

    sig = df.loc[df["signal"]!=0, ["dt_min","signal"]].sort_values("dt_min").reset_index(drop=True)
    return sig

# ===================== 2) 回測（單一倉位 + 某段區間） =====================
def backtest_once_window(trade_tape, signals,
                         tp_ticks=20, sl_ticks=15,
                         tick_size=1.0, tick_value=200,
                         fee_one_side=131, slippage_ticks=1,
                         end_h=10, end_m=0):
    """
    進場=訊號後第一筆 TRADE；停利/停損/到 end_h:end_m 強制平；單一倉位。
    """
    from datetime import time as _time

    pos = 0
    entry_px = entry_t = None
    round_cost = 2 * fee_one_side
    slippage_cost = slippage_ticks * tick_value
    rec = []

    T = trade_tape.sort_values("dt").reset_index(drop=True)
    t_idx = 0

    for _, s in signals.iterrows():
        if pos != 0:
            continue

        sig_t = s["dt_min"]
        side  = s["signal"]

        while t_idx < len(T) and T.iloc[t_idx]["dt"] < sig_t:
            t_idx += 1
        if t_idx >= len(T):
            break

        entry_t = T.iloc[t_idx]["dt"]
        entry_px= float(T.iloc[t_idx]["Price"])
        pos = side

       
        day_end = pd.Timestamp(entry_t.date()) + pd.Timedelta(hours=end_h, minutes=end_m)

        tp_px = entry_px + (tp_ticks * tick_size) * (1 if pos==1 else -1)
        sl_px = entry_px - (sl_ticks * tick_size) * (1 if pos==1 else -1)

        j = t_idx + 1
        exit_t = exit_px = None
        while j < len(T):
            px = float(T.iloc[j]["Price"])
            t  = T.iloc[j]["dt"]

            # 觸價平倉
            if pos==1 and (px >= tp_px or px <= sl_px):
                exit_t, exit_px = t, px
                break
            if pos==-1 and (px <= tp_px or px >= sl_px):
                exit_t, exit_px = t, px
                break

            # 達到 time-stop → 用 >= 門檻的第一筆 TRADE 出場
            if t >= day_end:
                exit_t, exit_px = t, px
                break
            j += 1

        if exit_t is None:
            break  # 後面沒有成交，無法平倉

        pnl_ticks = (exit_px - entry_px) if pos==1 else (entry_px - exit_px)
        pnl_money = pnl_ticks * tick_value - round_cost - slippage_cost

        rec.append({
            "entry_time": entry_t, "entry_px": entry_px, "side": pos,
            "exit_time": exit_t,  "exit_px": exit_px,
            "pnl_ticks": pnl_ticks, "pnl_money": pnl_money
        })

        pos = 0
        t_idx = j + 1
        if t_idx >= len(T):
            break

    recdf = pd.DataFrame(rec)
    if recdf.empty:
        return 0.0, 0, 0.0, recdf
    return recdf["pnl_money"].sum(), len(recdf), (recdf["pnl_money"]>0).mean(), recdf

# ===================== 3) 樣本內 Random Search =====================
def random_search_IS(ofi_min, trade_tape,
                     date_start, date_end,
                     n_iter=200,
                     long_range=(50, 400), short_range=(-400, -50),
                     tp_range=(5, 40), sl_range=(5, 40),
                     start_h=8, start_m=45, end_h=10, end_m=0,
                     seed=42,
                     capital=1_000_000):
    random.seed(seed)
    trials = []
    for _ in range(n_iter):
        L  = random.uniform(*long_range)
        S  = random.uniform(*short_range)
        TP = int(random.uniform(*tp_range))
        SL = int(random.uniform(*sl_range))

        sig = make_signals_window(ofi_min, L, S,
                                  date_start=date_start, date_end=date_end,
                                  start_h=start_h, start_m=start_m,
                                  end_h=end_h, end_m=end_m)

        pnl, n_tr, wr, trades_df = backtest_once_window(
            trade_tape, sig,
            tp_ticks=TP, sl_ticks=SL,
            end_h=end_h, end_m=end_m
        )

        # ---- 新增：計算指標 ----
        mets = compute_trade_metrics(trades_df, capital=capital)

        trials.append({
            "L": L, "S": S, "TP": TP, "SL": SL,
            "PnL": pnl, "Trades": n_tr, "WinRate": wr,
            "Sharpe": mets["Sharpe"],
            "MDD": mets["MDD"],
            "ProfitFactor": mets["ProfitFactor"],
            "AvgWinLoss": mets["AvgWinLoss"]
        })

    return (pd.DataFrame(trials)
              .sort_values(["PnL", "Trades"], ascending=[False, False])
              .reset_index(drop=True))

# ===================== 4) Walk-Forward（每日移動） =====================
def walk_forward_daily(ofi_min, trade_tape,
                       train_start="2025-08-01",
                       test_start="2025-08-13",
                       test_end="2025-08-14",
                       mode="expanding",           # "expanding" 或 "rolling"
                       lookback_days=10,           # rolling 時使用
                       n_iter=300,
                       long_range=(50,400), short_range=(-400,-50),
                       tp_range=(5,40), sl_range=(5,40),
                       start_h=8, start_m=45, end_h=10, end_m=0,
                       seed=42):
    """
    對每個測試日 t：用 [train_start, t-1] (expanding) 或 [t-lookback+1, t-1] (rolling) 做 Random Search，
    取最佳參數，並在 t 當天的時間窗回測。
    回傳： (summary_df, all_trades_df)
    """
    random.seed(seed)

    # 可交易日期（兩個資料都要有）
    trade_days = pd.to_datetime(trade_tape["dt"]).dt.date.unique()
    ofi_days   = pd.to_datetime(ofi_min["date"]).astype("datetime64[ns]").dt.date.unique()
    valid_days = sorted(list(set(trade_days).intersection(set(ofi_days))))

    test_days = pd.date_range(test_start, test_end, freq="D").date
    test_days = [d for d in test_days if d in valid_days]

    daily_rows, trades_all = [], []

    for tday in test_days:
        train_end = pd.Timestamp(tday) - pd.Timedelta(days=1)
        if mode == "expanding":
            cur_train_start = pd.Timestamp(train_start)
        else:  # rolling 視窗
            cur_train_start = max(pd.Timestamp(train_start),
                                  train_end - pd.Timedelta(days=lookback_days-1))

        if cur_train_start.date() >= train_end.date():
            continue  # 沒有訓練天數

        
        res = random_search_IS(
            ofi_min, trade_tape,
            date_start=cur_train_start.date().isoformat(),
            date_end=train_end.date().isoformat(),
            n_iter=n_iter,
            long_range=long_range, short_range=short_range,
            tp_range=tp_range, sl_range=sl_range,
            start_h=start_h, start_m=start_m, end_h=end_h, end_m=end_m,
            seed=seed
        )
        if res.empty:
            continue
        best = res.iloc[0]

       
        sig = make_signals_window(
            ofi_min, best["L"], best["S"],
            date_start=tday, date_end=tday,
            start_h=start_h, start_m=start_m, end_h=end_h, end_m=end_m
        )

        
        pnl, n_tr, wr, trades = backtest_once_window(
            trade_tape, sig,
            tp_ticks=int(best["TP"]), sl_ticks=int(best["SL"]),
            end_h=end_h, end_m=end_m
        )

        daily_rows.append({
            "date": pd.to_datetime(tday),
            "L": best["L"], "S": best["S"], "TP": int(best["TP"]), "SL": int(best["SL"]),
            "PnL": pnl, "Trades": n_tr, "WinRate": wr
        })
        if not trades.empty:
            trades["test_date"] = pd.to_datetime(tday)
            trades_all.append(trades)

    summary = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    if not summary.empty:
        summary["cumPnL"] = summary["PnL"].cumsum()
    trades_all = pd.concat(trades_all, ignore_index=True) if len(trades_all)>0 else pd.DataFrame()
    return summary, trades_all

def compute_trade_metrics(trades_df, capital=1_000_000):
    """
    trades_df 需要包含欄位：
      - 'pnl_money'：每筆損益金額
      - 'exit_time'：平倉時間 (datetime)
    回傳：Sharpe（年化、以每日PnL/資金計算）、MDD（金額）、ProfitFactor、AvgWinLoss
    """
    import numpy as np
    import pandas as pd

    if trades_df is None or trades_df.empty:
        return {"Sharpe": np.nan, "MDD": 0.0, "ProfitFactor": np.nan, "AvgWinLoss": np.nan}

    # === MDD（用逐筆損益累積的權益曲線）===
    pnl = trades_df["pnl_money"].to_numpy()
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    mdd_abs = float(-drawdown.min()) if drawdown.size > 0 else 0.0  # 金額

    # === 每日報酬，用每日PnL/資金 ===
    daily = (trades_df.assign(date=trades_df["exit_time"].dt.date)
             .groupby("date")["pnl_money"].sum())
    ret = daily / float(capital)
    sharpe = float((ret.mean() / ret.std() * np.sqrt(252)) if ret.std() and ret.std() > 0 else np.nan)

    # === 賺賠比 ===
    profits = trades_df.loc[trades_df["pnl_money"] > 0, "pnl_money"]
    losses  = trades_df.loc[trades_df["pnl_money"] < 0, "pnl_money"]

    total_profit = profits.sum()
    total_loss   = -losses.sum()

    profit_factor = float(total_profit / total_loss) if total_loss > 0 else (np.inf if total_profit > 0 else np.nan)
    avg_win_loss  = float(profits.mean() / (-losses.mean())) if (len(profits) > 0 and len(losses) > 0) else np.nan

    return {"Sharpe": sharpe, "MDD": mdd_abs, "ProfitFactor": profit_factor, "AvgWinLoss": avg_win_loss}

# 只做 08:45–10
wf_sum, wf_trades = walk_forward_daily(
    ofi_per_min, trade_tape,
    train_start="2025-08-01",
    test_start="2025-08-13",
    test_end="2025-08-14",
    mode="expanding",          # 8/14 用 8/1~8/13；8/15 用 8/1~8/14；以此類推
    n_iter=300,
    start_h=8, start_m=45, end_h=10, end_m=0
)

print(wf_sum)
print("樣本外總PnL：", wf_sum["PnL"].sum())


# === OOS 整段指標===
capital = 1_000_000 
oos_mets = compute_trade_metrics(wf_trades, capital=capital)


if not wf_sum.empty:
    eq = wf_sum["PnL"].cumsum().to_numpy()
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    oos_MDD_equity = float(-dd.min()) if dd.size > 0 else 0.0
else:
    oos_MDD_equity = 0.0

print("\n===== OOS 指標（8/13–8/31）=====")
print(f"Sharpe: {oos_mets['Sharpe']:.3f}  (以每日PnL/資金年化)")
print(f"MDD (金額): {oos_mets['MDD']:.0f}")
print(f"ProfitFactor: {oos_mets['ProfitFactor']:.3f}")
print(f"AvgWinLoss: {oos_mets['AvgWinLoss']:.3f}")
print(f"MDD(以每日權益算): {oos_MDD_equity:.0f}")

