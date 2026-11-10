"""
抓取「通信 / 算力」主题（概念板块）股票的历史日线与每日指标。

输出目录: quant/comm_data/
  - comm_universe.csv      股票池（含所属概念板块）
  - comm_price.csv         日线，列结构同 price_PD.csv
  - comm_daily_basic.csv   每日指标，列结构同 daily_basic.csv
  - raw/price/*.parquet    单只股票中间结果（断点续传用）

可重复运行：已抓过的股票会跳过。
"""
import os
import sys
import time
import traceback

import pandas as pd

# 复用项目里现成的 tushare 连接封装（token 已在其内部）
_ML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wqpython", "ML")
sys.path.insert(0, _ML_DIR)
from tusharecon import connect_tushare  # noqa: E402

# ------------------------- 配置 -------------------------
# 通信 / 算力 主题概念板块（同花顺 THS 指数代码）
BOARDS = {
    "5G": "885556.TI",
    "共封装光学(CPO)": "886033.TI",
    "光纤概念": "886084.TI",
    "东数西算(算力)": "885957.TI",
    "数据中心(AIDC)": "885887.TI",
    "液冷服务器": "886044.TI",
    "算力租赁": "886050.TI",
    "卫星通信": "865176.TI",
    "通信设备": "881129.TI",
    "通信服务": "881162.TI",
}

START_DATE = "20190101"
END_DATE = "20261107"  # 实际只会返回已有数据

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "comm_data")
RAW_PRICE = os.path.join(OUT_DIR, "raw", "price")
RAW_BASIC = os.path.join(OUT_DIR, "raw", "basic")

BASIC_FIELDS = (
    "ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,"
    "ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv"
)

SLEEP = 0.08        # 每次调用间隔，规避频率限制
MAX_RETRY = 5       # 单只股票失败重试次数


def ensure_dirs():
    for d in (OUT_DIR, RAW_PRICE, RAW_BASIC):
        os.makedirs(d, exist_ok=True)


def build_universe(pro):
    """按概念板块取成分并集，附带股票名称与行业。"""
    rows = []
    for board_name, board_code in BOARDS.items():
        try:
            df = pro.ths_member(ts_code=board_code)
        except Exception as e:
            print(f"[warn] ths_member {board_name}({board_code}) 失败: {e}")
            continue
        if df is None or df.empty:
            continue
        for code in df["con_code"].astype(str):
            rows.append((code, board_name))
        print(f"  + {board_name:<16} 成分 {df['con_code'].nunique()} 只")

    mem = pd.DataFrame(rows, columns=["ts_code", "board"])
    # 每只股票所属板块聚合为分号分隔字符串
    boards_map = mem.groupby("ts_code")["board"].apply(lambda s: ";".join(sorted(set(s))))
    universe = boards_map.reset_index().rename(columns={"board": "boards"})
    # 只保留 A 股代码
    universe = universe[universe["ts_code"].str.contains(r"\.(SH|SZ|BJ)$", regex=True)]

    # 补名称/行业/上市日
    basic = pro.stock_basic(
        exchange="", list_status="L",
        fields="ts_code,name,industry,market,list_date",
    )
    universe = universe.merge(basic, on="ts_code", how="left")
    universe = universe.sort_values("ts_code").reset_index(drop=True)
    return universe


def _fetch_with_retry(func, **kwargs):
    last = None
    for i in range(MAX_RETRY):
        try:
            df = func(**kwargs)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def fetch_one(pro, ts_code):
    """抓单只股票的日线 + 每日指标，返回 (price_df, basic_df)。"""
    price = _fetch_with_retry(
        pro.daily, ts_code=ts_code, start_date=START_DATE, end_date=END_DATE
    )
    time.sleep(SLEEP)
    basic = _fetch_with_retry(
        pro.daily_basic, ts_code=ts_code, start_date=START_DATE,
        end_date=END_DATE, fields=BASIC_FIELDS,
    )
    time.sleep(SLEEP)
    return price, basic


def main():
    ensure_dirs()
    pro = connect_tushare(True)

    uni_path = os.path.join(OUT_DIR, "comm_universe.csv")
    if os.path.exists(uni_path):
        universe = pd.read_csv(uni_path, dtype={"ts_code": str})
        print(f"复用已存在的股票池: {len(universe)} 只")
    else:
        print("构建通信概念股票池 ...")
        universe = build_universe(pro)
        universe.to_csv(uni_path, index=False, encoding="utf-8-sig")
        print(f"股票池已保存: {uni_path} ({len(universe)} 只)")

    codes = universe["ts_code"].tolist()
    total = len(codes)
    print(f"开始抓取 {total} 只股票，日期 {START_DATE}~{END_DATE}\n")

    done = 0
    fail = []
    for idx, code in enumerate(codes, 1):
        p_file = os.path.join(RAW_PRICE, f"{code}.parquet")
        b_file = os.path.join(RAW_BASIC, f"{code}.parquet")
        if os.path.exists(p_file) and os.path.exists(b_file):
            done += 1
            continue
        try:
            price, basic = fetch_one(pro, code)
            if not price.empty:
                price.to_parquet(p_file, index=False)
            if not basic.empty:
                basic.to_parquet(b_file, index=False)
            # 空数据也落一个空标记，避免下次重复请求
            if price.empty:
                pd.DataFrame().to_parquet(p_file, index=False)
            if basic.empty:
                pd.DataFrame().to_parquet(b_file, index=False)
            done += 1
        except Exception as e:
            fail.append(code)
            print(f"[fail] {code}: {repr(e)[:100]}")

        if idx % 50 == 0 or idx == total:
            print(f"  进度 {idx}/{total}  成功 {done}  失败 {len(fail)}")

    print(f"\n抓取完成：成功 {done}/{total}，失败 {len(fail)}")
    if fail:
        print("失败代码:", fail)

    # 合并成总表
    print("\n合并分片 ...")
    price_parts, basic_parts = [], []
    for code in codes:
        p_file = os.path.join(RAW_PRICE, f"{code}.parquet")
        b_file = os.path.join(RAW_BASIC, f"{code}.parquet")
        if os.path.exists(p_file):
            df = pd.read_parquet(p_file)
            if not df.empty:
                price_parts.append(df)
        if os.path.exists(b_file):
            df = pd.read_parquet(b_file)
            if not df.empty:
                basic_parts.append(df)

    if price_parts:
        price_all = pd.concat(price_parts, ignore_index=True)
        price_all = price_all.drop_duplicates().sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        price_all.to_csv(os.path.join(OUT_DIR, "comm_price.csv"), index=False, encoding="utf-8-sig")
        print(f"  comm_price.csv: {len(price_all)} 行, {price_all['ts_code'].nunique()} 只股票")

    if basic_parts:
        basic_all = pd.concat(basic_parts, ignore_index=True)
        basic_all = basic_all.drop_duplicates().sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        basic_all.to_csv(os.path.join(OUT_DIR, "comm_daily_basic.csv"), index=False, encoding="utf-8-sig")
        print(f"  comm_daily_basic.csv: {len(basic_all)} 行, {basic_all['ts_code'].nunique()} 只股票")

    print("\n全部完成。输出目录:", OUT_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
