import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import traceback

# ==========================================
# 0. 页面基础配置
# ==========================================
st.set_page_config(
    page_title="缠论分析", 
    layout="centered"
)
st.title("📈 简易缠论分析")

# ==========================================
# 1. 核心数据获取 (增强版)
# ==========================================
@st.cache_data(ttl=600) 
def get_stock_data(code_input):
    # 清洗代码
    symbol = code_input.replace('sh.', '').replace('sz.', '').strip()
    
    # 时间设定 (最近365天)
    end_dt = pd.Timestamp.now().strftime('%Y%m%d')
    start_dt = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime('%Y%m%d')
    
    # --- 尝试线路 1 (东方财富 - 历史行情) ---
    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol, 
            period="daily", 
            start_date=start_dt, 
            end_date=end_dt, 
            adjust="qfq"
        )
        if not df.empty:
            # 统一列名
            df = df.rename(columns={
                '日期': 'date', '开盘': 'open', '收盘': 'close', 
                '最高': 'high', '最低': 'low', '成交量': 'volume'
            })
            df['date'] = pd.to_datetime(df['date'])
            return df, "线路1 (历史行情)"
    except Exception as e1:
        st.warning(f"⚠️ 线路1访问受阻，正在尝试线路2... (错误: {str(e1)})")

    # --- 尝试线路 2 (实时行情 - 最近交易日) ---
    # 如果海外IP被封历史接口，有时候实时接口能通
    try:
        df = ak.stock_zh_a_spot_em()
        # 筛选单只股票
        df = df[df['代码'] == symbol]
        if not df.empty:
            # 只有一行数据，虽然不能画图，但至少能证明连通性
            # 这里为了跑通缠论，我们其实需要历史数据，如果线路1挂了，
            # 线路2通常只能救急看当前价，无法计算MACD。
            # 所以这里抛出更详细的错误给用户
            raise Exception("无法获取历史K线，无法计算指标")
    except Exception as e2:
        pass

    return pd.DataFrame(), f"所有线路均失败。请查看下方错误详情。"

# ==========================================
# 2. 缠论计算逻辑
# ==========================================
def calculate_indicators(df):
    df = df.copy()
    for c in ['close', 'high', 'low']: 
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['dif'] = df['ema12'] - df['ema26']
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['dif'] - df['dea']) * 2
    return df

def preprocess_inclusion(df):
    if len(df) < 2: return df
    data = df.to_dict('records')
    for d in data: 
        if 'real_date' not in d: d['real_date'] = d['date']

    processed = [data[0]]
    direction = 1 
    if data[1]['close'] < data[0]['close']: direction = -1

    for i in range(1, len(data)):
        cur = data[i]
        last = processed[-1]
        is_cur_in = (cur['high'] <= last['high'] and cur['low'] >= last['low'])
        is_last_in = (cur['high'] >= last['high'] and cur['low'] <= last['low'])
        
        if is_cur_in or is_last_in:
            last['date'] = cur['date']
            last['close'] = cur['close']
            if direction == 1:
                last['high'] = max(last['high'], cur['high'])
                last['low'] = max(last['low'], cur['low'])
            else:
                last['high'] = min(last['high'], cur['high'])
                last['low'] = min(last['low'], cur['low'])
        else:
            if cur['high'] > last['high'] and cur['low'] > last['low']: direction = 1
            elif cur['high'] < last['high'] and cur['low'] < last['low']: direction = -1
            processed.append(cur)
    return pd.DataFrame(processed)

def calculate_bi(df):
    if len(df) < 10: return []
    k_df = preprocess_inclusion(df)
    k_df = k_df.reset_index(drop=True)
    if len(k_df) < 5: return []

    k_df['fx_type'] = 0 
    for i in range(1, len(k_df)-1):
        prev = k_df.iloc[i-1]; curr = k_df.iloc[i]; next_ = k_df.iloc[i+1]
        if curr['high'] > prev['high'] and curr['high'] > next_['high']:
            if curr['low'] > prev['low'] and curr['low'] > next_['low']: k_df.loc[i, 'fx_type'] = 1
        elif curr['low'] < prev['low'] and curr['low'] < next_['low']:
            if curr['high'] < prev['high'] and curr['high'] < next_['high']: k_df.loc[i, 'fx_type'] = -1

    fractals = k_df[k_df['fx_type'] != 0].copy()
    if len(fractals) < 2: return []

    bi_list = []
    stack = [fractals.iloc[0]]
    for i in range(1, len(fractals)):
        curr = fractals.iloc[i]
        last = stack[-1]
        if curr['fx_type'] == last['fx_type']:
            if curr['fx_type'] == 1 and curr['high'] > last['high']: stack.pop(); stack.append(curr)
            elif curr['fx_type'] == -1 and curr['low'] < last['low']: stack.pop(); stack.append(curr)
        elif curr.name - last.name >= 3:
            stack.append(curr)
            start_n = stack[-2]; end_n = stack[-1]
            sub_df = df[(df['date'] >= start_n['real_date']) & (df['date'] <= end_n['real_date'])]
            macd_sum = sub_df['macd'].abs().sum() if 'macd' in sub_df.columns else 0
            bi_list.append({
                '方向': '向上' if start_n['fx_type'] == -1 else '向下',
                '日期': end_n['real_date'].strftime('%Y-%m-%d'),
                '价格': float(end_n['high'] if end_n['fx_type']==1 else end_n['low']),
                'MACD力度': round(macd_sum, 2)
            })
    return bi_list[::-1]

# ==========================================
# 3. 界面交互
# ==========================================
code = st.text_input("输入股票代码", value="600519", placeholder="例如 600519")

if st.button("开始分析 🚀"):
    with st.spinner("正在连接国内数据源 (可能较慢)..."):
        try:
            df, source_name = get_stock_data(code)
            
            if df.empty:
                st.error("❌ 数据获取失败")
                st.write("可能原因：")
                st.write("1. 股票代码错误 (请输入6位数字)")
                st.write("2. 云服务器IP被国内拦截 (请查看上方黄色警告信息)")
            else:
                st.success(f"✅ 获取成功 ({source_name}): {code}")
                
                df = calculate_indicators(df)
                bi_data = calculate_bi(df)
                
                if bi_data:
                    last_bi = bi_data[0]
                    trend = last_bi['方向']
                    msg = f"当前 **{trend}笔** 延伸中 | 力度: {last_bi['MACD力度']}"
                    if trend == '向上': st.info(msg)
                    else: st.warning(msg)
                    
                    st.write("📋 **结构详情:**")
                    st.table(bi_data[:5])
                else:
                    st.warning("K线数量不足，无法生成笔结构")
                
                with st.expander("🔍 查看原始K线数据"):
                    st.dataframe(df.tail(10))
                    
        except Exception as e:
            st.error("程序运行出错")
            st.code(traceback.format_exc())
