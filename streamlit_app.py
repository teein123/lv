import streamlit as st
import yfinance as yf
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
st.title("📈 简易缠论分析 (雅虎源)")

# ==========================================
# 1. 核心数据获取 (雅虎财经版)
# ==========================================
@st.cache_data(ttl=600) 
def get_stock_data(code_input):
    # 1. 处理代码格式
    symbol = code_input.strip()
    
    # 雅虎财经规则：沪市加.SS，深市加.SZ
    # 简单的判断逻辑：6开头是沪市，0或3开头是深市
    if symbol.isdigit():
        if symbol.startswith('6'):
            symbol = symbol + ".SS"
        elif symbol.startswith('0') or symbol.startswith('3'):
            symbol = symbol + ".SZ"
        elif symbol.startswith('4') or symbol.startswith('8'):
            symbol = symbol + ".BJ" # 北交所
    
    # 2. 获取数据
    try:
        # 获取最近2年的数据
        stock = yf.Ticker(symbol)
        df = stock.history(period="2y")
        
        if df.empty: return pd.DataFrame()
        
        # 3. 数据清洗 (统一成你的算法需要的格式)
        df = df.reset_index()
        # 雅虎的列名是 Date, Open, High, Low, Close, Volume
        df = df.rename(columns={
            'Date': 'date', 'Open': 'open', 'Close': 'close', 
            'High': 'high', 'Low': 'low', 'Volume': 'volume'
        })
        
        # 移除时区信息，防止报错
        df['date'] = df['date'].dt.tz_localize(None)
        
        return df
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

# ==========================================
# 2. 缠论计算逻辑 (保持不变)
# ==========================================
def calculate_indicators(df):
    df = df.copy()
    # 确保数值类型
    cols = ['close', 'high', 'low']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['dif'] = df['ema12'] - df['ema26']
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['dif'] - df['dea']) * 2
    return df

def preprocess_inclusion(df):
    if len(df) < 2: return df
    data = df.to_dict('records')
    # 初始化
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
    # 包含处理
    k_df = preprocess_inclusion(df)
    k_df = k_df.reset_index(drop=True)
    if len(k_df) < 5: return []

    # 分型
    k_df['fx_type'] = 0 
    for i in range(1, len(k_df)-1):
        prev = k_df.iloc[i-1]; curr = k_df.iloc[i]; next_ = k_df.iloc[i+1]
        if curr['high'] > prev['high'] and curr['high'] > next_['high']:
            if curr['low'] > prev['low'] and curr['low'] > next_['low']: k_df.loc[i, 'fx_type'] = 1
        elif curr['low'] < prev['low'] and curr['low'] < next_['low']:
            if curr['high'] < prev['high'] and curr['high'] < next_['high']: k_df.loc[i, 'fx_type'] = -1

    fractals = k_df[k_df['fx_type'] != 0].copy()
    if len(fractals) < 2: return []

    # 笔生成
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
            # 计算力度
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
    with st.spinner("正在连接 Yahoo Finance (美国线路)..."):
        try:
            df = get_stock_data(code)
            
            if df.empty:
                st.error(f"❌ 获取失败: {code}")
                st.write("请检查代码是否正确。")
            else:
                st.success(f"✅ 获取成功 (Yahoo源): {code}")
                
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
