import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import traceback

# ==========================================
# 0. 页面配置
# ==========================================
st.set_page_config(page_title="缠论分析工具", layout="mobile")
st.title("📈 简易缠论分析")

# ==========================================
# 1. 核心计算逻辑
# ==========================================
@st.cache_data(ttl=300) 
def get_stock_data(stock_code):
    # 处理代码格式，支持输入 600519 或 sh.600519
    code = stock_code.replace('sh.', '').replace('sz.', '')
    try:
        # 获取最近500天数据
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=500)).strftime('%Y%m%d')
        end_date = pd.Timestamp.now().strftime('%Y%m%d')
        
        df = ak.stock_zh_a_hist(symbol=code, start_date=start_date, end_date=end_date, adjust="qfq")
        if df.empty: return pd.DataFrame()
        
        df = df.rename(columns={
            '日期': 'date', '开盘': 'open', '收盘': 'close', 
            '最高': 'high', '最低': 'low', '成交量': 'volume'
        })
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        return pd.DataFrame()

def calculate_macd(df):
    df = df.copy()
    cols = ['close', 'high', 'low', 'volume']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # 精度修正
    df['high'] = df['high'].round(2)
    df['low'] = df['low'].round(2)
    df['close'] = df['close'].round(2)
    
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['dif'] = df['ema12'] - df['ema26']
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['dif'] - df['dea']) * 2
    return df

def get_segment_metrics_by_date(raw_df, start_date, end_date, direction):
    mask = (raw_df['date'] >= start_date) & (raw_df['date'] <= end_date)
    segment_df = raw_df.loc[mask].copy()
    if segment_df.empty: return 0.0, 0.0, 0.0

    if direction == '向上':
        macd_area = segment_df[segment_df['macd'] > 0]['macd'].sum()
        # idx_price_extreme = segment_df['high'].idxmax()
        # peak_dif = segment_df.loc[idx_price_extreme, 'dif']
        # 简化处理防止索引报错
        peak_dif = segment_df['dif'].max()
    else:
        macd_area = abs(segment_df[segment_df['macd'] < 0]['macd'].sum())
        peak_dif = segment_df['dif'].min()
    
    avg_vol = segment_df['volume'].mean() / 10000
    return round(macd_area, 4), round(peak_dif, 4), round(avg_vol, 2)

def preprocess_inclusion(df):
    # 简化版包含处理，确保速度
    if len(df) < 2: return df
    raw_data = df.to_dict('records')
    for d in raw_data: 
        if 'real_date' not in d: d['real_date'] = d['date']

    processed = [raw_data[0]]
    direction = 1 
    if raw_data[1]['close'] < raw_data[0]['close']: direction = -1

    for i in range(1, len(raw_data)):
        cur = raw_data[i]
        last = processed[-1]
        
        is_cur_inside = (cur['high'] <= last['high'] and cur['low'] >= last['low'])
        is_last_inside = (cur['high'] >= last['high'] and cur['low'] <= last['low'])
        
        if is_cur_inside or is_last_inside:
            # 发生包含，合并K线
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

def calculate_chanlun_structure(df):
    if len(df) < 10: return [] 

    # 1. 包含处理
    k_df = preprocess_inclusion(df)
    k_df = k_df.reset_index(drop=True)
    if len(k_df) < 5: return [] 

    # 2. 顶底分型
    k_df['type'] = 0 
    for i in range(1, len(k_df)-1):
        prev, curr, next_ = k_df.iloc[i-1], k_df.iloc[i], k_df.iloc[i+1]
        if curr['high'] > prev['high'] and curr['high'] > next_['high'] and curr['low'] > prev['low'] and curr['low'] > next_['low']:
            k_df.loc[i, 'type'] = 1  # 顶
        elif curr['low'] < prev['low'] and curr['low'] < next_['low'] and curr['high'] < prev['high'] and curr['high'] < next_['high']:
            k_df.loc[i, 'type'] = -1 # 底

    fractals = k_df[k_df['type'] != 0].copy()
    if len(fractals) < 2: return []
    
    # 3. 笔生成
    stack = [fractals.iloc[0]]
    bi_list = []
    
    for i in range(1, len(fractals)):
        curr = fractals.iloc[i]
        last = stack[-1]
        
        # 同向延伸（简单处理）
        if curr['type'] == last['type']:
            if (curr['type'] == 1 and curr['high'] > last['high']) or \
               (curr['type'] == -1 and curr['low'] < last['low']):
                stack.pop()
                stack.append(curr)
            continue
            
        # 反向成笔条件（简化：只要中间有K线即可）
        if curr.name - last.name >= 3:
            stack.append(curr)
            
            # 生成一笔的数据
            start_node = stack[-2]
            end_node = stack[-1]
            bi_dir = '向上' if start_node['type'] == -1 else '向下'
            
            # 计算MACD面积等
            try:
                a, p, v = get_segment_metrics_by_date(df, start_node['date'], end_node['date'], bi_dir)
            except:
                a, p, v = 0, 0, 0
            
            bi_list.append({
                '方向': bi_dir,
                '开始日期': start_node['real_date'].strftime('%Y-%m-%d'), 
                '结束日期': end_node['real_date'].strftime('%Y-%m-%d'),
                '开始价格': float(start_node['low']) if start_node['type'] == -1 else float(start_node['high']),
                '结束价格': float(end_node['low']) if end_node['type'] == -1 else float(end_node['high']),
                'MACD力度': a
            })
            
    return bi_list[::-1] # 倒序，把最新的放前面

# ==========================================
# 2. 界面交互
# ==========================================
code_input = st.text_input("输入股票代码", value="600519", placeholder="例如 600519")

if st.button("开始分析 🚀"):
    with st.spinner('正在获取数据...'):
        df = get_stock_data(code_input)
        
    if df.empty:
        st.error("❌ 未找到数据，请检查代码是否正确（如：600519）。")
    else:
        st.success(f"✅ 成功获取：{code_input}")
        
        # 计算MACD
        df = calculate_macd(df)
        
        # 计算笔
        try:
            bi_results = calculate_chanlun_structure(df)
            
            if bi_results:
                st.subheader("📋 笔结构分析 (最近优先)")
                st.dataframe(bi_results)
                
                # 简单的趋势判断
                last_bi = bi_results[0]
                if last_bi['方向'] == '向上':
                    st.info(f"当前处于 **向上笔** 延伸中，MACD力度: {last_bi['MACD力度']}")
                else:
                    st.warning(f"当前处于 **向下笔
