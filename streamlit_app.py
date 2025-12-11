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
    layout="mobile"
)
st.title("📈 简易缠论分析")

# ==========================================
# 1. 核心计算函数
# ==========================================
@st.cache_data(ttl=300) 
def get_stock_data(code_input):
    # 清洗代码格式
    symbol = code_input.replace('sh.', '').replace('sz.', '')
    try:
        # 获取最近300天数据，减少计算量加快速度
        start_dt = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime('%Y%m%d')
        end_dt = pd.Timestamp.now().strftime('%Y%m%d')
        
        df = ak.stock_zh_a_hist(
            symbol=symbol, 
            start_date=start_dt, 
            end_date=end_dt, 
            adjust="qfq"
        )
        
        if df.empty: return pd.DataFrame()
        
        # 重命名列
        df = df.rename(columns={
            '日期': 'date', '开盘': 'open', '收盘': 'close', 
            '最高': 'high', '最低': 'low', '成交量': 'volume'
        })
        df['date'] = pd.to_datetime(df['date'])
        return df
    except:
        return pd.DataFrame()

def calculate_indicators(df):
    # 计算MACD
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
    # K线包含处理
    if len(df) < 2: return df
    data = df.to_dict('records')
    
    # 初始化真实时间
    for d in data: 
        if 'real_date' not in d: d['real_date'] = d['date']

    processed = [data[0]]
    # 初始方向判断
    direction = 1 
    if data[1]['close'] < data[0]['close']: direction = -1

    for i in range(1, len(data)):
        cur = data[i]
        last = processed[-1]
        
        # 判断包含关系
        is_cur_in = (cur['high'] <= last['high'] and cur['low'] >= last['low'])
        is_last_in = (cur['high'] >= last['high'] and cur['low'] <= last['low'])
        
        if is_cur_in or is_last_in:
            # 合并处理
            last['date'] = cur['date']
            last['close'] = cur['close']
            last['volume'] = last['volume'] + cur['volume']
            
            if direction == 1: # 向上处理
                last['high'] = max(last['high'], cur['high'])
                last['low'] = max(last['low'], cur['low'])
            else: # 向下处理
                last['high'] = min(last['high'], cur['high'])
                last['low'] = min(last['low'], cur['low'])
        else:
            # 无包含，更新方向并添加新K线
            if cur['high'] > last['high'] and cur['low'] > last['low']: 
                direction = 1
            elif cur['high'] < last['high'] and cur['low'] < last['low']: 
                direction = -1
            processed.append(cur)
            
    return pd.DataFrame(processed)

def calculate_bi(df):
    if len(df) < 10: return []
    
    # 1. 处理包含
    k_df = preprocess_inclusion(df)
    k_df = k_df.reset_index(drop=True)
    if len(k_df) < 5: return []

    # 2. 找顶底分型
    k_df['fx_type'] = 0 # 1=顶, -1=底
    for i in range(1, len(k_df)-1):
        prev = k_df.iloc[i-1]
        curr = k_df.iloc[i]
        next_ = k_df.iloc[i+1]
        
        if curr['high'] > prev['high'] and curr['high'] > next_['high']:
            if curr['low'] > prev['low'] and curr['low'] > next_['low']:
                k_df.loc[i, 'fx_type'] = 1
        elif curr['low'] < prev['low'] and curr['low'] < next_['low']:
            if curr['high'] < prev['high'] and curr['high'] < next_['high']:
                k_df.loc[i, 'fx_type'] = -1

    fractals = k_df[k_df['fx_type'] != 0].copy()
    if len(fractals) < 2: return []

    # 3. 连成笔
    bi_list = []
    stack = [fractals.iloc[0]]
    
    for i in range(1, len(fractals)):
        curr = fractals.iloc[i]
        last = stack[-1]
        
        # 同向延续，更新极值
        if curr['fx_type'] == last['fx_type']:
            if curr['fx_type'] == 1 and curr['high'] > last['high']:
                stack.pop()
                stack.append(curr)
            elif curr['fx_type'] == -1 and curr['low'] < last['low']:
                stack.pop()
                stack.append(curr)
        # 反向成笔（简化判断：索引距离>3）
        elif curr.name - last.name >= 3:
            stack.append(curr)
            
            # 记录这一笔
            start_n = stack[-2]
            end_n = stack[-1]
            
            # 计算MACD面积
            sub_df = df[(df['date'] >= start_n['real_date']) & 
                        (df['date'] <= end_n['real_date'])]
            macd_sum = sub_df['macd'].abs().sum()
            
            bi_list.append({
                '方向': '向上' if start_n['fx_type'] == -1 else '向下',
                '日期': end_n['real_date'].strftime('%Y-%m-%d'),
                '价格': float(end_n['high'] if end_n['fx_type']==1 else end_n['low']),
                'MACD力度': round(macd_sum, 2)
            })

    return bi_list[::-1] # 倒序，最新的在最前

# ==========================================
# 2. 界面展示逻辑
# ==========================================
code = st.text_input("输入代码 (如 600519)", value="600519")

if st.button("开始分析"):
    with st.spinner("数据获取中..."):
        df = get_stock_data(code)
    
    if df.empty:
        st.error("❌ 获取失败，请检查代码或等待几秒重试")
    else:
        # 计算流程
        try:
            df = calculate_indicators(df)
            bi_data = calculate_bi(df)
            
            st.success(f"✅ 分析成功: {code}")
            
            if bi_data:
                # 1. 显示最新状态
                last_bi = bi_data[0]
                curr_dir = last_bi['方向']
                curr_pow = last_bi['MACD力度']
                
                # 拼接字符串（防止手机端报错）
                msg_title = f"当前处于: {curr_dir}笔 延伸中"
                msg_body = f"最近一笔MACD力度: {curr_pow}"
                
                if curr_dir == '向上':
                    st.info(f"{msg_title}\n\n{msg_body}")
                else:
                    st.warning(f"{msg_title}\n\n{msg_body}")
                
                # 2. 显示列表
                st.write("📋 **最近5笔结构:**")
                st.table(bi_data[:5])
                
            else:
                st.warning("数据不足，无法形成笔结构")
                
            # 3. 显示行情数据
            with st.expander("查看最近行情数据"):
                st.dataframe(df.tail(10))
                
        except Exception as e:
            st.error("计算过程发生错误")
            st.code(traceback.format_exc())
