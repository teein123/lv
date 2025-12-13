import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import datetime
import traceback

# ==========================================
# 核心算法区域 (保持不变，与终极版一致)
# ==========================================
@st.cache_data(ttl=3600)  # 增加缓存，1小时内重复查不耗流量
def get_stock_data(code, freq='d'):
    """ 从东方财富获取数据 """
    pure_code = code.split('.')[-1]
    try:
        if freq == 'd':
            start_date = "20200101" 
            end_date = datetime.date.today().strftime('%Y%m%d')
            df = ak.stock_zh_a_hist(symbol=pure_code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            df = df.rename(columns={'日期':'date','开盘':'open','最高':'high','最低':'low','收盘':'close','成交量':'volume'})
        else:
            df = ak.stock_zh_a_hist_min_em(symbol=pure_code, period='30', adjust='qfq')
            df = df.rename(columns={'时间':'date','开盘':'open','最高':'high','最低':'low','收盘':'close','成交量':'volume'})
            
        df['date'] = pd.to_datetime(df['date'])
        cols = ['open','high','low','close','volume']
        for c in cols: df[c] = pd.to_numeric(df[c])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        return pd.DataFrame()

def calculate_macd(df):
    """ 通达信算法复刻 """
    df = df.copy()
    close = df['close'].values
    def calc_ema_recursive(series, span):
        alpha = 2 / (span + 1)
        ema = np.zeros_like(series)
        ema[0] = series[0]
        for i in range(1, len(series)):
            ema[i] = alpha * series[i] + (1 - alpha) * ema[i-1]
        return ema

    df['ema12'] = calc_ema_recursive(close, 12)
    df['ema26'] = calc_ema_recursive(close, 26)
    df['dif'] = df['ema12'] - df['ema26']
    df['dea'] = calc_ema_recursive(df['dif'].values, 9)
    df['macd'] = (df['dif'] - df['dea']) * 2
    return df

def get_segment_metrics_by_date(raw_df, start_date, end_date, direction):
    mask = (raw_df['date'] >= start_date) & (raw_df['date'] <= end_date)
    segment_df = raw_df.loc[mask].copy()
    if segment_df.empty: return 0.0, 0.0, 0.0

    macd_area = segment_df['macd'].sum()
    if direction == '向上':
        idx_price = segment_df['high'].idxmax()
    else:
        macd_area = abs(macd_area)
        idx_price = segment_df['low'].idxmin()
    
    peak_dif = segment_df.loc[idx_price, 'dif']
    avg_vol = segment_df['volume'].mean() / 10000 
    return round(macd_area, 4), round(peak_dif, 4), round(avg_vol, 2)

def preprocess_inclusion(df):
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
            last['volume'] = float(last['volume']) + float(cur['volume'])
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
    k_df = preprocess_inclusion(df)
    k_df = k_df.reset_index(drop=True)
    if len(k_df) < 5: return [] 

    k_df['type'] = 0 
    for i in range(1, len(k_df)-1):
        prev, curr, next_ = k_df.iloc[i-1], k_df.iloc[i], k_df.iloc[i+1]
        if curr['high'] > prev['high'] and curr['high'] > next_['high'] and curr['low'] > prev['low'] and curr['low'] > next_['low']:
            k_df.loc[i, 'type'] = 1 
        elif curr['low'] < prev['low'] and curr['low'] < next_['low'] and curr['high'] < prev['high'] and curr['high'] < next_['high']:
            k_df.loc[i, 'type'] = -1 

    fractals = k_df[k_df['type'] != 0].copy()
    if len(fractals) < 2: return []
    
    stack = [fractals.iloc[0]]
    for i in range(1, len(fractals)):
        curr = fractals.iloc[i]
        last = stack[-1]
        if curr['type'] == last['type']:
            if curr['type'] == 1 and curr['high'] > last['high']:
                stack.pop(); stack.append(curr)
            elif curr['type'] == -1 and curr['low'] < last['low']:
                stack.pop(); stack.append(curr)
        else:
            if (curr.name - last.name >= 4): stack.append(curr)
            
    bi_list = []
    if len(stack) < 2: return []
    for i in range(1, len(stack)):
        start_node = stack[i-1]; end_node = stack[i]
        bi_dir = '向上' if start_node['type'] == -1 else '向下'
        a, p, v = get_segment_metrics_by_date(df, start_node['date'], end_node['date'], bi_dir)
        bi_list.append({
            'start_date': start_node['date'], 'end_date': end_node['date'],
            'start_price': float(start_node['low']) if start_node['type'] == -1 else float(start_node['high']),
            'end_price': float(end_node['low']) if end_node['type'] == -1 else float(end_node['high']),
            'direction': bi_dir, 'macd_area': a, 'peak_dif': p, 'avg_vol': v
        })
    return bi_list

def analyze_unformed_segment(df, last_bi):
    last_end_date = last_bi['end_date']
    unformed_df = df[df['date'] > last_end_date].copy()
    if len(unformed_df) == 0: return None
    
    last_dir = last_bi['direction']
    last_end_price = last_bi['end_price']
    current_dir = '向下' if last_dir == '向上' else '向上'
    if last_dir == '向上' and unformed_df['high'].max() > last_end_price: current_dir = '向上'
    if last_dir == '向下' and unformed_df['low'].min() < last_end_price: current_dir = '向下'

    raw_area = unformed_df['macd'].sum()
    if current_dir == '向上':
        macd_area = raw_area; peak_dif = unformed_df.loc[unformed_df['high'].idxmax(), 'dif']
    else:
        macd_area = abs(raw_area); peak_dif = unformed_df.loc[unformed_df['low'].idxmin(), 'dif']
        
    prev_mask = (df['date'] >= last_bi['start_date']) & (df['date'] <= last_bi['end_date'])
    prev_vol_mean = df.loc[prev_mask, 'volume'].mean() / 10000
    curr_vol_mean = unformed_df['volume'].mean() / 10000
    ratio = curr_vol_mean / prev_vol_mean if prev_vol_mean > 0 else 0

    return {
        'count': len(unformed_df), 'high': unformed_df['high'].max(),
        'low': unformed_df['low'].min(), 'close': unformed_df.iloc[-1]['close'],
        'macd_area': round(macd_area, 4), 'peak_dif': round(peak_dif, 4),
        'avg_vol': round(curr_vol_mean, 2), 'vol_ratio': round(ratio, 2), 'current_dir': current_dir
    }

# ==========================================
# 网页界面区域 (UI)
# ==========================================
st.set_page_config(page_title="AI缠论投喂系统", page_icon="📈")

st.title("📈 AI缠论数据生成器 (手机版)")
st.markdown("基于 `Akshare` + `通达信算法` | 净面积计算")

code = st.text_input("请输入股票代码 (例如 600885):", value="")

if st.button("开始分析", type="primary"):
    if not code:
        st.error("请输入代码！")
    else:
        with st.status("正在拉取数据...", expanded=True) as status:
            try:
                # 1. 日线处理
                st.write("📥 下载日线数据 (东方财富)...")
                df_d = get_stock_data(code, 'd')
                if df_d.empty:
                    st.error("日线数据获取失败，请检查代码是否正确。")
                    st.stop()
                df_d = calculate_macd(df_d)
                
                # 数据验钞机 (Web版)
                st.write("🔍 数据校验 (最后5日):")
                tail_df = df_d.tail(5)[['date', 'close', 'dif', 'dea', 'macd']].copy()
                tail_df['date'] = tail_df['date'].dt.strftime('%Y-%m-%d')
                st.dataframe(tail_df, hide_index=True)
                
                st.write("🧮 计算日线结构...")
                bi_d = calculate_chanlun_structure(df_d)

                # 2. 30分钟处理
                st.write("📥 下载30分钟数据...")
                df_30 = get_stock_data(code, '30')
                bi_30 = []
                if not df_30.empty:
                    df_30 = calculate_macd(df_30)
                    bi_30 = calculate_chanlun_structure(df_30)

                status.update(label="计算完成！", state="complete", expanded=False)

                # 3. 生成 Prompt
                prompt = f"""
基于《AI缠论分析系统最高指令 v2.2》，数据经人工核对，MACD算法已对齐通达信。
**分析规则：**
1. **MACD力度**：采用净面积逻辑（红绿抵扣），真实反映动能。
2. **DIF极值**：取极值点瞬时DIF。
3. **数据源**：东方财富QFQ，无复权偏差。

【分析标的】：{code}

=== 级别一：日线 (定方向) ===
数据范围：{df_d.iloc[0]['date'].date()} 至 {df_d.iloc[-1]['date'].date()} (最新: {df_d.iloc[-1]['close']})
【日线标准笔序列 (最后13笔)】
"""
                d_num = min(13, len(bi_d))
                if d_num > 0:
                    for i, bi in enumerate(bi_d[-d_num:]):
                        s_str = bi['start_date'].strftime('%Y-%m-%d')
                        e_str = bi['end_date'].strftime('%Y-%m-%d')
                        bi_idx = len(bi_d) - (d_num - 1) + i
                        prompt += f"- 笔{bi_idx} [{bi['direction']}]: {s_str} -> {e_str} | 价:{bi['start_price']}->{bi['end_price']} | 面积:{bi['macd_area']} | DIF极值:{bi['peak_dif']} | 均量:{bi['avg_vol']}万手\n"
                
                if bi_d:
                    unf = analyze_unformed_segment(df_d, bi_d[-1])
                    if unf:
                        prompt += f"""
【日线当下状态 (未成笔段)】
- 运行: {unf['count']}交易日
- 方向: {unf['current_dir']} (新高/新低判定)
- 极值: 高{unf['high']} / 低{unf['low']} / 收{unf['close']}
- 力度: MACD净面积 {unf['macd_area']}, DIF极值 {unf['peak_dif']}
- 量能: 均量{unf['avg_vol']}万手 (比值{unf['vol_ratio']})
"""

                prompt += "\n=== 级别二：30分钟 (找买卖点) ===\n"
                if bi_30:
                    d30_num = min(13, len(bi_30))
                    prompt += f"【30分钟标准笔序列 (最后{d30_num}笔)】\n"
                    for i, bi in enumerate(bi_30[-d30_num:]):
                        s_str = bi['start_date'].strftime('%m-%d %H:%M')
                        e_str = bi['end_date'].strftime('%m-%d %H:%M')
                        bi_idx = len(bi_30) - (d30_num - 1) + i
                        prompt += f"- 笔{bi_idx} [{bi['direction']}]: {s_str} -> {e_str} | 价:{bi['start_price']}->{bi['end_price']} | 面积:{bi['macd_area']} | DIF极值:{bi['peak_dif']}\n"
                    
                    unf30 = analyze_unformed_segment(df_30, bi_30[-1])
                    if unf30:
                        prompt += f"""
【30分钟当下状态】
- 方向: {unf30['current_dir']}
- 极值: 高{unf30['high']} / 低{unf30['low']} / 收{unf30['close']}
- 力度: MACD净面积 {unf30['macd_area']}, DIF极值 {unf30['peak_dif']}
"""
                else: prompt += "（30分钟数据不足）\n"
                
                prompt += """
【任务】
1. **日线定性**：基于修正后的MACD面积，重新判定趋势背驰情况。
2. **中枢精算**：严格使用Min(g)/Max(d)输出中枢区间。
3. **买卖点**：结合30分钟当下状态，给出明确的操作策略。
"""
                st.success("生成成功！请点击右上角复制按钮：")
                st.code(prompt, language="markdown")
                
            except Exception as e:
                st.error(f"发生错误: {str(e)}")
                st.code(traceback.format_exc())
