import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import datetime
import traceback

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(
    page_title="AI缠论投喂系统 v6.0",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. 核心算法：通达信版 MACD (递归计算)
# ==========================================
def calculate_macd(df):
    """
    模拟通达信/同花顺的MACD计算公式
    """
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

# ==========================================
# 2. 基础力度计算 (同向红绿柱逻辑)
# ==========================================
def get_segment_metrics_by_date(raw_df, start_date, end_date, direction):
    mask = (raw_df['date'] >= start_date) & (raw_df['date'] <= end_date)
    segment_df = raw_df.loc[mask].copy()
    if segment_df.empty: return 0.0, 0.0, 0.0

    # 【规则】：上涨只算红柱，下跌只算绿柱
    if direction == '向上':
        # 只加红柱子 (macd > 0)
        macd_area = segment_df[segment_df['macd'] > 0]['macd'].sum()
        idx_price = segment_df['high'].idxmax()
    else:
        # 只加绿柱子 (macd < 0)，取绝对值
        macd_area = abs(segment_df[segment_df['macd'] < 0]['macd'].sum())
        idx_price = segment_df['low'].idxmin()
    
    peak_dif = segment_df.loc[idx_price, 'dif']
    avg_vol = segment_df['volume'].mean() / 10000 
    
    return round(macd_area, 4), round(peak_dif, 4), round(avg_vol, 2)

# ==========================================
# 3. 缠论K线包含处理 (视觉兼容版)
# ==========================================
def preprocess_inclusion(df):
    if len(df) < 2: return df
    raw_data = df.to_dict('records')
    # 初始化 real_date
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
            last['date'] = cur['date'] # 逻辑时间推移
            last['close'] = cur['close']
            
            # 视觉保护逻辑
            last_amp = (last['high'] - last['low']) / last['low'] if last['low'] > 0 else 0
            if last_amp > 0.015: 
                last['high'] = max(last['high'], cur['high'])
                last['low'] = min(last['low'], cur['low']) 
            else:
                if direction == 1:
                    last['high'] = max(last['high'], cur['high'])
                    last['low'] = max(last['low'], cur['low'])
                else:
                    last['high'] = min(last['high'], cur['high'])
                    last['low'] = min(last['low'], cur['low'])
            
            # 【重要】更新真实时间 (real_date)
            # 如果新K线创造了新的极值，则更新真实时间；否则保留原极值时间
            if direction == 1 and cur['high'] == last['high']: last['real_date'] = cur['real_date']
            elif direction == -1 and cur['low'] == last['low']: last['real_date'] = cur['real_date']
        else:
            if cur['high'] > last['high'] and cur['low'] > last['low']: direction = 1
            elif cur['high'] < last['high'] and cur['low'] < last['low']: direction = -1
            processed.append(cur)
            
    return pd.DataFrame(processed)

# ==========================================
# 4. 缠论分笔核心 (已修复MACD计算范围)
# ==========================================
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
        
        if k_df.loc[i, 'type'] == 0:
            if curr['high'] >= prev['high'] and curr['high'] >= next_['high'] and curr['high'] > max(prev['high'], next_['high']):
                 k_df.loc[i, 'type'] = 1
            elif curr['low'] <= prev['low'] and curr['low'] <= next_['low'] and curr['low'] < min(prev['low'], next_['low']):
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
            continue
            
        if curr.name - last.name >= 4:
            is_valid_space = False
            if last['type'] == 1 and curr['type'] == -1 and curr['low'] < last['low']: is_valid_space = True
            if last['type'] == -1 and curr['type'] == 1 and curr['high'] > last['high']: is_valid_space = True
            if (curr.name - last.name >= 9): is_valid_space = True
            if is_valid_space: stack.append(curr)
            
    bi_list = []
    if len(stack) < 2: return []
    
    for i in range(1, len(stack)):
        start_node = stack[i-1]
        end_node = stack[i]
        bi_dir = '向上' if start_node['type'] == -1 else '向下'
        
        # 【核心修复】：使用 real_date (真实极值时间) 截止，确保面积不被多算
        a, p, v = get_segment_metrics_by_date(df, start_node['real_date'], end_node['real_date'], bi_dir)
        
        bi_list.append({
            'start_date': start_node['date'], 'end_date': end_node['date'],
            'display_start_date': start_node['real_date'], 'display_end_date': end_node['real_date'],
            'start_price': float(start_node['low']) if start_node['type'] == -1 else float(start_node['high']),
            'end_price': float(end_node['low']) if end_node['type'] == -1 else float(end_node['high']),
            'direction': bi_dir, 'macd_area': a, 'peak_dif': p, 'avg_vol': v
        })
            
    return bi_list

# ==========================================
# 5. 智能量能分析 (同向红绿柱逻辑)
# ==========================================
def analyze_unformed_segment(df, last_bi):
    last_end_date = last_bi['end_date']
    unformed_df = df[df['date'] > last_end_date].copy()
    if len(unformed_df) == 0: return None
    
    last_dir = last_bi['direction']
    last_end_price = last_bi['end_price']
    
    current_dir = '向下' if last_dir == '向上' else '向上'
    status_note = "【正常回调】"
    
    if last_dir == '向上':
        if unformed_df['high'].max() > last_end_price:
            current_dir = '向上'; status_note = "【强势延续】"
    else: 
        if unformed_df['low'].min() < last_end_price:
            current_dir = '向下'; status_note = "【下跌中继】"

    physical_count = len(unformed_df)
    logical_df = preprocess_inclusion(unformed_df)
    logical_count = len(logical_df)
    current_avg_vol = unformed_df['volume'].mean() / 10000

    # 力度计算：同向逻辑
    if current_dir == '向上':
        macd_area = unformed_df[unformed_df['macd'] > 0]['macd'].sum()
        peak_dif = unformed_df.loc[unformed_df['high'].idxmax(), 'dif']
    else:
        macd_area = abs(unformed_df[unformed_df['macd'] < 0]['macd'].sum())
        peak_dif = unformed_df.loc[unformed_df['low'].idxmin(), 'dif']

    return {
        'count': physical_count, 'logical_count': logical_count,
        'high': unformed_df['high'].max(), 'low': unformed_df['low'].min(), 'close': unformed_df.iloc[-1]['close'],
        'macd_area': round(macd_area, 4), 'peak_dif': round(peak_dif, 4),
        'avg_vol': round(current_avg_vol, 2), 
        'current_dir': current_dir,
        'status': status_note
    }

# ==========================================
# 6. 数据获取
# ==========================================
@st.cache_data(ttl=3600) 
def get_stock_data(code, freq='d'):
    """ 从东方财富获取优质数据 (最近2年) """
    pure_code = code.split('.')[-1]
    try:
        start_date = (datetime.date.today() - datetime.timedelta(days=730)).strftime('%Y%m%d')
        end_date = datetime.date.today().strftime('%Y%m%d')

        if freq == 'd':
            df = ak.stock_zh_a_hist(symbol=pure_code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            if not df.empty:
                df = df.rename(columns={'日期':'date','开盘':'open','最高':'high','最低':'low','收盘':'close','成交量':'volume'})
        else:
            df = ak.stock_zh_a_hist_min_em(symbol=pure_code, period='30', adjust='qfq')
            if not df.empty:
                df = df.rename(columns={'时间':'date','开盘':'open','最高':'high','最低':'low','收盘':'close','成交量':'volume'})
            
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            cols = ['open','high','low','close','volume']
            for c in cols: df[c] = pd.to_numeric(df[c])
            df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"数据获取异常: {e}")
        return pd.DataFrame()

# ==========================================
# Main App Logic
# ==========================================
def main():
    st.title("🧙‍♂️ AI缠论投喂系统 v6.0 (最终修复版)")
    st.markdown("""
    **版本特性**: 
    1. **MACD精度修正**：面积计算精确对齐K线真实极值时间，消除包含处理带来的误差。
    2. **实战力度**：向上笔只算红柱，向下笔只算绿柱。
    3. **视觉兼容**：保护大阴大阳线不被算法吞没。
    """)
    
    with st.sidebar:
        st.header("参数设置")
        code = st.text_input("输入股票代码", value="600885", help="支持A股代码，如 600519")
        run_btn = st.button("开始分析", type="primary")

    if run_btn and code:
        try:
            with st.spinner(f'正在深入分析 {code} ...'):
                df_d = get_stock_data(code, 'd')
                if df_d.empty:
                    st.warning("未获取到数据，请检查代码。")
                    return
                df_d = calculate_macd(df_d)
                
                # 数据验钞机
                with st.expander("🔍 数据验钞机 (点击展开)"):
                    st.markdown("请核对最后3日的MACD数据：")
                    cols_to_show = ['date', 'close', 'dif', 'dea', 'macd']
                    st.dataframe(df_d.tail(3)[cols_to_show].style.format({
                        'close': '{:.2f}', 'dif': '{:.3f}', 'dea': '{:.3f}', 'macd': '{:.3f}'
                    }))
                
                # 计算结构
                bi_d = calculate_chanlun_structure(df_d)
                
                df_30 = get_stock_data(code, '30')
                bi_30 = []
                if not df_30.empty:
                    df_30 = calculate_macd(df_30)
                    bi_30 = calculate_chanlun_structure(df_30)
                
                # 生成提示词
                prompt = f"""
基于《AI缠论分析系统最高指令 v6.0》，数据源通达信对齐，MACD面积计算已修正为真实时间窗口。
**核心规则：**
1. **力度计算**：采用【同向柱体累计】。
   - 向上笔：仅累加MACD红柱面积。
   - 向下笔：仅累加MACD绿柱面积。
2. **画笔逻辑**：视觉兼容模式（逻辑K线视角）。

【分析标的】：{code}

=== 级别一：日线 (定方向) ===
数据范围：{df_d.iloc[0]['date'].date()} 至 {df_d.iloc[-1]['date'].date()}
【日线标准笔序列 (最后13笔)】
"""
                d_num = min(13, len(bi_d))
                if d_num > 0:
                    for i, bi in enumerate(bi_d[-d_num:]):
                        s_str = bi['display_start_date'].strftime('%Y-%m-%d')
                        e_str = bi['display_end_date'].strftime('%Y-%m-%d')
                        bi_idx = len(bi_d) - (d_num - 1) + i
                        area_desc = "红柱面积" if bi['direction'] == '向上' else "绿柱面积"
                        prompt += f"- 笔{bi_idx} [{bi['direction']}]: {s_str} -> {e_str} | 价:{bi['start_price']}->{bi['end_price']} | {area_desc}:{bi['macd_area']} | DIF极值:{bi['peak_dif']} | 均量:{bi['avg_vol']}万\n"
                
                if bi_d:
                    unf = analyze_unformed_segment(df_d, bi_d[-1])
                    if unf:
                        area_type = "红柱" if unf['current_dir'] == '向上' else "绿柱"
                        prompt += f"""
【日线当下状态 (未成笔段)】
- 运行: {unf['count']}天 (逻辑K线: {unf['logical_count']}根)
- 方向: {unf['current_dir']} ({unf['status']})
- 极值: 高{unf['high']} / 低{unf['low']} / 收{unf['close']}
- 力度: MACD{area_type}面积 {unf['macd_area']}, DIF极值 {unf['peak_dif']}
"""

                prompt += "\n=== 级别二：30分钟 (找买卖点) ===\n"
                if bi_30:
                    d30_num = min(13, len(bi_30))
                    prompt += f"【30分钟标准笔序列 (最后{d30_num}笔)】\n"
                    for i, bi in enumerate(bi_30[-d30_num:]):
                        s_str = bi['display_start_date'].strftime('%m-%d %H:%M')
                        e_str = bi['display_end_date'].strftime('%m-%d %H:%M')
                        bi_idx = len(bi_30) - (d30_num - 1) + i
                        area_desc = "红积" if bi['direction'] == '向上' else "绿积"
                        prompt += f"- 笔{bi_idx} [{bi['direction']}]: {s_str} -> {e_str} | 价:{bi['start_price']}->{bi['end_price']} | {area_desc}:{bi['macd_area']} | DIF极值:{bi['peak_dif']}\n"
                    
                    unf30 = analyze_unformed_segment(df_30, bi_30[-1])
                    if unf30:
                        prompt += f"""
【30分钟当下状态】
- 方向: {unf30['current_dir']}
- 结构: 物理{unf30['count']}根 / 逻辑{unf30['logical_count']}根
- 力度: MACD{("红积" if unf30['current_dir'] == '向上' else "绿积")} {unf30['macd_area']}, DIF极值 {unf30['peak_dif']}
"""
                else: prompt += "（数据不足）\n"
                
                prompt += """
【你的任务】
1. **背驰判断**：基于修正后的MACD面积（真实时间窗口）判断趋势衰竭。
2. **中枢精算**：严格输出 ZG/ZD 区间。
3. **策略**：结合30分钟买卖点提示。
"""
                st.success("分析完成！")
                st.code(prompt, language='text')
                
        except Exception:
            st.error("发生错误，请检查代码")
            st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
