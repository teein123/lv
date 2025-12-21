import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import datetime
import traceback
import time
import warnings
import requests
import random
from requests.sessions import Session

# ==========================================
# 0. 页面配置与补丁 (放在最前面)
# ==========================================
st.set_page_config(
    page_title="AI缠论投喂系统 v6.3",
    page_icon="📈",
    layout="wide"
)

# 忽略警告
warnings.filterwarnings("ignore")

# --- 网络请求补丁 (保持原版逻辑) ---
_original_request = Session.request

def patched_request(self, method, url, *args, **kwargs):
    # 1. 强制清空代理
    kwargs['proxies'] = {"http": None, "https": None}
    # 2. 忽略 SSL
    kwargs['verify'] = False
    # 3. 强制添加伪装头
    if 'headers' not in kwargs:
        kwargs['headers'] = {}
    if not kwargs['headers'].get('User-Agent'):
        kwargs['headers']['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        kwargs['headers']['Referer'] = 'http://quote.eastmoney.com/'
        
    try:
        return _original_request(self, method, url, *args, **kwargs)
    except Exception as e:
        raise e

# 应用补丁
Session.request = patched_request

# ==========================================
# 1. 核心算法 (逻辑保持不变)
# ==========================================
def calculate_macd(df):
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

    if direction == '向上':
        macd_area = segment_df[segment_df['macd'] > 0]['macd'].sum()
        try:
            idx_price = segment_df['high'].idxmax()
            peak_dif = segment_df.loc[idx_price, 'dif']
        except:
            peak_dif = 0
    else:
        macd_area = abs(segment_df[segment_df['macd'] < 0]['macd'].sum())
        try:
            idx_price = segment_df['low'].idxmin()
            peak_dif = segment_df.loc[idx_price, 'dif']
        except:
            peak_dif = 0
    
    avg_vol = segment_df['volume'].mean() / 10000 if not segment_df.empty else 0
    
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
            
            if direction == 1 and cur['high'] == last['high']: last['real_date'] = cur['real_date']
            elif direction == -1 and cur['low'] == last['low']: last['real_date'] = cur['real_date']
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
        a, p, v = get_segment_metrics_by_date(df, start_node['real_date'], end_node['real_date'], bi_dir)
        
        bi_list.append({
            'start_date': start_node['date'], 'end_date': end_node['date'],
            'display_start_date': start_node['real_date'], 'display_end_date': end_node['real_date'],
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

    if current_dir == '向上':
        macd_area = unformed_df[unformed_df['macd'] > 0]['macd'].sum()
        try:
            peak_dif = unformed_df.loc[unformed_df['high'].idxmax(), 'dif']
        except: peak_dif = 0
    else:
        macd_area = abs(unformed_df[unformed_df['macd'] < 0]['macd'].sum())
        try:
            peak_dif = unformed_df.loc[unformed_df['low'].idxmin(), 'dif']
        except: peak_dif = 0

    return {
        'count': physical_count, 'logical_count': logical_count,
        'high': unformed_df['high'].max(), 'low': unformed_df['low'].min(), 'close': unformed_df.iloc[-1]['close'],
        'macd_area': round(macd_area, 4), 'peak_dif': round(peak_dif, 4),
        'avg_vol': round(current_avg_vol, 2), 
        'current_dir': current_dir,
        'status': status_note
    }

# ==========================================
# 2. 数据获取 (使用缓存)
# ==========================================
@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(code, freq='d'):
    """ 从东方财富获取优质数据，使用 Streamlit 缓存避免重复请求 """
    # 模拟随机延迟，虽然缓存了，但初次请求还是模拟一下好
    sleep_time = random.uniform(0.5, 1.0)
    time.sleep(sleep_time)
    
    pure_code = code.split('.')[-1]
    
    try:
        start_date = (datetime.date.today() - datetime.timedelta(days=730)).strftime('%Y%m%d')
        end_date = datetime.date.today().strftime('%Y%m%d')

        if freq == 'd':
            df = ak.stock_zh_a_hist(symbol=pure_code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            if not df.empty:
                df = df.rename(columns={'日期':'date','开盘':'open','最高':'high','最低':'low','收盘':'close','成交量':'volume'})
        elif freq == '30':
            df = ak.stock_zh_a_hist_min_em(symbol=pure_code, period='30', adjust='qfq')
            if not df.empty:
                df = df.rename(columns={'时间':'date','开盘':'open','最高':'high','最低':'low','收盘':'close','成交量':'volume'})
        elif freq == '5':
            df = ak.stock_zh_a_hist_min_em(symbol=pure_code, period='5', adjust='qfq')
            if not df.empty:
                df = df.rename(columns={'时间':'date','开盘':'open','最高':'high','最低':'low','收盘':'close','成交量':'volume'})
            
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            cols = ['open','high','low','close','volume']
            for c in cols: df[c] = pd.to_numeric(df[c])
            df = df.sort_values('date').reset_index(drop=True)
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        # st.error(f"{freq}线获取失败: {e}")
        return pd.DataFrame()

# ==========================================
# 3. Streamlit 主程序
# ==========================================
def main():
    st.title("AI缠论投喂系统 v6.3 (Web版)")
    st.markdown("### 特性: 强制直连去代理 | 日线/30分/5分 联立")
    
    with st.sidebar:
        st.header("设置")
        stock_code = st.text_input("请输入股票代码", value="600885", help="例如 600885, 不需要加后缀").strip()
        run_btn = st.button("开始分析", type="primary")
        
        st.info("说明：点击分析后，系统会获取数据并生成缠论结构 Prompt，可直接复制给 GPT/Claude 使用。")

    if run_btn and stock_code:
        pure_code = stock_code.split('.')[-1]
        
        status_container = st.status(f"正在分析 {pure_code}...", expanded=True)
        
        try:
            # --- 1. 日线处理 ---
            status_container.write("📥 正在下载日线数据...")
            df_d = get_stock_data(pure_code, 'd')
            
            if df_d.empty: 
                status_container.update(label="❌ 数据获取失败", state="error")
                st.error("无法获取日线数据，请检查代码是否正确或IP是否被限制。")
                return

            df_d = calculate_macd(df_d)
            
            # 验钞机展示
            st.subheader("🧐 数据验钞机 (最近3日)")
            tail_df = df_d.tail(3)[['date', 'close', 'dif', 'dea', 'macd']].copy()
            tail_df['date'] = tail_df['date'].dt.strftime('%Y-%m-%d')
            st.dataframe(tail_df, hide_index=True)
            
            status_container.write("🧮 计算日线缠论结构...")
            bi_d = calculate_chanlun_structure(df_d)
            
            # --- 2. 30分钟处理 ---
            status_container.write("📥 正在下载30分钟数据...")
            df_30 = get_stock_data(pure_code, '30')
            bi_30 = []
            if not df_30.empty:
                df_30 = calculate_macd(df_30)
                bi_30 = calculate_chanlun_structure(df_30)

            # --- 3. 5分钟处理 ---
            status_container.write("📥 正在下载5分钟数据...")
            df_5 = get_stock_data(pure_code, '5')
            bi_5 = []
            if not df_5.empty:
                df_5 = calculate_macd(df_5)
                bi_5 = calculate_chanlun_structure(df_5)
            
            status_container.update(label="✅ 分析完成", state="complete", expanded=False)

            # --- 4. 生成提示词 ---
            prompt = f"""
基于《AI缠论分析系统最高指令 v6.0》，数据源通达信对齐，MACD面积计算已修正为真实时间窗口。
**核心规则：**
1. **力度计算**：采用【同向柱体累计】。
2. **分析架构**：请执行【日线-30F-5F】三级联立的区间套分析。

【分析标的】：{stock_code}

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
                    area_desc = "红积" if bi['direction'] == '向上' else "绿积"
                    prompt += f"- 笔{bi_idx} [{bi['direction']}]: {s_str} -> {e_str} | 价:{bi['start_price']}->{bi['end_price']} | {area_desc}:{bi['macd_area']} | DIF极值:{bi['peak_dif']} | 均量:{bi['avg_vol']}万\n"
            
            if bi_d:
                unf = analyze_unformed_segment(df_d, bi_d[-1])
                if unf:
                    area_type = "红积" if unf['current_dir'] == '向上' else "绿积"
                    prompt += f"""
【日线当下状态 (未成笔段)】
- 运行: {unf['count']}天
- 方向: {unf['current_dir']} ({unf['status']})
- 极值: 高{unf['high']} / 低{unf['low']} / 收{unf['close']}
- 力度: MACD{area_type} {unf['macd_area']}, DIF极值 {unf['peak_dif']}
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
- 力度: MACD{("红积" if unf30['current_dir'] == '向上' else "绿积")} {unf30['macd_area']}, DIF极值 {unf30['peak_dif']}
"""
            
            prompt += "\n=== 级别三：5分钟 (精准狙击) ===\n"
            if bi_5:
                d5_num = min(20, len(bi_5))
                prompt += f"【5分钟标准笔序列 (最后{d5_num}笔)】\n"
                for i, bi in enumerate(bi_5[-d5_num:]):
                    s_str = bi['display_start_date'].strftime('%d日%H:%M')
                    e_str = bi['display_end_date'].strftime('%d日%H:%M')
                    bi_idx = len(bi_5) - (d5_num - 1) + i
                    area_desc = "红积" if bi['direction'] == '向上' else "绿积"
                    prompt += f"- 笔{bi_idx} [{bi['direction']}]: {s_str} -> {e_str} | 价:{bi['start_price']}->{bi['end_price']} | {area_desc}:{bi['macd_area']} | DIF极值:{bi['peak_dif']}\n"
                
                unf5 = analyze_unformed_segment(df_5, bi_5[-1])
                if unf5:
                    prompt += f"""
【5分钟当下状态】
- 方向: {unf5['current_dir']}
- 极值: 高{unf5['high']} / 低{unf5['low']} / 收{unf5['close']}
- 力度: MACD{("红积" if unf5['current_dir'] == '向上' else "绿积")} {unf5['macd_area']}, DIF极值 {unf5['peak_dif']}
"""
            else: prompt += "（5分钟数据不足）\n"

            prompt += """
【你的任务】
1. **区间套定位**：利用 5分钟数据解析 30分钟未完成段的内部结构。
2. **狙击点计算**：计算周一开盘的精确介入点（5F二买/类二买）及硬止损位。
3. **风控指令**：给出毫秒级止损条件。
"""
            st.subheader("📋 生成的 AI 提示词")
            st.code(prompt, language="markdown")
            
        except Exception: 
            status_container.update(label="❌ 发生错误", state="error")
            st.error(traceback.format_exc())

if __name__ == "__main__": 
    main()
