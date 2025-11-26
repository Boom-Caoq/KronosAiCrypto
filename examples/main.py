# main.py
# 完全保留你的原始预测逻辑，仅封装成函数
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import pandas as pd

# 👇 必须在导入 pyplot 前设置非 GUI 后端
import matplotlib
matplotlib.use('Agg')  # 防止多线程/GUI 错误
import matplotlib.pyplot as plt

from io import BytesIO
import os
from datetime import datetime
from flask import Flask, request, send_file

# -------------------------------
# 配置
# -------------------------------
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

BINANCE_URL = "https://fapi.binance.com/fapi/v1/klines"

# -------------------------------
# 1. 从 Binance 获取数据并保存为 CSV（带未来时间戳）
# -------------------------------
def fetch_kline_to_csv(symbol: str, interval: str, limit: int = 400, future_count: int = 20) -> str:
    """
    获取 K 线数据，并追加 future_count 个未来时间戳（NaN 值），用于预测
    """
    params = {
        'symbol': symbol.upper(),
        'interval': interval,
        'limit': limit
    }

    print(f"🔍 请求数据: {symbol} @ {interval}")
    response = requests.get(BINANCE_URL, params=params)
    response.raise_for_status()

    data = response.json()
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ]
    df = pd.DataFrame(data, columns=columns)

    # 转换时间戳
    df['timestamps'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']]
    df.rename(columns={'quote_asset_volume': 'amount'}, inplace=True)

    # 转换数值类型
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    df[numeric_cols] = df[numeric_cols].astype(float)

    # -------------------------------
    # 🔮 添加未来时间戳（空数据）
    # -------------------------------
    if future_count > 0:
        last_timestamp = df['timestamps'].iloc[-1]

        # 时间频率映射
        freq_map = {
            '1m': '1T', '3m': '3T', '5m': '5T', '15m': '15T', '30m': '30T',
            '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '8h': '8H', '12h': '12H',
            '1d': '1D', '3d': '3D', '1w': '1W', '1M': '1M'
        }
        freq = freq_map.get(interval, '1D')

        future_times = pd.date_range(
            start=last_timestamp + pd.Timedelta(freq),
            periods=future_count,
            freq=freq
        )

        # 创建未来行（除时间外全为 NaN）
        future_df = pd.DataFrame({'timestamps': future_times})
        for col in numeric_cols:
            future_df[col] = float('nan')

        # 拼接
        df = pd.concat([df, future_df], ignore_index=True)

    # 生成文件名
    timestamp = int(datetime.now().timestamp())
    filename = f"{symbol.upper()}_{interval}_{timestamp}.csv"
    filepath = os.path.join(DATA_DIR, filename)

    df.to_csv(filepath, index=False)
    print(f"✅ CSV 已保存: {filepath}")
    return filepath

# -------------------------------
# 2. 原始绘图函数（完全不变）
# -------------------------------
def plot_prediction(kline_df, pred_df, symbol, interval, pred_len=20):
    """
    绘制价格和成交量预测图，并添加标题说明
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False

    # 索引对齐
    pred_df.index = kline_df.index[-len(pred_df):]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = '实际价格'
    sr_pred_close.name = "预测价格"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = '实际成交量'
    sr_pred_volume.name = "预测成交量"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # 📈 价格图
    ax1.plot(close_df['实际价格'], label='实际价格', color='blue', linewidth=1.5)
    ax1.plot(close_df['预测价格'], label='预测价格', color='red', linewidth=1.5)
    ax1.set_ylabel('收盘价', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True)

    # 🔝 添加标题
    title = f"{symbol} ({interval}) 级别预测\n预测周期: {interval} 下的 {pred_len} 条K线"
    ax1.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # 📊 成交量图
    ax2.plot(volume_df['实际成交量'], label='实际成交量', color='blue', linewidth=1.5)
    ax2.plot(volume_df['预测成交量'], label='预测成交量', color='red', linewidth=1.5)
    ax2.set_ylabel('成交量', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    return fig

# -------------------------------
# 3. 核心预测函数（预测未来）
# -------------------------------
def run_prediction(csv_path: str, symbol: str, interval: str):
    from model import Kronos, KronosTokenizer, KronosPredictor

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

    df = pd.read_csv(csv_path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.reset_index(drop=True)

    lookback = 400
    pred_len = 20

    # 特征数据
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    x_df = df[feature_cols].iloc[:lookback].copy()

    # 时间戳
    x_timestamp = df['timestamps'].iloc[:lookback]
    y_timestamp = df['timestamps'].iloc[lookback:lookback + pred_len]

    # 预测
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True
    )

    print("🔮 预测结果：")
    print(pred_df)

    # 构造绘图数据
    kline_df = df.iloc[:lookback + pred_len].copy()
    kline_df.iloc[lookback:lookback + pred_len,
                 kline_df.columns.get_loc('open'):kline_df.columns.get_loc('amount')+1] = pred_df.values

    pred_df = pred_df.reset_index(drop=True)

    # 绘图（传入 symbol 和 interval）
    fig = plot_prediction(kline_df, pred_df, symbol=symbol, interval=interval, pred_len=pred_len)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)  # 关闭 figure，释放内存
    buf.seek(0)
    return buf

# -------------------------------
# 4. Flask 接口
# -------------------------------
app = Flask(__name__)

from flask_cors import CORS
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symbol = data.get('symbol')
    interval = data.get('interval')

    if not symbol or not interval:
        return {'error': '缺少 symbol 或 interval'}, 400

    try:
        csv_path = fetch_kline_to_csv(symbol, interval, limit=400, future_count=20)
        img_buf = run_prediction(csv_path, symbol=symbol, interval=interval)

        return send_file(
            img_buf,
            mimetype='image/png',
            as_attachment=False,
            download_name=f"{symbol}_{interval}_prediction.png"
        )
    except Exception as e:
        return {'error': str(e)}, 500

# -------------------------------
# 5. 测试入口
# -------------------------------
def test_run():
    print("🧪 开始测试...")
    try:
        symbol = "ETHUSDT"
        interval = "1m"
        csv_path = fetch_kline_to_csv(symbol, interval, limit=400, future_count=20)
        img_buf = run_prediction(csv_path, symbol=symbol, interval=interval)

        with open("test_result.png", "wb") as f:
            f.write(img_buf.getvalue())
        print("✅ 测试成功！图片已保存为 test_result.png")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

# -------------------------------
# 启动方式
# -------------------------------
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_run()
    else:
        print("🚀 启动服务: http://localhost:7890/predict")
        app.run(port=7890, debug=False, threaded=True)  # 推荐关闭 debug 并显式启用线程