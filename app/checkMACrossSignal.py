import tushare as ts
import pandas as pd
import numpy as np
import smtplib
import logging
from email.mime.text import MIMEText

# ================== 策略参数配置 ==================
# 参数设置
SHORT_MA = 5  # 短期均线周期
LONG_MA = 20  # 长期均线周期
SYMBOL = '603300.SH'  # 示例期货合约（海南华铁）
FREQ = '15MIN'  # 15分钟周期

# ================== 邮件配置 ==================
FROM_EMAIL = ""  #发送邮箱账号
TO_EMAIL = ""  #接收邮箱账号
SMTP_SERVER = 'smtp.qq.com'
SMTP_PORT = 465
EMAIL_PASSWORD = ""  #qq邮箱授权码

# ================== Tushare API 设置 ==================
ts.set_token("")  #你的TushareToken
pro = ts.pro_api()

# ================== 日志配置 ==================
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s',
	handlers=[
		logging.FileHandler('futures_strategy.log'),
		logging.StreamHandler()
	]
)


#测试数据
def generate_test_data(signal_type='golden', n=50, freq='15min'):
	"""
	生成模拟数据，最后一根K线触发金叉或死叉信号。

	参数:
	  signal_type: 'golden'（金叉）或 'death'（死叉）
	  n: 数据点数量，默认为50
	  freq: 时间间隔，默认为15分钟

	返回:
	  DataFrame 包含 time, open, close, high, low, amount, oi
	"""
	np.random.seed(42)

	# 获取当前时间并对齐到最近的15分钟
	current_time = pd.Timestamp.now()
	minutes_offset = current_time.minute % 15
	aligned_time = current_time - pd.Timedelta(minutes=minutes_offset, seconds=current_time.second)
	times = pd.date_range(start=aligned_time, periods=n, freq=freq)

	# 构造收盘价：
	# 如果需要金叉：前面大部分数据价格较低（例如10附近），最后5个数据跳升到较高水平（例如12附近）
	# 如果需要死叉：前面大部分数据价格较高（例如12附近），最后5个数据跌至较低水平（例如10附近）
	close_price = np.empty(n)
	if signal_type == 'golden':
		close_price[:n - 5] = 10 + np.random.normal(loc=0, scale=0.05, size=n - 5)
		close_price[n - 5:] = 12 + np.random.normal(loc=0, scale=0.05, size=5)
	elif signal_type == 'death':
		close_price[:n - 5] = 12 + np.random.normal(loc=0, scale=0.05, size=n - 5)
		close_price[n - 5:] = 10 + np.random.normal(loc=0, scale=0.05, size=5)
	else:
		raise ValueError("signal_type 只能为 'golden' 或 'death'")

	# 其他价格数据简单模拟
	open_price = close_price + np.random.normal(loc=0, scale=0.02, size=n)
	high_price = close_price + np.random.normal(loc=0.05, scale=0.02, size=n)
	low_price = close_price - np.random.normal(loc=0.05, scale=0.02, size=n)
	amount = np.random.normal(loc=10000, scale=500, size=n)
	oi = np.random.normal(loc=5000, scale=200, size=n)

	# 构建DataFrame
	df = pd.DataFrame({
		'time': times,
		'open': open_price,
		'close': close_price,
		'high': high_price,
		'low': low_price,
		'amount': amount,
		'oi': oi
	})

	return df


def get_tushare_futures_data(symbol, freq=FREQ, count=200):
	"""获取Tushare期货K线数据"""
	try:
		# 计算时间范围
		df = pro.rt_fut_min(ts_code=symbol, freq=freq)

		# 数据清洗
		df = df.rename(columns={
			'trade_time': 'time',
			'open': 'open',
			'high': 'high',
			'low': 'low',
			'close': 'close'
		})
		df['trade_time'] = pd.to_datetime(df['trade_time'])
		df = df.set_index('trade_time').sort_index()

		# 过滤最近count条数据
		return df.iloc[-count:] if len(df) > count else df

	except Exception as e:
		logging.error(f"Tushare数据获取失败: {str(e)}")
		return None


def calculate_ma(df):
	"""计算均线"""
	try:
		df = df.copy()
		df['Short_MA'] = df['close'].rolling(window=SHORT_MA, min_periods=1).mean()
		df['Long_MA'] = df['close'].rolling(window=LONG_MA, min_periods=1).mean()
		return df.dropna().iloc[-100:]  # 保留最近100个有效数据
	except KeyError:
		logging.error("数据缺少必要列：close")
		return None


def detect_cross(df):
	"""信号检测"""
	if df is None or len(df) < 2:
		logging.warning("数据不足无法检测信号")
		return None

	try:
		df['Signal'] = 0
		cross_up = (df['Short_MA'] > df['Long_MA']) & (df['Short_MA'].shift(1) <= df['Long_MA'].shift(1))
		cross_down = (df['Short_MA'] < df['Long_MA']) & (df['Short_MA'].shift(1) >= df['Long_MA'].shift(1))

		df.loc[cross_up, 'Signal'] = 1
		df.loc[cross_down, 'Signal'] = -1
		df = df.dropna(subset=['Signal'])  # 删除Signal为NaN的行

		return df[['time', 'open', 'high', 'low', 'close', 'Short_MA', 'Long_MA', 'Signal']]
	except Exception as e:
		logging.error(f"信号计算错误: {str(e)}")
		return None


def send_email_alert(df, signal_type, price, timestamp):
	"""发送预警邮件"""
	try:
		msg = MIMEText(
			f"期货合约：{SYMBOL}\n"
			f"信号时间：{timestamp.strftime('%Y-%m-%d %H:%M')}\n"
			f"信号类型：{signal_type}\n"
			f"当前价格：{price:.2f}\n"
			f"短期均线：{df.iloc[-1]['Short_MA']:.2f}\n"
			f"长期均线：{df.iloc[-1]['Long_MA']:.2f}",
			'plain'
		)
		msg['Subject'] = f"[期货信号] {SYMBOL} - {signal_type}"
		msg['From'] = FROM_EMAIL
		msg['To'] = TO_EMAIL

		with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
			server.login(FROM_EMAIL, EMAIL_PASSWORD)
			server.send_message(msg)
		logging.info("邮件发送成功")

	except Exception as e:
		logging.error(f"邮件发送失败: {str(e)}")


def strategy_main():
	"""策略主逻辑"""
	logging.info("=== 策略开始运行 ===")

	# 获取数据
	df = get_tushare_futures_data(SYMBOL, FREQ)
	if df is None or df.empty:
		logging.error("获取数据失败")
		return

	# 计算指标
	df = calculate_ma(df)
	if df is None:
		return

	# 检测信号
	df = detect_cross(df)
	if df is None:
		return

	# 获取最新信号
	latest = df.iloc[-1]

	if abs(latest['Signal']) > 0:  # 使用绝对值判断
		signal_type = "金叉" if latest['Signal'] > 0 else "死叉"
		print(latest.name)
		send_email_alert(df=df,
		                 signal_type=signal_type,
		                 price=latest['close'],
		                 timestamp=latest['time']
		                 )
	else:
		logging.info("未检测到有效信号")


if __name__ == "__main__":
	try:
		strategy_main()
	except Exception as e:
		logging.critical(f"主程序异常: {str(e)}", exc_info=True)
