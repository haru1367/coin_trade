import pandas as pd
import requests
import talib as ta
import time
import jwt
import uuid
import json
from urllib.parse import urlencode
import pyupbit
import numpy as np
import hashlib
from urllib.parse import urlencode, unquote
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime
import pytz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# TensorFlow 로그 메시지 숨기기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

server_url = 'https://api.upbit.com'
special_coin_list = ['ADA','ALGO','BLUR','CELO', 'ELF', 'EOS', 'GRS', 'GRT', 'ICX', 'MANA', 'MINA', 'POL', 'SAND', 'SEI', 'STG', 'TRX']

#전일 대비 가격 상승률 상위 n개 종목을 가져오는 함수

def get_top_gainers_upbit(number):
    base_url = "https://api.upbit.com/v1"

    try:
        # Step 1: Get market data
        markets_response = requests.get(f"{base_url}/market/all")
        markets_response.raise_for_status()
        markets = markets_response.json()

        # Filter for KRW markets only and extract symbols
        krw_symbols = [market['market'].split('-')[1] for market in markets if market['market'].startswith('KRW-')]

        # Step 2: Get ticker data for all KRW markets
        krw_markets = [f"KRW-{symbol}" for symbol in krw_symbols]
        tickers_response = requests.get(f"{base_url}/ticker", params={"markets": ",".join(krw_markets)})
        tickers_response.raise_for_status()
        tickers = tickers_response.json()

        # Step 3: Calculate price change percentages and filter for >3% gainers
        gainers = []
        for ticker in tickers:
            market = ticker['market']
            symbol = market.split('-')[1]
            prev_close = ticker['prev_closing_price']  # 전일 종가
            current_price = ticker['trade_price']     # 현재가

            except_coin = ['USDC','USDT','BTC','ETH','BCH','AAVE','SOL','BSV','AVAX','EGLD']
            
            if prev_close > 0 and symbol not in except_coin:  # 유효한 가격만 계산
                change_percent = ((current_price - prev_close) / prev_close) * 100
            
                gainers.append({
                    'symbol': symbol,
                    'change_percent': change_percent,
                    'current_price': current_price
                })

        # Step 4: Sort by price change percentage in descending order
        gainers = sorted(gainers, key=lambda x: x['change_percent'], reverse=True)

        # Step 5: Return top 5 gainers
        return [gainer['symbol'] for gainer in gainers[:number]]

    except requests.exceptions.RequestException as e:
        print(f"Error while fetching data from Upbit API: {e}")
        return []
    
# JSON 파일에서 API 키 불러오기
def load_api_keys(config_file="upbit_config.json"):
    try:
        with open(config_file, "r") as file:
            keys = json.load(file)
            return keys["api_key"], keys["secret_key"]
    except Exception as e:
        print(f"Error loading API keys: {e}")
        return None, None
    
# 현재 업비트 계좌 정보를 불러오는 함수
def get_upbit_account_info(api_key, secret_key):
    base_url = "https://api.upbit.com/v1"

    # JWT 토큰 생성
    payload = {
        'access_key': api_key,
        'nonce': str(uuid.uuid4()),
    }
    jwt_token = jwt.encode(payload, secret_key, algorithm="HS256")
    authorization_token = f"Bearer {jwt_token}"
    
    headers = {
        "Authorization": authorization_token,
    }

    # 계좌 조회 API 호출
    try:
        account_res = requests.get(f"{base_url}/accounts", headers=headers)
        account_data = account_res.json()
        if account_res.status_code != 200:
            print(f"Error fetching account data: {account_data}")
            return

        # 코인별 정보 조회
        market_data = requests.get(f"{base_url}/market/all").json()
        market_dict = {item['market']: item['korean_name'] for item in market_data}

        total_balance = 0
        coin_info_list = []

        for account in account_data:
            if float(account['balance']) > 0:  # 잔액이 0보다 클 경우
                ticker = account['currency']
                if ticker == "KRW":
                    total_balance += float(account['balance'])
                    continue
                
                market = f"KRW-{ticker}"
                avg_buy_price = float(account['avg_buy_price'])
                balance = float(account['balance'])

                # 현재 가격 조회
                ticker_url = f"{base_url}/ticker?markets={market}"
                ticker_res = requests.get(ticker_url).json()
                current_price = float(ticker_res[0]['trade_price'])
                
                # 평가 금액과 수익률 계산
                eval_amount = balance * current_price
                profit_rate = ((current_price - avg_buy_price) / avg_buy_price) * 100
                
                # 총 자산에 반영
                total_balance += eval_amount
                
                coin_info_list.append({
                    "코인": market_dict.get(market, ticker),
                    "보유수량": balance,
                    "평균단가": avg_buy_price,
                    "현재가격": current_price,
                    "수익률": profit_rate,
                })

        # 출력
        print(f"총 자산: {total_balance:,.0f} KRW")
        print('')
        print("보유 코인 정보:")
        print(coin_info_list)
        return coin_info_list,total_balance
    except Exception as e:
        print(f"An error occurred: {e}")

# 보유중인 코인의 심볼과 수량을 가져오는 함수
def get_coin_holdings(api_key, secret_key):
    """
    업비트 API를 사용하여 보유 중인 코인의 종류와 수량을 가져옵니다.

    Returns:
        dict: 코인의 종류와 보유 수량을 포함한 딕셔너리
    """
    base_url = "https://api.upbit.com/v1/accounts"

    # JWT 토큰 생성
    payload = {
        'access_key': api_key,
        'nonce': str(uuid.uuid4()),
    }
    jwt_token = jwt.encode(payload, secret_key, algorithm="HS256")
    headers = {"Authorization": f"Bearer {jwt_token}"}

    try:
        # 계좌 정보 요청
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
        accounts = response.json()

        # 보유 코인 정보 필터링
        holdings = {account['currency']: float(account['balance']) for account in accounts if float(account['balance']) > 0}

        # 보유 코인이 없을 경우 처리
        if not holdings:
            print("보유 중인 코인이 없습니다.")
            return {}

        return holdings

    except Exception as e:
        print(f"Error fetching coin holdings: {e}")
        return {}

#미체결된 주문 조회 함수
def check_unfilled_orders(api_key, secret_key):
    """
    미체결 주문이 있는지 확인합니다.

    Returns:
        list: 미체결 주문 리스트 (없으면 빈 리스트)
    """
    base_url = "https://api.upbit.com/v1/orders"
    query = {
        'state': 'wait',  # 'wait' 상태는 미체결 주문을 의미
    }

    # Query Hash 계산
    query_string = '&'.join([f"{key}={value}" for key, value in query.items()])
    query_hash = hashlib.sha512(query_string.encode()).hexdigest()

    # JWT 토큰 생성
    payload = {
        'access_key': api_key,
        'nonce': str(uuid.uuid4()),
        'query_hash': query_hash,
        'query_hash_alg': 'SHA512',
    }
    jwt_token = jwt.encode(payload, secret_key, algorithm="HS256")
    headers = {"Authorization": f"Bearer {jwt_token}"}

    try:
        # 미체결 주문 요청
        response = requests.get(base_url, headers=headers, params=query)
        response.raise_for_status()
        orders = response.json()

        # 미체결 주문이 없는 경우 처리
        if not orders:
            print("미체결 주문이 없습니다.")
            return []

        print("미체결 주문이 있습니다.")
        return orders

    except Exception as e:
        print(f"Error fetching unfilled orders: {e}")
        return []
    
#미체결된 주문 취소하는 함수
def cancel_unfilled_order(api_key, secret_key, order_id):
    """
    특정 미체결 주문을 취소합니다.

    Args:
        api_key (str): 업비트 API 키
        secret_key (str): 업비트 Secret 키
        order_id (str): 취소할 주문의 UUID

    Returns:
        dict: 주문 취소 결과
    """
    base_url = "https://api.upbit.com/v1/order"
    query = {
        'uuid': order_id,
    }

    # JWT 토큰 생성
    query_string = '&'.join([f"{key}={value}" for key, value in query.items()])
    payload = {
        'access_key': api_key,
        'nonce': str(uuid.uuid4()),
        'query_hash': hashlib.sha512(query_string.encode()).hexdigest(),
        'query_hash_alg': 'SHA512',
    }
    jwt_token = jwt.encode(payload, secret_key, algorithm="HS256")
    headers = {"Authorization": f"Bearer {jwt_token}"}

    try:
        # 주문 취소 요청
        response = requests.delete(base_url, headers=headers, params=query)
        response.raise_for_status()
        cancel_result = response.json()

        print(f"주문이 성공적으로 취소되었습니다: {cancel_result}")
        return cancel_result

    except requests.exceptions.HTTPError as e:
        print(f"HTTP 에러 발생: {e.response.status_code}, {e.response.text}")
        return {}
    except Exception as e:
        print(f"Error canceling order: {e}")
        return {}
    
#매수 1호가와 매도1호가 리턴하는 함수    
def get_orderbook(api_key, market):
    """
    주어진 마켓의 매수, 매도 1호가를 가져옵니다.

    Args:
        api_key (str): 업비트 API 키
        market (str): 마켓 코드 (예: "KRW-BTC")

    Returns:
        dict: 매수 1호가와 매도 1호가를 포함한 딕셔너리
    """
    url = f"https://api.upbit.com/v1/orderbook?markets={market}&level=0"

    headers = {"accept": "application/json"}

    try:
        # 주문서 정보 요청
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        orderbook = response.json()

        if not orderbook:
            print(f"{market}의 주문서 정보가 없습니다.")
            return None

        # 매수 1호가 (가장 높은 매도 호가)
        bid_price = orderbook[0]['orderbook_units'][0]['bid_price']
        # 매도 1호가 (가장 낮은 매수 호가)
        ask_price = orderbook[0]['orderbook_units'][0]['ask_price']

        return {"bid_price": bid_price, "ask_price": ask_price}

    except Exception as e:
        print(f"Error fetching orderbook: {e}")
        return None
    
#지정가 매수함수    
def place_limit_buy_order(api_key, secret_key, market, price, budget):
    max_quantity = str(budget / price)
    params = {
        'market': market,
        'side': 'bid',
        'ord_type': 'limit',
        'price': price,
        'volume': max_quantity
    }
    query_string = unquote(urlencode(params, doseq=True)).encode("utf-8")

    m = hashlib.sha512()
    m.update(query_string)
    query_hash = m.hexdigest()

    payload = {
        'access_key': api_key,
        'nonce': str(uuid.uuid4()),
        'query_hash': query_hash,
        'query_hash_alg': 'SHA512',
    }

    jwt_token = jwt.encode(payload, secret_key)
    authorization = 'Bearer {}'.format(jwt_token)
    headers = {
    'Authorization': authorization,
    }

    res = requests.post(server_url + '/v1/orders', json=params, headers=headers)
    order_result = res.json()
    # 성공적으로 주문을 제출했다면 주문 UUID 반환
    order_uuid = order_result.get('uuid')
    if order_uuid:
        print(f"매수 주문이 성공적으로 생성되었습니다. 주문 UUID: {order_uuid}")
        time.sleep(5)
        return order_uuid
    else:
        print("주문 생성 실패: UUID를 찾을 수 없습니다.")
        return None
    
# 지정가 매도함수
def place_limit_sell_order(api_key, secret_key, market, price, quantity):
    max_quantity = str(quantity)
    params = {
        'market': market,
        'side': 'ask',
        'ord_type': 'limit',
        'price': price,
        'volume': max_quantity
    }
    query_string = unquote(urlencode(params, doseq=True)).encode("utf-8")

    m = hashlib.sha512()
    m.update(query_string)
    query_hash = m.hexdigest()

    payload = {
        'access_key': api_key,
        'nonce': str(uuid.uuid4()),
        'query_hash': query_hash,
        'query_hash_alg': 'SHA512',
    }

    jwt_token = jwt.encode(payload, secret_key)
    authorization = 'Bearer {}'.format(jwt_token)
    headers = {
    'Authorization': authorization,
    }

    res = requests.post(server_url + '/v1/orders', json=params, headers=headers)
    order_result = res.json()
    # 성공적으로 주문을 제출했다면 주문 UUID 반환
    order_uuid = order_result.get('uuid')
    if order_uuid:
        print(f"매도 주문이 성공적으로 생성되었습니다. 주문 UUID: {order_uuid}")
        time.sleep(5)
        return order_uuid
    else:
        print("주문 생성 실패: UUID를 찾을 수 없습니다.")
        return None
    
#코인의 현재가격을 불러오는 함수
def get_current_price(coin):
    url = f"https://api.upbit.com/v1/ticker"
    params = {
        'markets': f'KRW-{coin}'  # 코인 이름 입력 (예: 'BTC' -> 'KRW-BTC')
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if len(data) > 0:
        return data[0]['trade_price']  # 현재 거래 가격 반환
    else:
        return None
    
#코인의 캔들데이터를 가져오는 함수
def get_coin_data(coin):
    url = "https://api.upbit.com/v1/candles/minutes/1"  # Adjusted to minutes/1 endpoint for real data
    params = {  
        'market': f'KRW-{coin}',  
        'count': 200,
    }  
    headers = {"accept": "application/json"}

    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    df = pd.DataFrame(data)
    df = df.reindex(index=df.index[::-1]).reset_index(drop=True)  # Reset index after reversing
    return df

#받아온 코인 데이터를 저장할 csv파일을 만드는 함수수
def save_to_csv(df):
    filename = 'coin_data.csv'
    df.to_csv(filename, index=False)

#거래기록을 저장할 csv 파일일
columns = ['시간', '코인명', '매수/매도', '평균 단가','현재 보유잔고']
df_trade = pd.DataFrame(columns=columns)
df_trade.to_csv('trade_list.csv', index=False, encoding='utf-8-sig')

#거래기록을 추가하는 함수
def add_trade(trade_time, coin_name, trade_type, avg_price,hold_won):
    # CSV 파일 읽기
    df = pd.read_csv('trade_list.csv', encoding='utf-8-sig')
    
    # 새로운 데이터 추가
    new_data = {
        '시간': trade_time,
        '코인명': coin_name,
        '매수/매도': trade_type,
        '평균 단가': avg_price,
        '현재 보유잔고':hold_won
    }
    df = df.append(new_data, ignore_index=True)
    
    # 업데이트된 데이터프레임을 CSV 파일로 저장
    df.to_csv('trade_list.csv', index=False, encoding='utf-8-sig')

#현재 보유중인 원화 잔고를 가져오는 함수
def get_balance():
    """
    업비트 API를 사용하여 원화 잔고를 조회하는 함수
    """
    url = server_url + '/v1/accounts'
    
    # 요청 헤더 준비
    payload = {
        'access_key': api_key,
        'nonce': str(uuid.uuid4())
    }
    
    # 서명 생성
    m = hashlib.sha512()
    m.update(payload['nonce'].encode('utf-8'))
    signature = jwt.encode(payload, secret_key, algorithm='HS256')
    
    headers = {
        'Authorization': f'Bearer {signature}'
    }
    
    # API 요청
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        accounts = response.json()
        for account in accounts:
            if account['currency'] == 'KRW':  # 원화 잔고 찾기
                return float(account['balance'])
    else:
        print(f"Error: {response.status_code}")
        return None
    
#각종 지표를 계산하는 함수
def calculate_indicators(df):
    # 'trade_price' (종가), 'high_price', 'low_price' 컬럼 확인 및 추출
    required_columns = ['trade_price', 'high_price', 'low_price','candle_acc_trade_volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"데이터에 '{col}' 컬럼이 없습니다.")
    
    close_prices = df['trade_price'].astype(float)
    high_prices = df['high_price'].astype(float)
    low_prices = df['low_price'].astype(float)
    volumes = df['candle_acc_trade_volume'].astype(float)

    # 데이터 확인
    if close_prices.isnull().any():
        raise ValueError("종가 데이터에 결측값이 있습니다.")
    if len(close_prices) < 10:  # Momentum 계산에 필요한 최소 데이터 길이
        raise ValueError("Momentum 지표를 계산하려면 데이터 길이가 최소 10 이상이어야 합니다.")
    
    # RSI (Relative Strength Index)
    df['RSI'] = ta.RSI(close_prices, timeperiod=14)
    rsi = df['RSI'].iloc[-1]

    # Stochastic RSI
    df['StochRSI'] = (df['RSI'] - df['RSI'].rolling(14).min()) / (df['RSI'].rolling(14).max() - df['RSI'].rolling(14).min()) * 100
    df['StochRSI_K'] = df['StochRSI'].rolling(3).mean()
    df['StochRSI_D'] = df['StochRSI_K'].rolling(3).mean()
    stoch_rsi_d = df['StochRSI_D'].iloc[-1]
    stoch_rsi_k = df['StochRSI_K'].iloc[-1]
    
    # Bollinger Bands
    df['Upper_Band'], df['Middle_Band'], df['Lower_Band'] = ta.BBANDS(
        close_prices,
        timeperiod=20,
        nbdevup=2.0,
        nbdevdn=2.0,
        matype=0
    )
    upper_band = df['Upper_Band'].iloc[-1]
    middle_band = df['Middle_Band'].iloc[-1]
    lower_band = df['Lower_Band'].iloc[-1]
    
    # MACD (Moving Average Convergence Divergence)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = ta.MACD(
        close_prices,
        fastperiod=12,
        slowperiod=26,
        signalperiod=9
    )
    macd = df['MACD'].iloc[-1]
    macd_signal = df['MACD_Signal'].iloc[-1]
    macd_hist = df['MACD_Hist'].iloc[-1]
    
    # TRIX (Triple Exponential Average)
    df['TRIX'] = ta.TRIX(
        close_prices,
        timeperiod=5
    )
    trix = df['TRIX'].iloc[-1]

    df['MFI'] = ta.MFI(
        high=high_prices,
        low=low_prices,
        close=close_prices,
        volume=volumes,
        timeperiod=14
    )
    mfi = df['MFI'].iloc[-1]

    df['Tenkan_sen'] = (high_prices.rolling(window=9).max() + low_prices.rolling(window=9).min()) / 2
    df['Kijun_sen'] = (high_prices.rolling(window=26).max() + low_prices.rolling(window=26).min()) / 2
    tenkan_sen = df['Tenkan_sen'].iloc[-1]  # 전환선 최신 값
    kijun_sen = df['Kijun_sen'].iloc[-1]    # 기준선 최신 값

    # Save data to CSV
    save_to_csv(df)
    
    # Return all indicator values
    return (
        upper_band, middle_band, lower_band, macd, macd_signal, macd_hist, trix, mfi, rsi, stoch_rsi_d,stoch_rsi_k,tenkan_sen,kijun_sen
    )

def preprocess_data(df):
    # 필요한 지표만 추출
    features = [
        'Upper_Band', 'Middle_Band', 'Lower_Band', 'MACD', 'MACD_Signal', 
        'MACD_Hist', 'TRIX', 'MFI', 'RSI', 'StochRSI_D', 'StochRSI_K', 'Tenkan_sen', 'Kijun_sen'
    ]

    # 결측값 제거
    df = df.dropna(subset=features).copy()  # .copy()를 추가하여 경고 제거

    # 목표 변수 설정 (다음 가격이 상승/하락)
    df['Target'] = (df['trade_price'].shift(-1) > df['trade_price']).astype(int)

    # 결측값 제거 (Target 열에 결측값이 생길 수 있으므로 제거)
    df = df.dropna(subset=['Target']).copy()  # .copy() 추가

    # 입력 (X)와 출력 (y) 분리
    X = df[features].values
    y = df['Target'].values

    # 데이터 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

# 모델 생성 함수
def create_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax')  # 확률 출력
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model


# 학습 및 예측 함수
def train_and_predict(df):
    # 데이터 전처리
    X, y, scaler = preprocess_data(df)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 모델 생성
    model = create_model(input_dim=X.shape[1])

    # 모델 학습
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),verbose=0)

    # 예측 수행
    last_sample = X[-1].reshape(1, -1)  # 마지막 샘플 예측
    probabilities = model.predict(last_sample)[0]

    return {
        'probability_up': probabilities[1],
        'probability_down': probabilities[0]
    }


api_key, secret_key = load_api_keys("upbit_config.json")
# coin_list = input("조회할 코인 심볼들을 입력하세요 (쉼표로 구분): ").upper().replace(" ", "").split(',')

while True:
    coin_holdings = get_coin_holdings(api_key, secret_key)
    money = int(get_balance() * 0.95)
    if len(coin_holdings)==1:
        coin_list = get_top_gainers_upbit(7)
        for coin in coin_list:
            market = f'KRW-{coin}'
            status_coin_price = get_current_price(coin)
            coin_order_unit = 0
            if status_coin_price >= 2000000:
                coin_order_unit = 1000
            elif status_coin_price >=1000000 and status_coin_price<2000000:
                coin_order_unit = 500
            elif status_coin_price >=500000 and status_coin_price <1000000:
                coin_order_unit = 100
            elif status_coin_price >=100000 and status_coin_price<500000:
                coin_order_unit = 50
            elif status_coin_price >=10000 and status_coin_price<100000:
                coin_order_unit = 10
            elif status_coin_price >=1000 and status_coin_price<10000:
                coin_order_unit = 1
            elif status_coin_price >=100 and status_coin_price<1000:
                coin_order_unit = 0.1
                if coin in special_coin_list:
                    coin_order_unit = 1
            elif status_coin_price >=10 and status_coin_price<100:
                coin_order_unit = 0.01
            elif status_coin_price >= 1 and status_coin_price<10:
                coin_order_unit = 0.001
            elif status_coin_price>=0.1 and status_coin_price<1:
                coin_order_unit = 0.0001
            elif status_coin_price>=0.01 and status_coin_price<0.1:
                coin_order_unit = 0.00001
            elif status_coin_price>=0.001 and status_coin_price<0.01:
                coin_order_unit = 0.000001
            elif status_coin_price>=0.0001 and status_coin_price<0.001 :
                coin_order_unit = 0.0000001
            else :
                coin_order_unit = 0.00000001
            df = get_coin_data(coin)  # 데이터 가져오기
            # df = df.iloc[:-1]
            indicators = calculate_indicators(df)  # 지표 계산
            prediction = train_and_predict(df)  # 상승/하락 확률 계산
            
            print('----------------------------------------------')
            print(f'{coin}의 현재 가격 : {status_coin_price}원')
            print('')
            print(f'현재 {coin}코인의 상승확률 : {prediction["probability_up"]*100:.3f}%')

            while prediction['probability_up']>0.85:
                orderbook = get_orderbook(api_key, market)
                buy_price = orderbook['bid_price'] + coin_order_unit
                buy_uuid = place_limit_buy_order(api_key,secret_key,market,buy_price,money)
                unfilled_orders = check_unfilled_orders(api_key, secret_key)
                if unfilled_orders:
                    print('지정가 매수 실패, 미체결 주문 취소 작업 중..')
                    for order in unfilled_orders:
                        cancel_result = cancel_unfilled_order(api_key, secret_key, buy_uuid)
                else:
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    my_money_info,total_balance = get_upbit_account_info(api_key,secret_key)
                    add_trade(current_time, f'{coin}', 'buy', buy_price,total_balance)
                    break
                df = get_coin_data(coin)  # 데이터 가져오기
                # df = df.iloc[:-1]
                indicators = calculate_indicators(df)  # 지표 계산
                prediction = train_and_predict(df)  # 상승/하락 확률 계산
            if len(get_coin_holdings(api_key,secret_key))>1:
                break
            print('')
        
    else:
        print('----------------------------------------------')
        hold_coin = list(coin_holdings.keys())[1]
        hold_coin_amount = float(coin_holdings[hold_coin])
        status_coin_price = get_current_price(hold_coin)
        coin_order_unit = 0
        market = f'KRW-{hold_coin}'
        if status_coin_price >= 2000000:
            coin_order_unit = 1000
        elif status_coin_price >=1000000 and status_coin_price<2000000:
            coin_order_unit = 500
        elif status_coin_price >=500000 and status_coin_price <1000000:
            coin_order_unit = 100
        elif status_coin_price >=100000 and status_coin_price<500000:
            coin_order_unit = 50
        elif status_coin_price >=10000 and status_coin_price<100000:
            coin_order_unit = 10
        elif status_coin_price >=1000 and status_coin_price<10000:
            coin_order_unit = 1
        elif status_coin_price >=100 and status_coin_price<1000:
            coin_order_unit = 0.1
            if hold_coin in special_coin_list:
                coin_order_unit = 1
        elif status_coin_price >=10 and status_coin_price<100:
            coin_order_unit = 0.01
        elif status_coin_price >= 1 and status_coin_price<10:
            coin_order_unit = 0.001
        elif status_coin_price>=0.1 and status_coin_price<1:
            coin_order_unit = 0.0001
        elif status_coin_price>=0.01 and status_coin_price<0.1:
            coin_order_unit = 0.00001
        elif status_coin_price>=0.001 and status_coin_price<0.01:
            coin_order_unit = 0.000001
        elif status_coin_price>=0.0001 and status_coin_price<0.001 :
            coin_order_unit = 0.0000001
        else :
            coin_order_unit = 0.00000001
        df = get_coin_data(hold_coin)  # 데이터 가져오기
        # df = df.iloc[:-1]
        indicators = calculate_indicators(df)  # 지표 계산
        prediction = train_and_predict(df)  # 상승/하락 확률 계산
    
        print(f'{hold_coin}의 현재 가격 : {status_coin_price}원')
        print('')
        print(f'현재 코인의 상승확률 : {prediction["probability_up"]*100:.3f}')
        my_money_info,total_balance = get_upbit_account_info(api_key,secret_key)
        while my_money_info[0]['수익률']<=-1.5:
            orderbook = get_orderbook(api_key, market)
            status_profit_ratio = get_upbit_account_info(api_key,secret_key)
            sell_price = orderbook['ask_price']-coin_order_unit
            sell_uuid = place_limit_sell_order(api_key,secret_key,market,sell_price,hold_coin_amount)
            if len(check_unfilled_orders(api_key,secret_key))>0:
                cancel_result = cancel_unfilled_order(api_key,secret_key,sell_uuid)
            else:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                my_money_info,total_balance = get_upbit_account_info(api_key,secret_key)
                add_trade(current_time, f'{hold_coin}', 'sell', sell_price,total_balance)
                break
            my_money_info,total_balance = get_upbit_account_info(api_key,secret_key)

        while prediction['probability_down']>=0.62:
            orderbook = get_orderbook(api_key, market)
            market = f'KRW-{hold_coin}'
            sell_price = orderbook['ask_price'] - coin_order_unit
            sell_uuid = place_limit_sell_order(api_key,secret_key,market,sell_price,hold_coin_amount)
            unfilled_orders = check_unfilled_orders(api_key, secret_key)
            if unfilled_orders:
                print('지정가 매도 실패, 미체결 주문 취소 작업 중..')
                for order in unfilled_orders:
                    cancel_result = cancel_unfilled_order(api_key, secret_key, sell_uuid)
            else:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                my_money_info,total_balance = get_upbit_account_info(api_key,secret_key)
                add_trade(current_time, f'{hold_coin}', 'sell', sell_price,total_balance)
                break
            df = get_coin_data(hold_coin)  # 데이터 가져오기
            # df = df.iloc[:-1]
            indicators = calculate_indicators(df)  # 지표 계산
            prediction = train_and_predict(df)  # 상승/하락 확률 계산
