# -*- coding: utf-8 -*-
"""
Robô de sinais SAR+EMA+ADX (IQ Option) – V1

Correções aplicadas:
- Alteração do alvo da entrada para 1 minuto após o fechamento da vela -2
- Atualizados os logs em CSV para incluir novos parâmetros para validação com analise de dados
"""

from dotenv import load_dotenv
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import talib
import joblib
from iqoptionapi.stable_api import IQ_Option

# ==============================
# Configurações
# ==============================

# Login IQ Option
load_dotenv()

EMAIL = os.getenv("IQ_EMAIL")
SENHA = os.getenv("IQ_SENHA")

ATIVO = "EURJPY-OTC"
TIMEFRAME_MIN = 1  # CORRIGIDO PARA 1 MINUTO
MODO = "PRACTICE"  # "REAL" ou "PRACTICE"

# IA
MODELO_PATH = "model/rf_model.pkl"
SCALER_PATH = "model/scaler.pkl"   # opcional
IA_THRESHOLD = 0.65

# Logs
ARQ_ALERTAS = "data/logs/sar_alertas.csv"
ARQ_SAR_MINUTO = "data/logs/sar_minuto.csv"
ARQ_SINAIS = "data/logs/sar_sinais.csv"

# Indicadores
SAR_ACCEL = 0.02
SAR_MAX = 0.2
EMA_PERIODO = 50
ADX_PERIODO = 14
ADX_MINIMO = 20

VALOR_ENTRADA = 100.0
STOP_LOSS = None
TAKE_PROFIT = None

DEBUG = True

# ===================================================
# Estado
# ===================================================
pnl_sessao = 0.0
ultimo_log_dt = None
sinal_pendente = None

# Controle de inversão e agendamento
ultima_inversao_ts = None
alvo_entrada_ts = None
entrada_pendente = False
direcao_agendada = None

# ==============================
# Utilidades
# ==============================

def log_csv(path, data_dict):
    df = pd.DataFrame([data_dict])
    df.to_csv(path, mode='a', index=False, header=not Path(path).exists())


def aguardar_resultado_digital(api, order_id, valor_entrada, timeout=180, intervalo=0.5):
    tempo_esperado = 0.0
    while tempo_esperado <= timeout:
        try:
            status, lucro = api.check_win_digital_v2(order_id)
            if status:
                lucro = float(lucro) if lucro is not None else 0.0
                return ("WIN", round(lucro, 2)) if lucro > 0 else ("LOSS", -valor_entrada)
        except Exception:
            pass
        time.sleep(intervalo)
        tempo_esperado += intervalo
    return "TIMEOUT", 0.0

def get_candles_safe(api, ativo, timeframe_min, qnt, timeout=10):
    inicio = time.time()

    while True:
        try:
            fim = time.time()
            v = api.get_candles(ativo, timeframe_min * 60, qnt, fim)

            if v and len(v) > 0:
                df = pd.DataFrame(v)
                df['from_ts'] = df['from']
                df['datetime'] = pd.to_datetime(df['from'], unit='s')
                df.rename(columns={'max': 'high', 'min': 'low'}, inplace=True)
                return df[['from_ts', 'datetime', 'open', 'high', 'low', 'close']]

        except Exception as e:
            print("[WARN] Erro ao buscar candles:", e)

        # TIMEOUT DURO
        if time.time() - inicio > timeout:
            print("[WATCHDOG] get_candles travado. Reconectando API...")
            try:
                api.disconnect()
            except:
                pass

            time.sleep(2)
            api.connect()
            api.change_balance(MODO)
            return None

        time.sleep(0.5)

# ==============================
# Conexão
# ==============================
API = IQ_Option(EMAIL, SENHA)
check, reason = API.connect()
if not check:
    raise SystemExit(f"Erro ao conectar: {reason}")

API.change_balance(MODO)
print(f"Conectado - {ATIVO} | Timeframe: 1M | Modo: {MODO}")

# ==============================
# Carregar IA
# ==============================
modelo = joblib.load(MODELO_PATH)
scaler = joblib.load(SCALER_PATH) if Path(SCALER_PATH).exists() else None

print("Modelo IA carregado")

# ==============================
# Loop principal
# ==============================
while True:
    try:
        df = get_candles_safe(API, ATIVO, TIMEFRAME_MIN, 200)
        if df is None:
            print("[INFO] Pulando ciclo por falha de candles")
            time.sleep(2)
            continue

        # ===== Indicador SAR =====
        df['sar'] = talib.SAR(df['high'], df['low'], acceleration=SAR_ACCEL, maximum=SAR_MAX)
        df['posicao'] = np.where(df['sar'] > df['close'], "ACIMA", "ABAIXO")

        # ===== EMA =====
        df['ema'] = talib.EMA(df['close'], timeperiod=EMA_PERIODO)

        # ===== ADX =====
        df['adx'] = talib.ADX(
            df['high'],
            df['low'],
            df['close'],
            timeperiod=ADX_PERIODO
        )

        # ===== Trabalhar apenas com velas FECHADAS =====
        vela_fechada = df.iloc[-2]
        vela_anterior = df.iloc[-3]

        # ===== Analisar Tendencia e força =====
        if vela_fechada['adx'] is not None:
            adx_value = round(float(vela_fechada['adx']), 2)
        else:
            adx_value = 0.0  # Ou algum valor padrão

        tendencia = "ALTA" if vela_fechada['close'] > vela_fechada['ema'] else "BAIXA"
        forca_ok = adx_value >= ADX_MINIMO

        # ===== Log por minuto (duplicado) =====
        if ultimo_log_dt is None or vela_fechada['datetime'] != ultimo_log_dt:
            log_csv(ARQ_SAR_MINUTO, {
                "data": vela_fechada['datetime'],
                "open": vela_fechada['open'],
                "high": vela_fechada['high'],
                "low": vela_fechada['low'],
                "close": vela_fechada['close'],
                "sar": float(vela_fechada['sar']),
                "posicao": vela_fechada['posicao']
            })
            print(f"SAR salvo: {vela_fechada['datetime']} SAR={round(float(vela_fechada['sar']), 6)} {vela_fechada['posicao']}")
            ultimo_log_dt = vela_fechada['datetime']

        # ===== Avalia resultado teórico do sinal anterior =====
        if sinal_pendente is not None:
            vela_resultado = df.iloc[-1]  # vela -1 fechada agora

            direcao = sinal_pendente["direcao"]
            open_v = vela_resultado["open"]
            close_v = vela_resultado["close"]

            if direcao == "call":
                resultado_teorico = "WIN" if close_v > open_v else "LOSS"
            else:
                resultado_teorico = "WIN" if close_v < open_v else "LOSS"

            log_csv(ARQ_SINAIS, {
                "data": sinal_pendente["data"],
                "ativo": ATIVO,
                "direcao_sar": direcao,
                "tendencia_ema": sinal_pendente["tendencia"],
                "ema_ok": sinal_pendente["ema_ok"],
                "adx": sinal_pendente["adx"],
                "adx_ok": sinal_pendente["adx_ok"],
                "ia_prob": sinal_pendente["ia_prob"],
                "status": sinal_pendente["status"],
                "motivo": sinal_pendente["motivo"],
                "open_resultado": open_v,
                "close_resultado": close_v,
                "resultado_teorico": resultado_teorico
            })

            sinal_pendente = None

        # ===== 1) Detecta inversão (-2 -> -1 FECHADAS) =====
        if not entrada_pendente:
            houve_inversao = vela_anterior['posicao'] != vela_fechada['posicao']
            if houve_inversao:
                ts_fechamento = int(vela_fechada['from_ts'])

                if ultima_inversao_ts != ts_fechamento:
                    ultima_inversao_ts = ts_fechamento

                    direcao_calculada = "put" if vela_fechada['posicao'] == "ACIMA" else "call"

                    ema_ok = (
                        (direcao_calculada == "call" and tendencia == "ALTA") or
                        (direcao_calculada == "put" and tendencia == "BAIXA")
                    )

                    adx_ok = forca_ok

                    # Prepare features for AI
                    features = np.array([[
                        adx_value,  # adx
                        1 if ema_ok else 0,  # ema_ok
                        1 if adx_ok else 0   # adx_ok
                    ]])

                    if scaler:
                        features = scaler.transform(features)

                    prob = modelo.predict_proba(features)[0][1]
                    ia_prob = round(prob, 4)

                    if prob >= IA_THRESHOLD:
                        status_sinal = "APROVADO"
                        motivo = "OK"
                    else:
                        status_sinal = "NEGADO"
                        motivo = "IA_BAIXA_PROBABILIDADE"

                    # ===== Salvando o sinal pendente para validação =====
                    if sinal_pendente is None:
                        sinal_pendente = {
                        "data": vela_fechada['datetime'],
                        "direcao": direcao_calculada,
                        "tendencia": tendencia,
                        "ema_ok": ema_ok,
                        "adx": round(float(vela_fechada['adx']), 2),
                        "adx_ok": adx_ok,
                        "ia_prob": ia_prob,
                        "status": status_sinal,
                        "motivo": motivo
                    }


                    if status_sinal == "NEGADO":
                        if DEBUG:
                            print(f"[SINAL NEGADO] {motivo}")
                        continue

                    # ===== SINAL APROVADO → AGENDA ENTRADA =====
                    alvo_entrada_ts = ts_fechamento + 60
                    direcao_agendada = direcao_calculada
                    entrada_pendente = True

                    if DEBUG:
                        print("[SINAL APROVADO] SAR + EMA + ADX")
                        print(f"  Direção: {direcao_agendada.upper()}")
                        print(f"  EMA: {tendencia}")
                        print(f"  ADX: {vela_fechada['adx']:.2f}")
                        print(f"  IA Prob: {ia_prob}")
                        print(f"  Entrada: {pd.to_datetime(alvo_entrada_ts, unit='s')}")

        # ===== 2) Executa entrada =====
        if entrada_pendente and alvo_entrada_ts and direcao_agendada:
            agora = time.time()
            if alvo_entrada_ts <= agora <= (alvo_entrada_ts + 15):
                print(f"Executando entrada: {direcao_agendada.upper()} | {pd.to_datetime(alvo_entrada_ts, unit='s')}")
                try:
                    ok, order_id = API.buy_digital_spot_v2(ATIVO, VALOR_ENTRADA, direcao_agendada, 1)
                    if ok:
                        resultado, lucro = aguardar_resultado_digital(API, order_id, VALOR_ENTRADA)
                        pnl_sessao += float(lucro)
                        try:
                            saldo_pos = API.get_balance()
                        except Exception:
                            saldo_pos = None

                        log_csv(ARQ_ALERTAS, {
                            "data_decisao": vela_fechada['datetime'],
                            "order_id": order_id,
                            "ativo": ATIVO,
                            "timeframe": TIMEFRAME_MIN,
                            "direcao_entrada": direcao_agendada,
                            "valor_entrada": VALOR_ENTRADA,
                            "resultado": resultado,
                            "lucro": round(float(lucro), 2),
                            "pnl_sessao": round(float(pnl_sessao), 2),
                            "saldo_pos_trade": saldo_pos
                        })

                        print(f"Resultado: {resultado} | Lucro: {lucro} | PnL: {pnl_sessao} | Saldo: {saldo_pos}")

                        if pnl_sessao <= STOP_LOSS:
                            print("STOP LOSS atingido. Encerrando robô.")
                            break
                        if TAKE_PROFIT is not None and pnl_sessao >= TAKE_PROFIT:
                            print("TAKE PROFIT atingido. Encerrando robô.")
                            break
                    else:
                        print("Falha ao abrir ordem.")
                except Exception as e:
                    print("Erro na ordem:", e)
                finally:
                    entrada_pendente = False
                    alvo_entrada_ts = None
                    direcao_agendada = None

        time.sleep(0.2)

    except Exception as e:
        print("Erro geral:", e)
        time.sleep(5)

# FIM DO SCRIPT
