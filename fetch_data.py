#!/usr/bin/env python3

import asyncio
import json
import csv
import time
import signal
from pathlib import Path
import ssl

import websockets

STOP = asyncio.Event()

ORDERBOOK_HEADERS = ["timestamp"] + [
    f"{side}_{field}_{i}"
    for side in ["bid", "ask"]
    for i in range(1, 6)
    for field in ["price", "volume"]
]

TRADES_HEADERS = ["timestamp", "price", "quantity", "side"]


def ts():
    return int(time.time() * 1000)


def parse_levels(levels, n=5):
    out = []
    for i in range(n):
        if i < len(levels):
            p, q = levels[i]
        else:
            p, q = "", ""
        out.extend([p, q])
    return out


async def writer(queue, ob_path, tr_path):
    ob_f = open(ob_path, "a", newline="")
    tr_f = open(tr_path, "a", newline="")

    ob_writer = csv.writer(ob_f)
    tr_writer = csv.writer(tr_f)

    while True:
        item = await queue.get()
        if item is None:
            break

        kind, data = item

        if kind == "depth":
            row = [ts()]
            row += parse_levels(data["bids"])
            row += parse_levels(data["asks"])
            ob_writer.writerow(row)
            ob_f.flush()

        elif kind == "trade":
            row = [
                ts(),
                data["p"],
                data["q"],
                "sell" if data["m"] else "buy",
            ]
            tr_writer.writerow(row)
            tr_f.flush()

        queue.task_done()

    ob_f.close()
    tr_f.close()


async def ws_loop(symbol, queue):
    url = (
        f"wss://stream.binance.com:9443/stream?"
        f"streams={symbol}@depth@100ms/{symbol}@trade"
    )

    # ✅ 关键：关闭SSL验证（解决你现在的问题）
    ssl_context = ssl._create_unverified_context()

    while not STOP.is_set():
        try:
            async with websockets.connect(url, ssl=ssl_context) as ws:
                print("[WS] Connected")

                async for msg in ws:
                    if STOP.is_set():
                        break

                    msg = json.loads(msg)

                    # DEBUG（确认有数据）
                    print("[STREAM]", msg.get("stream"))

                    data = msg.get("data", {})

                    # trade
                    if data.get("e") == "trade":
                        await queue.put(("trade", data))

                    # depth（关键）
                    elif "b" in data and "a" in data:
                        print("[DEPTH OK]")
                        await queue.put(
                            ("depth", {"bids": data["b"], "asks": data["a"]})
                        )

        except Exception as e:
            print("[WS ERROR]", e)
            await asyncio.sleep(3)


def ensure_header(path, headers):
    if not path.exists() or path.stat().st_size == 0:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(headers)


def main():
    symbol = "btcusdt"

    ob_path = Path("orderbook.csv")
    tr_path = Path("trades.csv")

    ensure_header(ob_path, ORDERBOOK_HEADERS)
    ensure_header(tr_path, TRADES_HEADERS)

    queue = asyncio.Queue()

    def stop(*args):
        print("\nStopping...")
        STOP.set()

    signal.signal(signal.SIGINT, stop)

    async def run():
        w = asyncio.create_task(writer(queue, ob_path, tr_path))
        c = asyncio.create_task(ws_loop(symbol, queue))

        await STOP.wait()

        c.cancel()
        await queue.put(None)
        await w

    asyncio.run(run())


if __name__ == "__main__":
    main()