#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_bn_data.py
================
从 https://github.com/bailiang0369/bn_hisdata/releases/tag/vision-base 自动获取 Release Assets，
支持根据 GitHub API 动态匹配 ETH 1分钟级别（现货/合约）Parquet 文件，并合并为全量数据集。
"""

import os
import time
import json
import argparse
import urllib.request
import polars as pl
from typing import List, Optional


RELEASE_API_URL = "https://api.github.com/repos/bailiang0369/bn_hisdata/releases/tags/vision-base"


def fetch_release_assets_info() -> List[dict]:
    """
    通过 GitHub API 动态获取 vision-base release 下所有资产列表。
    """
    req = urllib.request.Request(RELEASE_API_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    return data.get('assets', [])


def download_and_combine_bn_klines(
    symbol: str = "ETHUSDT",
    output_dir: str = "/tmp/bn_klines",
    merged_output_path: str = "/app/bn_ethusdt_1m_all.parquet",
    use_spot: bool = False
) -> str:
    """
    动态发现、下载并合并指定 symbol 的所有 1m parquet 文件。
    """
    os.makedirs(output_dir, exist_ok=True)
    assets = fetch_release_assets_info()

    # 筛选包含 symbol 且包含 1m 的 parquet 文件
    target_assets = [
        a for a in assets
        if symbol.lower() in a['name'].lower() and '1m' in a['name'].lower() and a['name'].endswith('.parquet')
    ]
    target_assets.sort(key=lambda x: x['name'])

    if not target_assets:
        raise RuntimeError(f"在 Release vision-base 中未找到匹配 {symbol} 的 1m parquet 资产！")

    print(f"=== 动态发现 {len(target_assets)} 个匹配资产文件 ===")
    downloaded_files = []

    for a in target_assets:
        filename = a['name']
        download_url = a['browser_download_url']
        target_path = os.path.join(output_dir, filename)

        if not os.path.exists(target_path) or os.path.getsize(target_path) < 1000:
            print(f"正在下载 {filename} ...")
            start_t = time.time()
            req = urllib.request.Request(download_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as resp, open(target_path, 'wb') as f:
                while chunk := resp.read(1024 * 1024 * 4): # 4MB 分块
                    f.write(chunk)
            size_mb = os.path.getsize(target_path) / (1024 * 1024)
            print(f"-> 下载完成 {filename} ({size_mb:.2f} MB, 耗时 {time.time() - start_t:.1f}s)")
        else:
            print(f"-> 本地已缓存: {filename}")

        downloaded_files.append(target_path)

    print("\n=== 正在合并所有年度 Parquet 文件并进行因果时序去重 ===")
    dfs = []
    for p in downloaded_files:
        df_part = pl.read_parquet(p)
        print(f"读取分片 {os.path.basename(p)}: 样本行数={len(df_part):,}")
        dfs.append(df_part)

    merged = pl.concat(dfs)
    print(f"合并原始总记录数: {len(merged):,}")

    # 标准化时间戳 ts
    for c in ['close_time', 'open_time', 'timestamp', 'time']:
        if c in merged.columns and 'ts' not in merged.columns:
            merged = merged.with_columns(ts = pl.col(c))
            break

    # 标准化现货或合约价格列
    cols = merged.columns
    if use_spot and 'spot_close' in cols:
        print("指定使用现货行情: 将 spot_open/high/low/close 映射为主行情列")
        merged = merged.with_columns([
            pl.col('spot_open').alias('open'),
            pl.col('spot_high').alias('high'),
            pl.col('spot_low').alias('low'),
            pl.col('spot_close').alias('close')
        ])
    else:
        if 'contract_close' in cols and 'close' not in cols:
            merged = merged.with_columns([
                pl.col('contract_open').alias('open'),
                pl.col('contract_high').alias('high'),
                pl.col('contract_low').alias('low'),
                pl.col('contract_close').alias('close')
            ])

    # 按时间戳去重并严格升序排序
    merged = merged.unique(subset=['ts']).sort('ts')

    # 保存最终合并全量文件
    merged.write_parquet(merged_output_path)
    print(f"全量时序数据集已成功持久化至: {merged_output_path}")
    print(f"有效时序总跨度: {len(merged):,} 根 1m K线")
    return merged_output_path


def main():
    parser = argparse.ArgumentParser(description="Binance 历史 1m K线全量下载与合并工具")
    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="币对，默认 ETHUSDT")
    parser.add_argument("--output", type=str, default="/app/bn_ethusdt_1m_all.parquet", help="输出路径")
    parser.add_argument("--spot", action="store_true", help="是否指定使用现货价格，默认合约价格")
    args = parser.parse_args()

    download_and_combine_bn_klines(symbol=args.symbol, merged_output_path=args.output, use_spot=args.spot)


if __name__ == '__main__':
    main()
