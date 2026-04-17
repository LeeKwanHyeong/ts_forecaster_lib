import numpy as np
import polars as pl

def dummy_part_future_exo(uid_list, start_idxs, H):
    B = len(uid_list)
    return np.zeros((B, H, 1), dtype = np.float32)

def build_lookup_tuple(df: pl.DataFrame, id_col: str, date_col: str, value_cols: list[str]):
    df_lu = (
        df.select([id_col, date_col] + value_cols)
        .drop_nulls([id_col, date_col])
    )

    lookup = {}
    for r in df_lu.to_dicts():
        key = (str(r[id_col]), int(r[date_col]))
        lookup[key] = tuple(r[c] for c in value_cols)

    return lookup

def build_week_index(df: pl.DataFrame, date_col: str):
    ww_list = (
        df.select(pl.col(date_col))
        .drop_nulls()
        .unique()
        .sort(date_col)
        .get_column(date_col)
        .to_list()
    )

    ww_list = [int(x) for x in ww_list]
    ww_to_pos = {ww: i for i, ww in enumerate(ww_list)}
    return ww_list, ww_to_pos


def make_part_future_exo_fn(lookup, ww_list, ww_to_pos):
    def part_future_exo_fn(uid_list, start_idxs, H):
        B = len(uid_list)
        H = int(H)
        out = np.zeros((B, H, 1), dtype = np.float32)

        for b, (uid, start_ww) in enumerate(zip(uid_list, start_idxs)):
            uid = str(uid)
            start_ww = int(start_ww)

            pos = ww_to_pos.get(start_ww, None)
            if pos is None:
                continue

            future_ww = ww_list[pos-1: pos + H]
            for k, ww in enumerate(future_ww[1:]):
                vals = lookup.get((uid, int(ww)), None)
                if vals is None:
                    continue

                warranty_end, demand_ago_log, order_cumsum_log, warranty, wty_ago27 = vals
                try:
                    if (int(warranty_end) > future_ww[1]) and (int(wty_ago27) <= future_ww[1]):
                        before_val = lookup.get((uid, future_ww[0]), None)
                        max_cumsum = before_val[2]
                        if ww < int(warranty_end):
                            value = max_cumsum
                        else:
                            value = max_cumsum - demand_ago_log

                    elif (int(warranty_end) <= future_ww[1]):
                        value = order_cumsum_log - demand_ago_log
                    else:
                        value = 0.0

                    out[b, k, 0] = float(value)

                except Exception as e:
                    print(before_val)
                    print(e)

        return out
    return part_future_exo_fn