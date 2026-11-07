"""
Generate a SYNTHETIC donation dataset for the Donor Reactivation demo.

Every donor, amount and date here is randomly generated. Nothing comes from
any real organization. The shape (programs, Ramadan seasonality, 50-60 day
giving intervals, a large dormant base) mimics a typical Indonesian
zakat/infaq/sedekah charity so the analysis is realistic.

Usage:  python generate_data.py            -> data/transactions.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_DONORS = 20_000
START = pd.Timestamp("2020-01-01")
SNAPSHOT = pd.Timestamp("2026-09-30")

# Approximate Ramadan windows (start, end) - drives the seasonal spike
RAMADAN = [
    ("2020-04-24", "2020-05-23"), ("2021-04-13", "2021-05-12"),
    ("2022-04-02", "2022-05-01"), ("2023-03-23", "2023-04-21"),
    ("2024-03-11", "2024-04-09"), ("2025-03-01", "2025-03-30"),
    ("2026-02-18", "2026-03-19"),
]

# program -> (akad, weight, median amount IDR)
PROGRAMS = {
    "Zakat Penghasilan":        ("Zakat",   0.20, 500_000),
    "Sedekah Umum":             ("Sedekah", 0.18, 150_000),
    "Zakat Maal":               ("Zakat",   0.07, 2_000_000),
    "Infaq Pendidikan Yatim":   ("Infaq",   0.10, 200_000),
    "Sedekah Jumat":            ("Sedekah", 0.10, 50_000),
    "Orang Tua Asuh":           ("Infaq",   0.07, 300_000),
    "Paket Berbuka Puasa":      ("Sedekah", 0.06, 100_000),
    "Wakaf Pembangunan":        ("Wakaf",   0.05, 1_000_000),
    "Sedekah Pangan":           ("Sedekah", 0.08, 75_000),
    "Bantuan Kemanusiaan":      ("Infaq",   0.09, 250_000),
}
ZAKAT_FITRAH = ("Zakat Fitrah", "Zakat", 45_000)

# archetype -> (share, mean gifts while active, mean interval days)
ARCHETYPES = {
    "one_time":   (0.48, 1.0, 0),
    "occasional": (0.30, 3.0, 150),
    "regular":    (0.16, 8.0, 55),
    "loyal":      (0.06, 25.0, 32),
}


def ramadan_mask(dates: pd.Series) -> np.ndarray:
    m = np.zeros(len(dates), dtype=bool)
    for s, e in RAMADAN:
        m |= (dates >= pd.Timestamp(s)) & (dates <= pd.Timestamp(e))
    return m


def main(out_dir: str = "data") -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    names = list(PROGRAMS)
    weights = np.array([PROGRAMS[p][1] for p in names])
    weights = weights / weights.sum()
    arch_names = list(ARCHETYPES)
    arch_p = np.array([ARCHETYPES[a][0] for a in arch_names])

    total_days = (SNAPSHOT - START).days
    rows = []
    for i in range(N_DONORS):
        donor_id = f"DMO{i + 1:06d}"
        arch = rng.choice(arch_names, p=arch_p)
        _, mean_gifts, mean_gap = ARCHETYPES[arch]
        # first gift skewed to earlier years (older base -> more dormant donors)
        first = START + pd.Timedelta(days=int(total_days * rng.beta(1.3, 2.2)))
        n_gifts = 1 if arch == "one_time" else 1 + rng.poisson(mean_gifts - 1)
        fav = rng.choice(names, p=weights)
        size_factor = rng.lognormal(0, 0.6)  # some donors give bigger
        d = first
        for g in range(n_gifts):
            if d > SNAPSHOT:
                break
            prog = fav if rng.random() < 0.7 else rng.choice(names, p=weights)
            akad, _, med = PROGRAMS[prog]
            amount = med * size_factor * rng.lognormal(0, 0.5)
            rows.append((donor_id, d, prog, akad, amount))
            if mean_gap:
                d = d + pd.Timedelta(days=max(7, int(rng.gamma(4, mean_gap / 4))))
        # some lapsed donors come back on their own at a random later date
        # (keeps the problem realistic: recency alone does not decide everything)
        if rng.random() < (0.04 if arch == "one_time" else 0.12):
            back = d + pd.Timedelta(days=int(rng.integers(60, 1500)))
            if back <= SNAPSHOT:
                akad, _, med = PROGRAMS[fav]
                rows.append((donor_id, back, fav, akad, med * size_factor * rng.lognormal(0, 0.5)))

    tx = pd.DataFrame(rows, columns=["id_donor", "tgl_donasi", "program", "akad", "nominal"])

    # Ramadan boost: extra Zakat Fitrah / Berbuka gifts from donors active that year
    extra = []
    for s, e in RAMADAN:
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        if s > SNAPSHOT:
            continue
        active = tx.loc[(tx.tgl_donasi > s - pd.Timedelta(days=365)) & (tx.tgl_donasi <= e), "id_donor"].unique()
        pick = rng.choice(active, size=int(len(active) * 0.35), replace=False) if len(active) else []
        for did in pick:
            day = s + pd.Timedelta(days=int(rng.integers(0, (e - s).days + 1)))
            if rng.random() < 0.6:
                prog, akad, med = ZAKAT_FITRAH
            else:
                prog, (akad, _, med) = "Paket Berbuka Puasa", PROGRAMS["Paket Berbuka Puasa"]
            extra.append((did, day, prog, akad, med * rng.lognormal(0, 0.4)))
    tx = pd.concat([tx, pd.DataFrame(extra, columns=tx.columns)], ignore_index=True)

    tx["nominal"] = (tx["nominal"] / 1000).round().clip(lower=10) * 1000  # round to Rp1.000
    tx = tx.sort_values(["tgl_donasi", "id_donor"]).reset_index(drop=True)
    tx.insert(0, "id_transaksi", [f"TRX{i + 1:07d}" for i in range(len(tx))])

    Path(out_dir).mkdir(exist_ok=True)
    tx.to_csv(Path(out_dir) / "transactions.csv", index=False, date_format="%Y-%m-%d")
    print(f"{len(tx):,} transactions, {tx.id_donor.nunique():,} donors -> {out_dir}/transactions.csv")
    return tx


if __name__ == "__main__":
    main()
