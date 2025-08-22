import io
from datetime import datetime, timedelta, date
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# =====================================
# Utilidades comunes
# =====================================

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Alojamiento", "Fecha alta", "Fecha entrada", "Fecha salida", "Precio"]
    for col in required:
        if col not in df.columns:
            st.error(f"Falta la columna obligatoria: {col}")
            st.stop()
    df["Fecha alta"] = pd.to_datetime(df["Fecha alta"], errors="coerce")
    df["Fecha entrada"] = pd.to_datetime(df["Fecha entrada"], errors="coerce")
    df["Fecha salida"] = pd.to_datetime(df["Fecha salida"], errors="coerce")
    df["Alojamiento"] = df["Alojamiento"].astype(str).str.strip()
    df["Precio"] = pd.to_numeric(df["Precio"], errors="coerce").fillna(0.0)
    return df

@st.cache_data(show_spinner=False)
def load_excel_from_blobs(file_blobs: List[tuple[str, bytes]]) -> pd.DataFrame:
    """Carga y concatena varios Excel a partir de blobs (nombre, bytes). Se cachea por contenido."""
    frames = []
    for name, data in file_blobs:
        try:
            xls = pd.ExcelFile(io.BytesIO(data))
            sheet = (
                "Estado de pagos de las reservas"
                if "Estado de pagos de las reservas" in xls.sheet_names
                else xls.sheet_names[0]
            )
            df = pd.read_excel(xls, sheet_name=sheet)
            df["__source_file__"] = name
            frames.append(df)
        except Exception as e:
            st.error(f"No se pudo leer {name}: {e}")
            st.stop()
    if not frames:
        return pd.DataFrame()
    df_all = pd.concat(frames, ignore_index=True)
    return parse_dates(df_all)

# --- Cálculo vectorizado de KPIs (rápido)

def compute_kpis(
    df_all: pd.DataFrame,
    cutoff: pd.Timestamp,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    inventory_override: Optional[int] = None,
    filter_props: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Calcula KPIs de forma vectorizada sin expandir noche a noche."""
    # 1) Filtrar por corte y propiedades
    df_cut = df_all[df_all["Fecha alta"] <= cutoff].copy()
    if filter_props:
        df_cut = df_cut[df_cut["Alojamiento"].isin(filter_props)]

    # Quitar filas sin fechas válidas
    df_cut = df_cut.dropna(subset=["Fecha entrada", "Fecha salida"]).copy()

    if df_cut.empty:
        inv = len(set(filter_props)) if filter_props else df_all["Alojamiento"].nunique()
        if inventory_override and inventory_override > 0:
            inv = int(inventory_override)
        days = (period_end - period_start).days + 1
        total = {
            "noches_ocupadas": 0,
            "noches_disponibles": inv * days,
            "ocupacion_pct": 0.0,
            "ingresos": 0.0,
            "adr": 0.0,
            "revpar": 0.0,
        }
        return pd.DataFrame(columns=["Alojamiento", "Noches ocupadas", "Ingresos", "ADR"]), total

    one_day = np.timedelta64(1, 'D')
    start_ns = np.datetime64(pd.to_datetime(period_start))
    end_excl_ns = np.datetime64(pd.to_datetime(period_end) + pd.Timedelta(days=1))  # fin inclusivo

    arr_e = df_cut["Fecha entrada"].values.astype('datetime64[ns]')
    arr_s = df_cut["Fecha salida"].values.astype('datetime64[ns]')
    arr_c = df_cut["Fecha alta"].values.astype('datetime64[ns]')

    total_nights = ((arr_s - arr_e) / one_day).astype('int64')
    total_nights = np.clip(total_nights, 0, None)

    ov_start = np.maximum(arr_e, start_ns)
    ov_end = np.minimum(arr_s, end_excl_ns)
    ov_days = ((ov_end - ov_start) / one_day).astype('int64')
    ov_days = np.clip(ov_days, 0, None)

    if ov_days.sum() == 0:
        inv = len(set(filter_props)) if filter_props else df_all["Alojamiento"].nunique()
        if inventory_override and inventory_override > 0:
            inv = int(inventory_override)
        days = (period_end - period_start).days + 1
        total = {
            "noches_ocupadas": 0,
            "noches_disponibles": inv * days,
            "ocupacion_pct": 0.0,
            "ingresos": 0.0,
            "adr": 0.0,
            "revpar": 0.0,
        }
        return pd.DataFrame(columns=["Alojamiento", "Noches ocupadas", "Ingresos", "ADR"]), total

    price = df_cut["Precio"].values.astype('float64')
    with np.errstate(divide='ignore', invalid='ignore'):
        share = np.where(total_nights > 0, ov_days / total_nights, 0.0)
    income = price * share

    props = df_cut["Alojamiento"].astype(str).values
    df_agg = pd.DataFrame({"Alojamiento": props, "Noches": ov_days, "Ingresos": income})
    by_prop = df_agg.groupby("Alojamiento", as_index=False).sum(numeric_only=True)
    by_prop.rename(columns={"Noches": "Noches ocupadas"}, inplace=True)
    by_prop["ADR"] = np.where(by_prop["Noches ocupadas"] > 0, by_prop["Ingresos"] / by_prop["Noches ocupadas"], 0.0)
    by_prop = by_prop.sort_values("Alojamiento")

    noches_ocupadas = int(by_prop["Noches ocupadas"].sum())
    ingresos = float(by_prop["Ingresos"].sum())
    adr = float(ingresos / noches_ocupadas) if noches_ocupadas > 0 else 0.0

    inv = len(set(filter_props)) if filter_props else df_all["Alojamiento"].nunique()
    if inventory_override and inventory_override > 0:
        inv = int(inventory_override)
    days = (period_end - period_start).days + 1
    noches_disponibles = inv * days
    ocupacion_pct = (noches_ocupadas / noches_disponibles * 100) if noches_disponibles > 0 else 0.0
    revpar = ingresos / noches_disponibles if noches_disponibles > 0 else 0.0

    tot = {
        "noches_ocupadas": noches_ocupadas,
        "noches_disponibles": noches_disponibles,
        "ocupacion_pct": ocupacion_pct,
        "ingresos": ingresos,
        "adr": adr,
        "revpar": revpar,
    }

    return by_prop, tot

# -------------------------------------
# Helpers adicionales
# -------------------------------------

def daily_series(df_all: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp, props: Optional[List[str]], inventory_override: Optional[int]) -> pd.DataFrame:
    """Devuelve serie diaria de KPIs (noches, ingresos, ocupación %, ADR, RevPAR)."""
    days = list(pd.date_range(start, end, freq='D'))
    rows = []
    for d in days:
        _bp, tot = compute_kpis(
            df_all=df_all,
            cutoff=cutoff,
            period_start=d,
            period_end=d,
            inventory_override=inventory_override,
            filter_props=props,
        )
        rows.append({"Fecha": d.normalize(), **tot})
    return pd.DataFrame(rows)


def build_calendar_matrix(df_all: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp, props: Optional[List[str]], mode: str = "Ocupado/Libre") -> pd.DataFrame:
    """Matriz (alojamientos x días) con '■' si ocupado o ADR por noche si mode='ADR'."""
    df_cut = df_all[(df_all["Fecha alta"] <= cutoff)].copy()
    if props:
        df_cut = df_cut[df_cut["Alojamiento"].isin(props)]
    df_cut = df_cut.dropna(subset=["Fecha entrada", "Fecha salida"])
    if df_cut.empty:
        return pd.DataFrame()

    rows = []
    for _, r in df_cut.iterrows():
        e, s, p = r["Fecha entrada"], r["Fecha salida"], float(r["Precio"])
        ov_start = max(e, start)
        ov_end = min(s, end + pd.Timedelta(days=1))
        n_nights = (s - e).days
        if ov_start >= ov_end or n_nights <= 0:
            continue
        adr_night = p / n_nights if n_nights > 0 else 0.0
        for d in pd.date_range(ov_start, ov_end - pd.Timedelta(days=1), freq='D'):
            rows.append({"Alojamiento": r["Alojamiento"], "Fecha": d.normalize(), "Ocupado": 1, "ADR_noche": adr_night})
    if not rows:
        return pd.DataFrame()
    df_nightly = pd.DataFrame(rows)

    if mode == "Ocupado/Libre":
        piv = df_nightly.pivot_table(index="Alojamiento", columns="Fecha", values="Ocupado", aggfunc='sum', fill_value=0)
        piv = piv.applymap(lambda x: '■' if x > 0 else '')
    else:
        piv = df_nightly.pivot_table(index="Alojamiento", columns="Fecha", values="ADR_noche", aggfunc='mean', fill_value='')

    piv = piv.reindex(sorted(piv.columns), axis=1)
    return piv


def pace_series(df_all: pd.DataFrame, period_start: pd.Timestamp, period_end: pd.Timestamp, d_max: int, props: Optional[List[str]], inv_override: Optional[int]) -> pd.DataFrame:
    """Curva Pace: para cada D (0..d_max), noches/ingresos confirmados a D días antes de la estancia.
    Fórmula por reserva: noches(D) = len( [max(ov_start, created_at + D), ov_end) ) en días
    donde ov_* es la intersección de [entrada, salida) con [period_start, period_end+1).
    """
    df = df_all.dropna(subset=["Fecha alta", "Fecha entrada", "Fecha salida"]).copy()
    if props:
        df = df[df["Alojamiento"].isin(props)]
    if df.empty:
        return pd.DataFrame({"D": list(range(d_max + 1)), "noches": 0, "ingresos": 0.0, "ocupacion_pct": 0.0, "adr": 0.0, "revpar": 0.0})

    one_day = np.timedelta64(1, 'D')
    start_ns = np.datetime64(pd.to_datetime(period_start))
    end_excl_ns = np.datetime64(pd.to_datetime(period_end) + pd.Timedelta(days=1))

    e = df["Fecha entrada"].values.astype('datetime64[ns]')
    s = df["Fecha salida"].values.astype('datetime64[ns]')
    c = df["Fecha alta"].values.astype('datetime64[ns]')
    price = df["Precio"].values.astype('float64')

    total_nights = ((s - e) / one_day).astype('int64')
    total_nights = np.clip(total_nights, 0, None)
    adr_night = np.where(total_nights > 0, price / total_nights, 0.0)

    ov_start = np.maximum(e, start_ns)
    ov_end = np.minimum(s, end_excl_ns)
    valid = (ov_end > ov_start) & (total_nights > 0)
    if not valid.any():
        inv = len(set(props)) if props else df_all["Alojamiento"].nunique()
        if inv_override and inv_override > 0:
            inv = int(inv_override)
        days = (period_end - period_start).days + 1
        return pd.DataFrame({"D": list(range(d_max + 1)), "noches": 0, "ingresos": 0.0, "ocupacion_pct": 0.0, "adr": 0.0, "revpar": 0.0})

    e = e[valid]; s = s[valid]; c = c[valid]; ov_start = ov_start[valid]; ov_end = ov_end[valid]; adr_night = adr_night[valid]

    D_vals = np.arange(0, d_max + 1, dtype='int64')
    D_td = D_vals * one_day  # shape (D,)

    # start threshold por D: c + D
    start_thr = c[:, None] + D_td[None, :]
    ov_start_b = np.maximum(ov_start[:, None], start_thr)  # (n, D)
    nights_D = ((ov_end[:, None] - ov_start_b) / one_day).astype('int64')
    nights_D = np.clip(nights_D, 0, None)

    nights_series = nights_D.sum(axis=0).astype(float)
    ingresos_series = (nights_D * adr_night[:, None]).sum(axis=0)

    inv = len(set(props)) if props else df_all["Alojamiento"].nunique()
    if inv_override and inv_override > 0:
        inv = int(inv_override)
    days = (period_end - period_start).days + 1
    disponibles = inv * days if days > 0 else 0

    occ_series = (nights_series / disponibles * 100.0) if disponibles > 0 else np.zeros_like(nights_series)
    adr_series = np.where(nights_series > 0, ingresos_series / nights_series, 0.0)
    revpar_series = (ingresos_series / disponibles) if disponibles > 0 else np.zeros_like(ingresos_series)

    return pd.DataFrame({
        "D": D_vals,
        "noches": nights_series,
        "ingresos": ingresos_series,
        "ocupacion_pct": occ_series,
        "adr": adr_series,
        "revpar": revpar_series,
    })

# =====================================
# App (con archivos persistentes en sesión)
# =====================================

st.set_page_config(page_title="Consultas OTB por corte", layout="wide")
st.title("📅 Consultas OTB – Ocupación, ADR y RevPAR a fecha de corte")
st.caption("Sube archivos una vez y úsalos en cualquiera de los modos.")

# --- Gestor de archivos global ---
with st.sidebar:
    st.header("Archivos de trabajo (persisten en la sesión)")
    files_master = st.file_uploader(
        "Sube uno o varios Excel",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
        key="files_master",
        help="Se admiten múltiples años (2024, 2025…).",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Usar estos archivos", type="primary"):
            if files_master:
                blobs = [(f.name, f.getvalue()) for f in files_master]
                df_loaded = load_excel_from_blobs(blobs)
                st.session_state["raw_df"] = df_loaded
                st.session_state["file_names"] = [n for n, _ in blobs]
                st.success(f"Cargados {len(blobs)} archivo(s)")
            else:
                st.warning("No seleccionaste archivos.")
    with col_b:
        if st.button("Limpiar archivos"):
            st.session_state.pop("raw_df", None)
            st.session_state.pop("file_names", None)
            st.info("Archivos eliminados de la sesión.")

# Targets opcionales (persisten en sesión)
with st.sidebar.expander("🎯 Cargar Targets (opcional)"):
    tgt_file = st.file_uploader("CSV Targets", type=["csv"], key="tgt_csv")
    if tgt_file is not None:
        try:
            df_tgt = pd.read_csv(tgt_file)
            # Normalizar columnas esperadas si existen
            # year, month, target_occ_pct, target_adr, target_revpar, target_nights, target_revenue
            st.session_state["targets_df"] = df_tgt
            st.success("Targets cargados en sesión.")
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")

raw = st.session_state.get("raw_df")
file_names = st.session_state.get("file_names", [])

if raw is not None:
    with st.expander("📂 Archivos cargados"):
        st.write("**Lista:**", file_names)
        st.write(f"**Alojamientos detectados:** {raw['Alojamiento'].nunique()}")
else:
    st.info("Sube archivos en la barra lateral y pulsa **Usar estos archivos** para empezar.")

# -----------------------------
# Menú de modos (independientes)
# -----------------------------
mode = st.sidebar.radio(
    "Modo de consulta",
    [
        "Consulta normal",
        "KPIs por meses",
        "Evolución por fecha de corte",
        "Pickup (entre dos cortes)",
        "Pace (curva D)",
        "Lead time & LOS",
        "DOW heatmap",
        "ADR bands & Targets",
        "Calendario por alojamiento",
    ],
)

# Helper para mapear nombres de métricas a columnas
METRIC_MAP = {"Ocupación %": "ocupacion_pct", "ADR (€)": "adr", "RevPAR (€)": "revpar"}

# =============================
# MODO 1: Consulta normal (+ comparación año anterior con inventario propio)
# =============================
if mode == "Consulta normal":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        cutoff_normal = st.date_input("Fecha de corte", value=date(2024, 8, 21), key="cutoff_normal")
        c1, c2 = st.columns(2)
        with c1:
            start_normal = st.date_input("Inicio del periodo", value=date(2024, 9, 1), key="start_normal")
        with c2:
            end_normal = st.date_input("Fin del periodo", value=date(2024, 9, 30), key="end_normal")
        inv_normal = st.number_input("Sobrescribir inventario (nº alojamientos)", min_value=0, value=0, step=1, key="inv_normal")
        props_normal = st.multiselect("Filtrar alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="props_normal")
        st.markdown("—")
        compare_normal = st.checkbox("Comparar con año anterior (mismo día/mes)", value=False, key="cmp_normal")
        inv_normal_prev = st.number_input("Inventario año anterior (opcional)", min_value=0, value=0, step=1, key="inv_normal_prev")

    # Cálculo base
    by_prop_n, total_n = compute_kpis(
        df_all=raw,
        cutoff=pd.to_datetime(cutoff_normal),
        period_start=pd.to_datetime(start_normal),
        period_end=pd.to_datetime(end_normal),
        inventory_override=int(inv_normal) if inv_normal > 0 else None,
        filter_props=props_normal if props_normal else None,
    )

    st.subheader("Resultados totales")
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
    c1.metric("Noches ocupadas", f"{total_n['noches_ocupadas']:,}".replace(",", "."))
    c2.metric("Noches disponibles", f"{total_n['noches_disponibles']:,}".replace(",", "."))
    c3.metric("Ocupación", f"{total_n['ocupacion_pct']:.2f}%")
    c4.metric("Ingresos (€)", f"{total_n['ingresos']:.2f}")
    c5.metric("ADR (€)", f"{total_n['adr']:.2f}")
    c6.metric("RevPAR (€)", f"{total_n['revpar']:.2f}")

    if compare_normal:
        cutoff_cmp = (pd.to_datetime(cutoff_normal) - pd.DateOffset(years=1)).date()
        start_cmp = (pd.to_datetime(start_normal) - pd.DateOffset(years=1)).date()
        end_cmp = (pd.to_datetime(end_normal) - pd.DateOffset(years=1)).date()
        _bp_c, total_cmp = compute_kpis(
            df_all=raw,
            cutoff=pd.to_datetime(cutoff_cmp),
            period_start=pd.to_datetime(start_cmp),
            period_end=pd.to_datetime(end_cmp),
            inventory_override=int(inv_normal_prev) if inv_normal_prev > 0 else None,
            filter_props=props_normal if props_normal else None,
        )
        st.markdown("**Comparativa con año anterior** (corte y periodo -1 año)")
        d1, d2, d3 = st.columns(3)
        d4, d5, d6 = st.columns(3)
        d1.metric("Noches ocupadas (prev)", f"{total_cmp['noches_ocupadas']:,}".replace(",", "."), delta=total_n['noches_ocupadas']-total_cmp['noches_ocupadas'])
        d2.metric("Noches disp. (prev)", f"{total_cmp['noches_disponibles']:,}".replace(",", "."), delta=total_n['noches_disponibles']-total_cmp['noches_disponibles'])
        d3.metric("Ocupación (prev)", f"{total_cmp['ocupacion_pct']:.2f}%", delta=f"{total_n['ocupacion_pct']-total_cmp['ocupacion_pct']:.2f}%")
        d4.metric("Ingresos (prev)", f"{total_cmp['ingresos']:.2f}", delta=f"{total_n['ingresos']-total_cmp['ingresos']:.2f}")
        d5.metric("ADR (prev)", f"{total_cmp['adr']:.2f}", delta=f"{total_n['adr']-total_cmp['adr']:.2f}")
        d6.metric("RevPAR (prev)", f"{total_cmp['revpar']:.2f}", delta=f"{total_n['revpar']-total_cmp['revpar']:.2f}")

    st.divider()
    st.subheader("Detalle por alojamiento")
    if by_prop_n.empty:
        st.warning("Sin noches ocupadas en el periodo a la fecha de corte.")
    else:
        st.dataframe(by_prop_n, use_container_width=True)
        csv = by_prop_n.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Descargar detalle (CSV)", data=csv, file_name="detalle_por_alojamiento.csv", mime="text/csv")

# =============================
# MODO 2: KPIs por meses (línea) + comparación con inventario previo
# =============================
elif mode == "KPIs por meses":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        cutoff_m = st.date_input("Fecha de corte", value=date(2024, 8, 21), key="cutoff_months")
        props_m = st.multiselect("Filtrar alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="props_months")
        inv_m = st.number_input("Inventario actual (opcional)", min_value=0, value=0, step=1, key="inv_months")
        inv_m_prev = st.number_input("Inventario año anterior (opcional)", min_value=0, value=0, step=1, key="inv_months_prev")
        _min = pd.concat([raw["Fecha entrada"].dropna(), raw["Fecha salida"].dropna()]).min()
        _max = pd.concat([raw["Fecha entrada"].dropna(), raw["Fecha salida"].dropna()]).max()
        months_options = [str(p) for p in pd.period_range(_min.to_period("M"), _max.to_period("M"), freq="M")] if pd.notna(_min) and pd.notna(_max) else []
        selected_months_m = st.multiselect("Meses a graficar (YYYY-MM)", options=months_options, default=[], key="months_months")
        metric_choice = st.radio("Métrica a graficar", ["Ocupación %", "ADR (€)", "RevPAR (€)"])
        compare_m = st.checkbox("Comparar con año anterior (mismo mes)", value=False, key="cmp_months")

    st.subheader("📈 KPIs por meses (a fecha de corte)")
    if selected_months_m:
        rows_actual = []
        rows_prev = []
        for ym in selected_months_m:
            p = pd.Period(ym, freq="M")
            start_m = p.to_timestamp(how="start")
            end_m = p.to_timestamp(how="end")
            _bp, _tot = compute_kpis(
                df_all=raw,
                cutoff=pd.to_datetime(cutoff_m),
                period_start=start_m,
                period_end=end_m,
                inventory_override=int(inv_m) if inv_m > 0 else None,
                filter_props=props_m if props_m else None,
            )
            rows_actual.append({"Mes": ym, **_tot})

            if compare_m:
                p_prev = p - 12
                start_prev = p_prev.to_timestamp(how="start")
                end_prev = p_prev.to_timestamp(how="end")
                cutoff_prev = pd.to_datetime(cutoff_m) - pd.DateOffset(years=1)
                _bp2, _tot_prev = compute_kpis(
                    df_all=raw,
                    cutoff=cutoff_prev,
                    period_start=start_prev,
                    period_end=end_prev,
                    inventory_override=int(inv_m_prev) if inv_m_prev > 0 else None,
                    filter_props=props_m if props_m else None,
                )
                rows_prev.append({"Mes": ym, **_tot_prev})

        df_actual = pd.DataFrame(rows_actual).sort_values("Mes")
        if not compare_m:
            st.line_chart(df_actual.set_index("Mes")[ [ METRIC_MAP[metric_choice] ] ].rename(columns={METRIC_MAP[metric_choice]: metric_choice}), height=280)
            st.dataframe(df_actual[["Mes", "noches_ocupadas", "noches_disponibles", "ocupacion_pct", "adr", "revpar", "ingresos"]].rename(columns={
                "noches_ocupadas": "Noches ocupadas",
                "noches_disponibles": "Noches disponibles",
                "ocupacion_pct": "Ocupación %",
                "adr": "ADR (€)",
                "revpar": "RevPAR (€)",
                "ingresos": "Ingresos (€)"
            }), use_container_width=True)
        else:
            df_prev = pd.DataFrame(rows_prev).sort_values("Mes") if rows_prev else pd.DataFrame()
            plot_df = pd.DataFrame({
                "Actual": df_actual[METRIC_MAP[metric_choice]].values
            }, index=df_actual["Mes"])
            if not df_prev.empty:
                plot_df["Año anterior"] = df_prev[METRIC_MAP[metric_choice]].values
            st.line_chart(plot_df, height=280)

            table_df = df_actual.merge(df_prev, on="Mes", how="left", suffixes=("", " (prev)")) if not df_prev.empty else df_actual
            rename_map = {
                "noches_ocupadas": "Noches ocupadas",
                "noches_disponibles": "Noches disponibles",
                "ocupacion_pct": "Ocupación %",
                "adr": "ADR (€)",
                "revpar": "RevPAR (€)",
                "ingresos": "Ingresos (€)",
                "noches_ocupadas (prev)": "Noches ocupadas (prev)",
                "noches_disponibles (prev)": "Noches disponibles (prev)",
                "ocupacion_pct (prev)": "Ocupación % (prev)",
                "adr (prev)": "ADR (€) (prev)",
                "revpar (prev)": "RevPAR (€) (prev)",
                "ingresos (prev)": "Ingresos (€) (prev)",
            }
            st.dataframe(table_df.rename(columns=rename_map), use_container_width=True)

        csvm = df_actual.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Descargar KPIs por mes (CSV)", data=csvm, file_name="kpis_por_mes.csv", mime="text/csv")
    else:
        st.info("Selecciona meses en la barra lateral para ver la gráfica.")

# =============================
# MODO 3: Evolución por fecha de corte + comparación con inventario previo
# =============================
elif mode == "Evolución por fecha de corte":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Rango de corte")
        evo_cut_start = st.date_input("Inicio de corte", value=date(2024, 4, 1), key="evo_cut_start_new")
        evo_cut_end = st.date_input("Fin de corte", value=date(2024, 4, 30), key="evo_cut_end_new")

        st.header("Periodo objetivo")
        evo_target_start = st.date_input("Inicio del periodo", value=date(2024, 9, 1), key="evo_target_start_new")
        evo_target_end = st.date_input("Fin del periodo", value=date(2024, 9, 30), key="evo_target_end_new")

        props_e = st.multiselect("Filtrar alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="props_evo")
        inv_e = st.number_input("Inventario actual (opcional)", min_value=0, value=0, step=1, key="inv_evo")
        inv_e_prev = st.number_input("Inventario año anterior (opcional)", min_value=0, value=0, step=1, key="inv_evo_prev")
        metric_choice_e = st.radio("Métrica a graficar", ["Ocupación %", "ADR (€)", "RevPAR (€)"], horizontal=True, key="metric_evo")
        compare_e = st.checkbox("Comparar con año anterior (alineado por día)", value=False, key="cmp_evo")
        run_evo = st.button("Calcular evolución", type="primary", key="btn_evo")

    st.subheader("📉 Evolución de KPIs vs fecha de corte")
    if run_evo:
        cut_start_ts = pd.to_datetime(evo_cut_start)
        cut_end_ts = pd.to_datetime(evo_cut_end)
        if cut_start_ts > cut_end_ts:
            st.error("El inicio del rango de corte no puede ser posterior al fin.")
        else:
            rows_e = []
            for c in pd.date_range(cut_start_ts, cut_end_ts, freq='D'):
                _bp, tot_c = compute_kpis(
                    df_all=raw,
                    cutoff=c,
                    period_start=pd.to_datetime(evo_target_start),
                    period_end=pd.to_datetime(evo_target_end),
                    inventory_override=int(inv_e) if inv_e > 0 else None,
                    filter_props=props_e if props_e else None,
                )
                rows_e.append({"Corte": c.normalize(), **tot_c})
            df_evo = pd.DataFrame(rows_e)

            if df_evo.empty:
                st.info("No hay datos para el rango seleccionado.")
            else:
                key_col = METRIC_MAP[metric_choice_e]
                idx = pd.to_datetime(df_evo["Corte"])  # eje X con fechas reales
                plot_df = pd.DataFrame({"Actual": df_evo[key_col].values}, index=idx)

                if compare_e:
                    rows_prev = []
                    cut_start_prev = cut_start_ts - pd.DateOffset(years=1)
                    cut_end_prev = cut_end_ts - pd.DateOffset(years=1)
                    target_start_prev = pd.to_datetime(evo_target_start) - pd.DateOffset(years=1)
                    target_end_prev = pd.to_datetime(evo_target_end) - pd.DateOffset(years=1)
                    prev_dates = list(pd.date_range(cut_start_prev, cut_end_prev, freq='D'))
                    for c in prev_dates:
                        _bp2, tot_c2 = compute_kpis(
                            df_all=raw,
                            cutoff=c,
                            period_start=target_start_prev,
                            period_end=target_end_prev,
                            inventory_override=int(inv_e_prev) if inv_e_prev > 0 else None,
                            filter_props=props_e if props_e else None,
                        )
                        rows_prev.append(tot_c2[key_col])
                    prev_idx_aligned = pd.to_datetime(prev_dates) + pd.DateOffset(years=1)
                    s_prev = pd.Series(rows_prev, index=prev_idx_aligned)
                    plot_df["Año anterior"] = s_prev.reindex(idx).values

                st.line_chart(plot_df, height=300)
                st.dataframe(df_evo[["Corte", "noches_ocupadas", "noches_disponibles", "ocupacion_pct", "adr", "revpar", "ingresos"]].rename(columns={
                    "noches_ocupadas": "Noches ocupadas",
                    "noches_disponibles": "Noches disponibles",
                    "ocupacion_pct": "Ocupación %",
                    "adr": "ADR (€)",
                    "revpar": "RevPAR (€)",
                    "ingresos": "Ingresos (€)"
                }), use_container_width=True)
                csve = df_evo.to_csv(index=False).encode("utf-8-sig")
                st.download_button("📥 Descargar evolución (CSV)", data=csve, file_name="evolucion_kpis.csv", mime="text/csv")
    else:
        st.caption("Configura los parámetros en la barra lateral, luego pulsa **Calcular evolución**.")

# =============================
# MODO 4: Pickup (entre dos cortes) — con Diario/Acumulado + Δ
# =============================
elif mode == "Pickup (entre dos cortes)":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        cutA = st.date_input("Corte A", value=date(2024, 8, 14), key="pickup_cutA")
        cutB = st.date_input("Corte B", value=date(2024, 8, 21), key="pickup_cutB")
        c1, c2 = st.columns(2)
        with c1:
            p_start = st.date_input("Inicio del periodo", value=date(2024, 9, 1), key="pickup_start")
        with c2:
            p_end = st.date_input("Fin del periodo", value=date(2024, 9, 30), key="pickup_end")
        inv_pick = st.number_input("Inventario (opcional)", min_value=0, value=0, step=1, key="inv_pick")
        props_pick = st.multiselect("Filtrar alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="props_pick")
        metric_pick = st.radio("Métrica gráfica", ["Noches", "Ingresos (€)", "Ocupación %", "ADR (€)", "RevPAR (€)"], horizontal=False)
        view_pick = st.radio("Vista", ["Diario", "Acumulado"], horizontal=True)
        topn = st.number_input("Top-N alojamientos (por pickup noches)", min_value=5, max_value=100, value=20, step=5)
        run_pick = st.button("Calcular pickup", type="primary")

    st.subheader("📈 Pickup entre cortes (B – A)")
    if run_pick:
        if pd.to_datetime(cutA) > pd.to_datetime(cutB):
            st.error("Corte A no puede ser posterior a Corte B.")
        else:
            inv_override = int(inv_pick) if inv_pick > 0 else None
            # Totales A y B
            _bpA, totA = compute_kpis(raw, pd.to_datetime(cutA), pd.to_datetime(p_start), pd.to_datetime(p_end), inv_override, props_pick if props_pick else None)
            _bpB, totB = compute_kpis(raw, pd.to_datetime(cutB), pd.to_datetime(p_start), pd.to_datetime(p_end), inv_override, props_pick if props_pick else None)
            # Deltas totales
            deltas = {
                "noches": totB['noches_ocupadas'] - totA['noches_ocupadas'],
                "ingresos": totB['ingresos'] - totA['ingresos'],
                "occ_delta": totB['ocupacion_pct'] - totA['ocupacion_pct'],
                "adr_delta": totB['adr'] - totA['adr'],
                "revpar_delta": totB['revpar'] - totA['revpar'],
            }
            c1, c2, c3 = st.columns(3)
            c1.metric("Pickup Noches", f"{deltas['noches']:,}".replace(",", "."))
            c2.metric("Pickup Ingresos (€)", f"{deltas['ingresos']:.2f}")
            c3.metric("Δ Ocupación", f"{deltas['occ_delta']:.2f}%")
            c4, c5 = st.columns(2)
            c4.metric("Δ ADR", f"{deltas['adr_delta']:.2f}")
            c5.metric("Δ RevPAR", f"{deltas['revpar_delta']:.2f}")

            # Series diarias A y B
            serA = daily_series(raw, pd.to_datetime(cutA), pd.to_datetime(p_start), pd.to_datetime(p_end), props_pick if props_pick else None, inv_override)
            serB = daily_series(raw, pd.to_datetime(cutB), pd.to_datetime(p_start), pd.to_datetime(p_end), props_pick if props_pick else None, inv_override)
            # Elegir métrica
            key_map = {"Noches": "noches_ocupadas", "Ingresos (€)": "ingresos", "Ocupación %": "ocupacion_pct", "ADR (€)": "adr", "RevPAR (€)": "revpar"}
            k = key_map[metric_pick]
            df_plot = serA.merge(serB, on="Fecha", suffixes=(" A", " B"))
            df_plot["Δ (B–A)"] = df_plot[f"{k} B"] - df_plot[f"{k} A"]
            if view_pick == "Acumulado":
                for col in [f"{k} A", f"{k} B", "Δ (B–A)"]:
                    df_plot[col] = df_plot[col].cumsum()
            chart_df = pd.DataFrame({
                f"A (≤ {pd.to_datetime(cutA).date()})": df_plot[f"{k} A"].values,
                f"B (≤ {pd.to_datetime(cutB).date()})": df_plot[f"{k} B"].values,
                "Δ (B–A)": df_plot["Δ (B–A)"].values,
            }, index=pd.to_datetime(df_plot["Fecha"]))
            st.line_chart(chart_df, height=320)

            # Top-N alojamientos por pickup (noches e ingresos)
            bpA, _ = compute_kpis(raw, pd.to_datetime(cutA), pd.to_datetime(p_start), pd.to_datetime(p_end), inv_override, props_pick if props_pick else None)
            bpB, _ = compute_kpis(raw, pd.to_datetime(cutB), pd.to_datetime(p_start), pd.to_datetime(p_end), inv_override, props_pick if props_pick else None)
            merge = bpA.merge(bpB, on="Alojamiento", how="outer", suffixes=(" A", " B")).fillna(0)
            merge["Pickup noches"] = merge["Noches ocupadas B"] - merge["Noches ocupadas A"]
            merge["Pickup ingresos (€)"] = merge["Ingresos B"] - merge["Ingresos A"]
            top = merge.sort_values("Pickup noches", ascending=False).head(int(topn))
            st.subheader("🏆 Top alojamientos por pickup (noches)")
            st.dataframe(top[["Alojamiento", "Pickup noches", "Pickup ingresos (€)", "Noches ocupadas A", "Noches ocupadas B"]], use_container_width=True)

            csvp = df_plot.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Descargar detalle pickup (CSV)", data=csvp, file_name="pickup_detalle.csv", mime="text/csv")
    else:
        st.caption("Configura parámetros y pulsa **Calcular pickup**.")

# =============================
# MODO 5: Pace (curva D-0…D-max)
# =============================
elif mode == "Pace (curva D)":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        c1, c2 = st.columns(2)
        with c1:
            p_start = st.date_input("Inicio del periodo", value=date(2024, 9, 1), key="pace_start")
        with c2:
            p_end = st.date_input("Fin del periodo", value=date(2024, 9, 30), key="pace_end")
        dmax = st.slider("D máximo (días antes)", min_value=30, max_value=365, value=120, step=10)
        props_p = st.multiselect("Alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="pace_props")
        inv_p = st.number_input("Inventario (opcional)", min_value=0, value=0, step=1, key="pace_inv")
        metric_p = st.radio("Métrica", ["Ocupación %", "Noches", "Ingresos (€)", "ADR (€)", "RevPAR (€)"], horizontal=False)
        compare_yoy = st.checkbox("Comparar con año anterior", value=False)
        inv_p_prev = st.number_input("Inventario año anterior (opcional)", min_value=0, value=0, step=1, key="pace_inv_prev")
        run_p = st.button("Calcular pace", type="primary")

    st.subheader("🏁 Pace: evolución hacia la estancia (D)")
    if run_p:
        base = pace_series(raw, pd.to_datetime(p_start), pd.to_datetime(p_end), int(dmax), props_p if props_p else None, int(inv_p) if inv_p > 0 else None)
        plot_df = base.copy()
        col = METRIC_MAP.get(metric_p, None)
        if metric_p == "Noches":
            y = "noches"
        elif metric_p == "Ingresos (€)":
            y = "ingresos"
        elif col is not None:
            y = col
        else:
            y = "noches"
        plot = pd.DataFrame({"Actual": plot_df[y].values}, index=plot_df["D"])  # eje X = D

        if compare_yoy:
            p_start_prev = pd.to_datetime(p_start) - pd.DateOffset(years=1)
            p_end_prev = pd.to_datetime(p_end) - pd.DateOffset(years=1)
            prev = pace_series(raw, p_start_prev, p_end_prev, int(dmax), props_p if props_p else None, int(inv_p_prev) if inv_p_prev > 0 else None)
            plot["Año anterior"] = prev[y].values
        st.line_chart(plot, height=320)
        st.dataframe(base, use_container_width=True)
        csvpace = base.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Descargar pace (CSV)", data=csvpace, file_name="pace_curva.csv", mime="text/csv")
    else:
        st.caption("Configura parámetros y pulsa **Calcular pace**.")

# =============================
# MODO 6: Lead time & LOS
# =============================
elif mode == "Lead time & LOS":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        c1, c2 = st.columns(2)
        with c1:
            lt_start = st.date_input("Inicio periodo (por llegada)", value=date(2024, 9, 1), key="lt_start")
        with c2:
            lt_end = st.date_input("Fin periodo (por llegada)", value=date(2024, 9, 30), key="lt_end")
        props_lt = st.multiselect("Alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="lt_props")
        run_lt = st.button("Calcular", type="primary")

    st.subheader("⏱️ Lead time (por reserva) y LOS")
    if run_lt:
        df = raw.copy()
        if props_lt:
            df = df[df["Alojamiento"].isin(props_lt)]
        df = df.dropna(subset=["Fecha alta", "Fecha entrada", "Fecha salida"]) 
        # Filtro por llegada en periodo
        mask = (df["Fecha entrada"] >= pd.to_datetime(lt_start)) & (df["Fecha entrada"] <= pd.to_datetime(lt_end))
        df = df[mask]
        if df.empty:
            st.info("Sin reservas en el rango seleccionado.")
        else:
            df["lead_days"] = (df["Fecha entrada"].dt.normalize() - df["Fecha alta"].dt.normalize()).dt.days.clip(lower=0)
            df["los"] = (df["Fecha salida"].dt.normalize() - df["Fecha entrada"].dt.normalize()).dt.days.clip(lower=1)
            # Percentiles Lead
            pcts = [50, 75, 90]
            lt_p = {f"P{p}": np.percentile(df["lead_days"], p) for p in pcts}
            los_p = {f"P{p}": np.percentile(df["los"], p) for p in pcts}
            c1, c2, c3 = st.columns(3)
            c1.metric("Lead medio (d)", f"{df['lead_days'].mean():.1f}")
            c2.metric("LOS medio (noches)", f"{df['los'].mean():.1f}")
            c3.metric("Lead mediana (d)", f"{np.percentile(df['lead_days'],50):.0f}")

            # Histogramas como tablas (conteos por bins estándar)
            lt_bins = [0,3,7,14,30,60,120,1e9]
            los_bins = [1,2,3,4,5,7,10,14,21,30, np.inf]
            lt_labels = ["0-3","4-7","8-14","15-30","31-60","61-120","120+"]
            los_labels = ["1","2","3","4","5-7","8-10","11-14","15-21","22-30","30+"]
            lt_cat = pd.cut(df["lead_days"], bins=lt_bins, labels=lt_labels, right=True)
            los_cat = pd.cut(df["los"], bins=los_bins, labels=los_labels, right=True, include_lowest=True)
            lt_tab = lt_cat.value_counts().reindex(lt_labels).fillna(0).astype(int).rename_axis("Lead bin").reset_index(name="Reservas")
            los_tab = los_cat.value_counts().reindex(los_labels).fillna(0).astype(int).rename_axis("LOS bin").reset_index(name="Reservas")
            st.markdown("**Lead time (reservas)**")
            st.dataframe(lt_tab, use_container_width=True)
            st.markdown("**LOS (reservas)**")
            st.dataframe(los_tab, use_container_width=True)
            csv_lt = lt_tab.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Descargar Lead bins (CSV)", data=csv_lt, file_name="lead_bins.csv", mime="text/csv")
            csv_los = los_tab.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Descargar LOS bins (CSV)", data=csv_los, file_name="los_bins.csv", mime="text/csv")
    else:
        st.caption("Elige el rango de llegada y pulsa **Calcular**.")

# =============================
# MODO 7: DOW heatmap (periodo)
# =============================
elif mode == "DOW heatmap":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        c1, c2 = st.columns(2)
        with c1:
            h_start = st.date_input("Inicio periodo", value=date(2024, 9, 1), key="dow_start")
        with c2:
            h_end = st.date_input("Fin periodo", value=date(2024, 9, 30), key="dow_end")
        props_h = st.multiselect("Alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="dow_props")
        mode_h = st.radio("Métrica", ["Ocupación (noches)", "ADR (€)"], horizontal=True)
        cutoff_h = st.date_input("Fecha de corte", value=date(2024, 8, 21), key="dow_cutoff")
        run_h = st.button("Generar heatmap", type="primary")

    st.subheader("🗓️ Heatmap Día de la Semana × Mes")
    if run_h:
        df_cut = raw[raw["Fecha alta"] <= pd.to_datetime(cutoff_h)].copy()
        if props_h:
            df_cut = df_cut[df_cut["Alojamiento"].isin(props_h)]
        df_cut = df_cut.dropna(subset=["Fecha entrada", "Fecha salida"]) 
        rows = []
        for _, r in df_cut.iterrows():
            e, s, p = r["Fecha entrada"], r["Fecha salida"], float(r["Precio"])
            ov_start = max(e, pd.to_datetime(h_start))
            ov_end = min(s, pd.to_datetime(h_end) + pd.Timedelta(days=1))
            n_nights = (s - e).days
            if ov_start >= ov_end or n_nights <= 0:
                continue
            adr_night = p / n_nights if n_nights > 0 else 0.0
            for d in pd.date_range(ov_start, ov_end - pd.Timedelta(days=1), freq='D'):
                rows.append({"Mes": d.strftime('%Y-%m'), "DOW": ("Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo")[d.weekday()], "Noches": 1, "ADR": adr_night})
        if not rows:
            st.info("Sin datos en el rango.")
        else:
            df_n = pd.DataFrame(rows)
            if mode_h.startswith("Ocupación"):
                piv = df_n.pivot_table(index="DOW", columns="Mes", values="Noches", aggfunc='sum', fill_value=0)
            else:
                piv = df_n.pivot_table(index="DOW", columns="Mes", values="ADR", aggfunc='mean', fill_value=0.0)
            # Orden DOW
            dow_order = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]
            piv = piv.reindex(dow_order)
            st.dataframe(piv, use_container_width=True)
            csvh = piv.reset_index().to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Descargar heatmap (CSV)", data=csvh, file_name="dow_heatmap.csv", mime="text/csv")
    else:
        st.caption("Elige periodo y pulsa **Generar heatmap**.")

# =============================
# MODO 8: ADR bands & Targets
# =============================
elif mode == "ADR bands & Targets":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros ADR bands")
        ab_cutoff = st.date_input("Fecha de corte", value=date(2024, 8, 21), key="ab_cutoff")
        c1, c2 = st.columns(2)
        with c1:
            ab_start = st.date_input("Inicio periodo", value=date(2024, 9, 1), key="ab_start")
        with c2:
            ab_end = st.date_input("Fin periodo", value=date(2024, 9, 30), key="ab_end")
        props_ab = st.multiselect("Alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="ab_props")
        run_ab = st.button("Calcular ADR bands", type="primary")

    st.subheader("📦 Bandas de ADR (percentiles por mes)")
    if run_ab:
        df = raw[raw["Fecha alta"] <= pd.to_datetime(ab_cutoff)].copy()
        if props_ab:
            df = df[df["Alojamiento"].isin(props_ab)]
        df = df.dropna(subset=["Fecha entrada", "Fecha salida"]) 
        df["los"] = (df["Fecha salida"].dt.normalize() - df["Fecha entrada"].dt.normalize()).dt.days.clip(lower=1)
        df["adr_reserva"] = df["Precio"] / df["los"]
        # Filtrar por periodo (por estancia que intersecta)
        ov_start = pd.to_datetime(ab_start)
        ov_end = pd.to_datetime(ab_end) + pd.Timedelta(days=1)
        mask = ~((df["Fecha salida"] <= ov_start) | (df["Fecha entrada"] >= ov_end))
        df = df[mask]
        if df.empty:
            st.info("Sin reservas en el rango.")
        else:
            df["Mes"] = df["Fecha entrada"].dt.to_period('M').astype(str)
            def pct_cols(x):
                arr = x.dropna().values
                if arr.size == 0:
                    return pd.Series({"P10": 0.0, "P25": 0.0, "Mediana": 0.0, "P75": 0.0, "P90": 0.0})
                return pd.Series({
                    "P10": np.percentile(arr, 10),
                    "P25": np.percentile(arr, 25),
                    "Mediana": np.percentile(arr, 50),
                    "P75": np.percentile(arr, 75),
                    "P90": np.percentile(arr, 90),
                })
            bands = df.groupby("Mes")["adr_reserva"].apply(pct_cols).reset_index()
            bands_wide = bands.pivot(index="Mes", columns="level_1", values="adr_reserva").sort_index()
            st.dataframe(bands_wide, use_container_width=True)
            # Pequeña gráfica de P10/Mediana/P90
            plot = bands_wide[["P10","Mediana","P90"]]
            st.line_chart(plot, height=300)
            csvb = bands_wide.reset_index().to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Descargar ADR bands (CSV)", data=csvb, file_name="adr_bands.csv", mime="text/csv")

    st.divider()
    st.subheader("🎯 Targets vs Real vs LY (opcional)")
    tgts = st.session_state.get("targets_df")
    if tgts is None:
        st.info("Carga un CSV de targets en la barra lateral (dentro del acordeón 🎯).")
    else:
        with st.sidebar:
            t_cutoff = st.date_input("Fecha de corte para 'Real'", value=date(2024, 8, 21), key="tgt_cutoff")
            months_sel = st.multiselect("Meses (YYYY-MM)", options=sorted(tgts.apply(lambda r: f"{int(r['year']):04d}-{int(r['month']):02d}", axis=1).unique().tolist()))
            inv_now = st.number_input("Inventario actual (opcional)", min_value=0, value=0, step=1, key="tgt_inv")
            inv_ly = st.number_input("Inventario LY (opcional)", min_value=0, value=0, step=1, key="tgt_inv_ly")
        if months_sel:
            rows = []
            for ym in months_sel:
                y, m = map(int, ym.split('-'))
                p = pd.Period(ym, freq='M')
                p_start = p.to_timestamp(how='start')
                p_end = p.to_timestamp(how='end')
                # Real
                _bp, real = compute_kpis(raw, pd.to_datetime(t_cutoff), p_start, p_end, int(inv_now) if inv_now>0 else None, None)
                # LY
                p_prev = p - 12
                _bp2, ly = compute_kpis(raw, pd.to_datetime(t_cutoff) - pd.DateOffset(years=1), p_prev.to_timestamp('M', 'start'), p_prev.to_timestamp('M', 'end'), int(inv_ly) if inv_ly>0 else None, None)
                # Target
                trow = tgts[(tgts['year']==y) & (tgts['month']==m)]
                tgt_occ = float(trow['target_occ_pct'].iloc[0]) if 'target_occ_pct' in tgts.columns and not trow.empty else np.nan
                tgt_adr = float(trow['target_adr'].iloc[0]) if 'target_adr' in tgts.columns and not trow.empty else np.nan
                tgt_revpar = float(trow['target_revpar'].iloc[0]) if 'target_revpar' in tgts.columns and not trow.empty else np.nan
                rows.append({
                    "Mes": ym,
                    "Occ Real %": real['ocupacion_pct'],
                    "Occ LY %": ly['ocupacion_pct'],
                    "Occ Target %": tgt_occ,
                    "ADR Real": real['adr'],
                    "ADR LY": ly['adr'],
                    "ADR Target": tgt_adr,
                    "RevPAR Real": real['revpar'],
                    "RevPAR LY": ly['revpar'],
                    "RevPAR Target": tgt_revpar,
                })
            df_t = pd.DataFrame(rows).set_index("Mes")
            st.dataframe(df_t, use_container_width=True)
            st.line_chart(df_t[["Occ Real %","Occ LY %","Occ Target %"]].dropna(), height=280)

# =============================
# MODO 9: Calendario por alojamiento (heatmap simple)
# =============================
elif mode == "Calendario por alojamiento":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        cutoff_cal = st.date_input("Fecha de corte", value=date(2024, 8, 21), key="cal_cutoff")
        c1, c2 = st.columns(2)
        with c1:
            cal_start = st.date_input("Inicio periodo", value=date(2024, 9, 1), key="cal_start")
        with c2:
            cal_end = st.date_input("Fin periodo", value=date(2024, 9, 30), key="cal_end")
        props_cal = st.multiselect("Alojamientos", options=sorted(raw["Alojamiento"].unique()), default=[], key="cal_props")
        mode_cal = st.radio("Modo", ["Ocupado/Libre", "ADR"], horizontal=True, key="cal_mode")
        run_cal = st.button("Generar calendario", type="primary", key="btn_cal")

    st.subheader("🗓️ Calendario por alojamiento")
    if run_cal:
        if pd.to_datetime(cal_start) > pd.to_datetime(cal_end):
            st.error("El inicio del periodo no puede ser posterior al fin.")
        else:
            piv = build_calendar_matrix(
                df_all=raw,
                cutoff=pd.to_datetime(cutoff_cal),
                start=pd.to_datetime(cal_start),
                end=pd.to_datetime(cal_end),
                props=props_cal if props_cal else None,
                mode=mode_cal,
            )
            if piv.empty:
                st.info("Sin datos para los filtros seleccionados.")
            else:
                piv.columns = [c.strftime('%Y-%m-%d') if isinstance(c, (pd.Timestamp, datetime, date)) else str(c) for c in piv.columns]
                st.dataframe(piv, use_container_width=True)
                csvc = piv.reset_index().to_csv(index=False).encode("utf-8-sig")
                st.download_button("📥 Descargar calendario (CSV)", data=csvc, file_name="calendario_alojamientos.csv", mime="text/csv")
    else:
        st.caption("Elige parámetros y pulsa **Generar calendario**.")
