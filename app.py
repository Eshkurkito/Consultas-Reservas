import io
import math
from datetime import datetime, timedelta, date
from typing import List, Tuple

import pandas as pd
import streamlit as st

# =============================
# Utilidades
# =============================

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Fecha alta", "Fecha entrada", "Fecha salida"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            st.error(f"Falta la columna obligatoria: {col}")
            st.stop()
    return df


def load_excel(files: List[io.BytesIO]) -> pd.DataFrame:
    frames = []
    for f in files:
        try:
            xls = pd.ExcelFile(f)
            sheet = "Estado de pagos de las reservas" if "Estado de pagos de las reservas" in xls.sheet_names else xls.sheet_names[0]
            df = pd.read_excel(xls, sheet_name=sheet)
            frames.append(df)
        except Exception as e:
            st.error(f"No se pudo leer un archivo: {e}")
            st.stop()
    if not frames:
        return pd.DataFrame()
    df_all = pd.concat(frames, ignore_index=True)
    # Normalizar columnas esperadas
    expected = ["Alojamiento", "Fecha alta", "Fecha entrada", "Fecha salida", "Precio"]
    missing = [c for c in expected if c not in df_all.columns]
    if missing:
        st.error(f"Faltan columnas requeridas: {', '.join(missing)}")
        st.stop()
    # Tipos y limpieza básica
    df_all = parse_dates(df_all)
    df_all["Alojamiento"] = df_all["Alojamiento"].astype(str).str.strip()
    # Forzar numérico en Precio
    df_all["Precio"] = pd.to_numeric(df_all["Precio"], errors="coerce").fillna(0.0)
    return df_all


def expand_reservations(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Devuelve filas a nivel-noche solo para noches dentro del periodo [start, end].
    end es inclusivo (ej. 01–30/09 incluye el 30).
    """
    rows = []
    for idx, r in df.iterrows():
        entrada, salida = r["Fecha entrada"], r["Fecha salida"]
        if pd.isna(entrada) or pd.isna(salida):
            continue
        # rango de noches ocupadas por reserva (check-out no cuenta)
        noches = pd.date_range(entrada, salida - timedelta(days=1), freq="D")
        if len(noches) == 0:
            continue
        # Intersección con el periodo
        for n in noches:
            if start <= n <= end:
                rows.append({
                    "Alojamiento": r["Alojamiento"],
                    "Fecha": n,
                    "Precio": r["Precio"],
                    "Total noches reserva": len(noches),
                })
    if not rows:
        return pd.DataFrame(columns=["Alojamiento", "Fecha", "Precio", "Total noches reserva"])
    return pd.DataFrame(rows)


def compute_kpis(df_all: pd.DataFrame,
                 cutoff: pd.Timestamp,
                 period_start: pd.Timestamp,
                 period_end: pd.Timestamp,
                 inventory_override: int | None = None,
                 filter_props: List[str] | None = None) -> Tuple[pd.DataFrame, dict]:
    """Calcula KPIs totales y por alojamiento a una fecha de corte.
    - df_all: reservas de uno o varios años con las columnas requeridas.
    - cutoff: incluir solo reservas con Fecha alta <= cutoff.
    - period_start/period_end: rango analizado (inclusive ambas).
    - inventory_override: si se especifica, reemplaza el número de alojamientos disponibles para el denominador.
    - filter_props: lista opcional de alojamientos a incluir (si None, se usan todos).
    """
    # 1) Filtrar por corte
    df_cut = df_all[df_all["Fecha alta"] <= cutoff].copy()
    if filter_props:
        df_cut = df_cut[df_cut["Alojamiento"].isin(filter_props)]

    # 2) Expandir a nivel-noche solo dentro del periodo
    nightly = expand_reservations(df_cut, period_start, period_end)

    # 3) KPI por alojamiento
    if nightly.empty:
        # Preparar estructura vacía
        total_props = len(filter_props) if filter_props else df_all["Alojamiento"].nunique()
        if inventory_override is not None:
            total_props = inventory_override
        days = (period_end - period_start).days + 1
        nights_avail = total_props * days
        total = {
            "noches_ocupadas": 0,
            "noches_disponibles": nights_avail,
            "ocupacion_pct": 0.0,
            "ingresos": 0.0,
            "adr": 0.0,
            "revpar": 0.0,
        }
        return pd.DataFrame(columns=["Alojamiento", "Noches ocupadas", "Ingresos", "ADR"]), total

    nightly["Ingreso proporcional"] = nightly["Precio"] / nightly["Total noches reserva"]

    by_prop = nightly.groupby("Alojamiento").agg(
        **{
            "Noches ocupadas": ("Fecha", "count"),
            "Ingresos": ("Ingreso proporcional", "sum")
        }
    ).reset_index()
    by_prop["ADR"] = by_prop["Ingresos"] / by_prop["Noches ocupadas"]

    # 4) KPI totales
    noches_ocupadas = int(by_prop["Noches ocupadas"].sum())
    ingresos = float(by_prop["Ingresos"].sum())
    adr = float(ingresos / noches_ocupadas) if noches_ocupadas > 0 else 0.0

    # Inventario disponible
    if filter_props:
        inventario = len(set(filter_props))
    else:
        inventario = df_all["Alojamiento"].nunique()

    if inventory_override is not None and inventory_override > 0:
        inventario = inventory_override

    days = (period_end - period_start).days + 1
    noches_disponibles = inventario * days
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

    return by_prop.sort_values("Alojamiento"), tot


# =============================
# UI Streamlit
# =============================

st.set_page_config(page_title="Consultas OTB por corte", layout="wide")

st.title("📅 Consultas de ocupación, ADR y RevPAR a fecha de corte")
st.caption("Carga tus ficheros de reservas, elige una fecha de corte y un periodo. El cálculo prorratea ingresos por noche y cuenta solo reservas con Fecha alta ≤ corte.")

with st.sidebar:
    st.header("1) Cargar archivos")
    files = st.file_uploader(
        "Arrastra uno o varios Excel (2024, 2025, etc.)",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
    )

    st.divider()
    st.header("2) Parámetros de consulta")
    cutoff = st.date_input("Fecha de corte", value=date(2024, 8, 21))

    colp1, colp2 = st.columns(2)
    with colp1:
        period_start = st.date_input("Inicio del periodo", value=date(2024, 9, 1))
    with colp2:
        period_end = st.date_input("Fin del periodo", value=date(2024, 9, 30))

    st.caption("El fin es inclusivo: del 1 al 30 incluye el 30.")

    st.divider()
    st.header("3) Opciones avanzadas")
    inventory_override = st.number_input(
        "Sobrescribir inventario (nº alojamientos)", min_value=0, value=0, step=1,
        help="Déjalo a 0 para usar el número de alojamientos únicos detectados en los ficheros cargados."
    )

# Cargar datos
if not files:
    st.info("➡️ Sube al menos un archivo para comenzar.")
    st.stop()

raw = load_excel(files)

# Selector de propiedades
all_props = sorted(raw["Alojamiento"].dropna().astype(str).unique().tolist())
with st.sidebar:
    st.header("4) Filtrado de alojamientos (opcional)")
    selected = st.multiselect(
        "Elige alojamientos a incluir (vacío = todos)",
        options=all_props,
        default=[],
    )

# Ejecutar
cutoff_ts = pd.to_datetime(cutoff)
start_ts = pd.to_datetime(period_start)
end_ts = pd.to_datetime(period_end)

inv_override = int(inventory_override) if inventory_override and inventory_override > 0 else None

by_prop, total = compute_kpis(
    df_all=raw,
    cutoff=cutoff_ts,
    period_start=start_ts,
    period_end=end_ts,
    inventory_override=inv_override,
    filter_props=selected if selected else None,
)

# =============================
# Salida
# =============================

st.subheader("Resultados totales")
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

col1.metric("Noches ocupadas", f"{total['noches_ocupadas']:,}".replace(",", "."))
col2.metric("Noches disponibles", f"{total['noches_disponibles']:,}".replace(",", "."))
col3.metric("Ocupación", f"{total['ocupacion_pct']:.2f}%")
col4.metric("Ingresos (€)", f"{total['ingresos']:.2f}")
col5.metric("ADR (€)", f"{total['adr']:.2f}")
col6.metric("RevPAR (€)", f"{total['revpar']:.2f}")

st.divider()

st.subheader("Detalle por alojamiento")
if by_prop.empty:
    st.warning("Sin noches ocupadas en el periodo a la fecha de corte.")
else:
    # Formateo visual (no altera tipo numérico)
    df_view = by_prop.copy()
    st.dataframe(df_view, use_container_width=True)

    # Descargar CSV
    csv = df_view.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 Descargar detalle por alojamiento (CSV)", data=csv, file_name="detalle_por_alojamiento.csv", mime="text/csv")

# Nota metodológica
with st.expander("📎 Metodología"):
    st.markdown(
        """
        **Cálculos:**
        - Se incluyen solo reservas con **Fecha alta ≤ Fecha de corte**.
        - Se expanden noches por reserva y se **prorratea el ingreso** (Precio / nº de noches de la reserva) a cada noche.
        - **ADR** = Ingresos del periodo / Noches ocupadas del periodo.
        - **Ocupación** = Noches ocupadas / (Inventario × días del periodo).
        - **RevPAR** = Ingresos del periodo / Noches disponibles.
        - El **Inventario** por defecto es el nº de alojamientos únicos detectados (o el valor que indiques en "Sobrescribir inventario").
        """
    )
