"""
Labor Rate Lookup - CBI Tool
Data is read from Azure Blob Storage via AZURE_STORAGE_CONNECTION_STRING.
"""

import os
import re
import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

# Azure Blob Storage — set these as environment variables in Azure App Service
# AZURE_STORAGE_CONNECTION_STRING : full connection string from the storage account
# AZURE_CONTAINER                 : blob container name   (default: processed)
# AZURE_BLOB_PATH                 : Delta table path      (default: final_merged_labor_rates)
AZURE_CONTAINER = os.environ.get('AZURE_CONTAINER', 'processed')
AZURE_BLOB_PATH = os.environ.get('AZURE_BLOB_PATH',  'final_merged_labor_rates')

DEFAULT_COLUMNS = [
    'POSITION', 'LABOR_TYPE', 'TRADE_TIER', 'SENIORITY_LEVEL',
    'TIME', 'CITY', 'STATE', 'COUNTRY',
    'BUILDING_TYPE', 'OWNER', 'CURRENCY', 'BILL_RATE', 'DATE',
]

BREAKDOWN_COLS = [
    'BASE',
    'FICA', 'FUTA', 'SUTA',
    'WORK_COMP', 'LIABILITY_INS', 'TAX_INS',
    'FRINGE_BENEFITS', 'PER_DIEM', 'SMALL_TOOLS',
    'OT', 'OTHER_BURDEN', 'G_AND_A_OH', 'PROFIT',
    'BILL_RATE',
]

# Columns with a known fixed set of values — rendered as multiselect (no operator needed)
FIXED_OPTIONS = {
    'SOURCE':                ['INTERNATIONAL', 'US'],
    'LABOR_TYPE':            ['TRADE', 'SUPERVISION'],
    'TRADE_TIER':            [
        '1-GENERAL FOREMAN', '2-FOREMAN', '3-JOURNEYMAN', '4-APPRENTICE',
        'HELPER', 'PRE-APPRENTICE', 'MASTER',
        'CE1', 'CE2', 'CE3', 'CW1', 'CW2', 'CW3', 'CW4',
        'CREW', 'ADMINISTRATIVE',
    ],
    'SENIORITY_LEVEL': sorted([
        'ASSISTANT', 'ASSISTANT MANAGER', 'ASSOCIATE', 'ASSOCIATE PRINCIPAL', 'CHIEF',
        'COORDINATOR', 'DIRECTOR', 'DOCTOR', 'ENGINEER', 'EXPEDITOR', 'INTERN', 'JUNIOR',
        'LEAD', 'MANAGER', 'MEDIC', 'MID', 'NURSE', 'PRINCIPAL', 'PROFESSIONAL',
        'REGIONAL', 'REGIONAL MANAGER', 'SENIOR', 'SENIOR ASSOCIATE', 'SENIOR MANAGER',
        'SENIOR PRINCIPAL', 'SENIOR SPECIALIST', 'SENIOR SUPERVISOR',
        'SENIOR VICE PRESIDENT', 'SPECIALIST', 'SUPERVISOR', 'VICE PRESIDENT',
        'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII',
    ]),
    'WORKER_ORIGIN':         ['LOCAL', 'INTERNATIONAL', 'TRAVELER'],
    'WORKER_CLASSIFICATION': ['UNION', 'OPEN SHOP', 'GC PM', 'OWNER PM', 'UNKNOWN'],
    'TIME':                  ['ST', 'OT', 'DT'],
    'CONTRACTOR_TYPE':       ['GC/CM', 'SUB', 'OWNER TEAM'],
    'WAGE_TYPE':             ['BURDENED', 'NON-BURDENED', 'NO OHP-BURDENED',
                              'CREW RATE-BURDENED', 'CREW RATE-UNBURDENED'],
    'BUILDING_TYPE': sorted([
        'AVIATION', 'DATA CENTER', 'EDUCATION', 'GOVERNMENT', 'HOSPITALITY',
        'LAB/RESEARCH', 'MANUFACTURING', 'MANUFACTURING-AUTOMOTIVE',
        'MANUFACTURING-F&B', 'MANUFACTURING-PHARMA', 'MULTIFAMILY',
        'OFFICE-COMMERCIAL', 'OFFICE-PHARMA', 'OFFICE-SEMICONDUCTOR',
        'OTHER', 'RETAIL', 'SEMICONDUCTOR', 'STADIUMS', 'UNKNOWN',
    ]),
    'REGION':    ['AMERICAS', 'ASIA PACIFIC', 'UK AND EUROPE', 'MIDDLE EAST', 'ANTARCTICA'],
    'CONFIRMED': ['BID', 'RESEARCHED'],
    'POSITION': sorted([
        'ACCOUNT MANAGER', 'ACCOUNTANT', 'ADMIN', 'ANALYST',
        'ASBESTOS, INSULATION, PIPE COVERERS', 'BIM', 'BILLING', 'BOILERMAKERS',
        'BRICKLAYERS', 'CAD', 'CARPENTERS', 'CARPET AND LINOLEUM LAYERS',
        'CEMENT MASON', 'COMMISSIONING', 'CONSTRUCTION MANAGER', 'CONSULTANT',
        'CONTRACTS', 'CONTROLS', 'COORDINATOR', 'DESIGN', 'ELECTRICIANS',
        'ELEVATOR MECHANIC', 'ENGINEER', 'EQUIPMENT OPERATOR: CRANE', 'ESTIMATOR',
        'FOOD SERVICE', 'GENERAL LABORER', 'GLAZIERS', 'HR', 'HVAC TECHNICIANS',
        'IT AND SOFTWARE', 'INTERN', 'IRONWORKER', 'LABORER',
        'LIGHT EQUIPMENT OPERATOR', 'MANAGER', 'MED EQUIPMENT OPERATOR', 'MEDICAL',
        'MILLWRIGHTS', 'OFFICE MANAGER', 'OPERATIONS MANAGER', 'PAINTER',
        'PILE DRIVERS', 'PIPE FITTER', 'PLASTERER',
        'PLUMBERS, PIPEFITTERS, STEAMFITTERS, SPRINKLER INSTALLERS',
        'PRECONSTRUCTION', 'PROCUREMENT', 'PROCUREMENT MANAGER',
        'PROGRAM COORDINATOR', 'PROGRAM DIRECTOR', 'PROGRAM MANAGER', 'PROGRAMMER',
        'PROJECT COORDINATOR', 'PROJECT DIRECTOR', 'PROJECT EXECUTIVE',
        'PROJECT MANAGER', 'PROJECT PRINCIPAL', 'PURCHASING', 'QC', 'RIGGING',
        'ROOFER', 'SAFETY', 'SCHEDULER', 'SECRETARY', 'SHEET METAL WORKERS',
        'SITE UTILITIES', 'SITEWORK', 'STONE MASON', 'SUPERINTENDENT', 'SUPPORT',
        'SURVEYOR', 'SUSTAINABILITY', 'TEAMSTER', 'TECHNICIAN', 'TECHNOLOGY',
        'TILE LAYERS', 'WINDOW COVERINGS',
    ]),
}


def _delta_current_files(container_client, table_prefix):
    """Read _delta_log to return only the parquet paths in the current snapshot."""
    import json
    log_prefix = f"{table_prefix}/_delta_log/"
    added, removed = {}, set()
    log_blobs = sorted(
        b.name for b in container_client.list_blobs(name_starts_with=log_prefix)
        if b.name.endswith('.json')
    )
    for blob_name in log_blobs:
        raw = container_client.get_blob_client(blob_name).download_blob().readall()
        for line in raw.decode('utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if 'add' in entry:
                added[entry['add']['path']] = True
            if 'remove' in entry:
                removed.add(entry['remove']['path'])
    return {p for p in added if p not in removed}


@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    conn_str = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
    if not conn_str:
        st.error("AZURE_STORAGE_CONNECTION_STRING environment variable is not set.")
        st.stop()

    from azure.storage.blob import BlobServiceClient
    client = BlobServiceClient.from_connection_string(conn_str)
    container_client = client.get_container_client(AZURE_CONTAINER)
    table_prefix = AZURE_BLOB_PATH.rstrip('/')

    # Read the Delta log to find which parquet files are in the current snapshot.
    # Overwrites mark old files as "remove", so stale parts are always skipped.
    current_files = _delta_current_files(container_client, table_prefix)
    if not current_files:
        st.error(f"Delta log at {AZURE_CONTAINER}/{table_prefix} resolved no active parquet files.")
        st.stop()

    frames = []
    for rel_path in sorted(current_files):
        blob_name = f"{table_prefix}/{rel_path}"
        raw = container_client.get_blob_client(blob_name).download_blob().readall()
        frames.append(pd.read_parquet(BytesIO(raw)))

    df = pd.concat(frames, ignore_index=True)

    dates = pd.to_datetime(df['DATE'], errors='coerce')
    df.insert(
        df.columns.get_loc('DATE') + 1,
        'YEAR',
        dates.dt.year.astype('Int64'),
    )
    df['DATE'] = dates.dt.strftime('%Y-%m-%d')
    return df


def apply_filter(series, op, val):
    val = val.strip()
    s = series.astype(str)
    if op == 'contains':
        return s.str.contains(re.escape(val), case=False, na=False)
    elif op == 'equals':
        return s.str.lower() == val.lower()
    elif op == 'starts with':
        return s.str.lower().str.startswith(val.lower())
    elif op == 'ends with':
        return s.str.lower().str.endswith(val.lower())
    elif op == 'pattern (use % wildcard)':
        # Split on %, escape each literal segment, rejoin with .*
        parts = [re.escape(p) for p in val.split('%')]
        pattern = '.*'.join(parts)
        return s.str.contains(f'^{pattern}$', case=False, na=False, regex=True)
    elif op in ('>', '<', '>=', '<=', '= (number)'):
        numeric = pd.to_numeric(series, errors='coerce')
        try:
            n = float(val)
        except ValueError:
            return pd.Series([False] * len(series), index=series.index)
        ops = {'>': numeric.__gt__, '<': numeric.__lt__,
               '>=': numeric.__ge__, '<=': numeric.__le__,
               '= (number)': numeric.__eq__}
        return ops[op](n).fillna(False)
    return pd.Series([True] * len(series), index=series.index)


def filter_dataframe(df, filters):
    """Apply AND/OR filter logic matching the Cortex app's group-then-OR approach."""
    clauses = []
    for f in filters:
        col = f.get('column', '')
        op  = f.get('operator', 'contains')
        val = f.get('value', '')
        connector = f.get('connector', 'AND')
        if not col or col not in df.columns:
            continue

        # Fixed-option columns store a list; free-text columns store a string
        if isinstance(val, list):
            if not val:  # empty selection = no filter
                continue
            mask = df[col].astype(str).isin([str(v) for v in val])
        else:
            if not val:
                continue
            mask = apply_filter(df[col], op, val)

        clauses.append((mask, connector))

    if not clauses:
        return df

    # Build consecutive AND groups, OR between groups
    groups = []
    current = [clauses[0][0]]
    for i in range(1, len(clauses)):
        if clauses[i - 1][1] == 'AND':
            current.append(clauses[i][0])
        else:
            groups.append(current)
            current = [clauses[i][0]]
    groups.append(current)

    group_masks = []
    for g in groups:
        combined = g[0]
        for m in g[1:]:
            combined = combined & m
        group_masks.append(combined)

    final = group_masks[0]
    for m in group_masks[1:]:
        final = final | m

    return df[final]


def to_excel(df):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Labor Rates')
    buf.seek(0)
    return buf


# ── App ────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Labor Rate Lookup", page_icon="👷", layout="wide")

    st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] { background: #f7f5f3; }
        [data-testid="stSidebar"] { background: #3b1f52; }
        [data-testid="stSidebar"] * { color: #e3dedb !important; }
        [data-testid="stSidebar"] .stMultiSelect > div { background: #52297a !important; }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1 { color: #3b1f52 !important; font-size: 1.6rem !important; }
        h2, h3 { color: #3b1f52 !important; }
        .stButton > button {
            background: #3b1f52; color: #e3dedb;
            border: none; border-radius: 6px; font-weight: 500;
        }
        .stButton > button:hover { background: #52297a; color: white; }
        .stDownloadButton > button {
            background: #e3dedb; color: #3b1f52;
            border: 1px solid #3b1f52; border-radius: 6px; font-weight: 500;
        }
        div[data-testid="stExpander"] {
            border: 1px solid #d4cfc9; border-radius: 8px; background: white;
        }
        [data-testid="metric-container"] { background: white; border-radius: 8px; padding: 0.5rem 1rem; border: 1px solid #d4cfc9; }
    </style>
    """, unsafe_allow_html=True)

    st.title("👷 Labor Rate Lookup")
    

    # ── Session state ──────────────────────────────────────────────────────────
    if 'filters' not in st.session_state:
        st.session_state.filters = [
            {'column': 'POSITION', 'operator': 'contains', 'value': '', 'connector': 'AND'}
        ]
    if 'selected_columns' not in st.session_state:
        st.session_state.selected_columns = DEFAULT_COLUMNS.copy()

    for f in st.session_state.filters:
        if 'connector' not in f:
            f['connector'] = 'AND'

    # ── Load data ──────────────────────────────────────────────────────────────
    with st.spinner("Loading data..."):
        df = load_data()

    all_columns = list(df.columns)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🗂️ Visible Columns")
        st.caption("Changes apply immediately when you click a column.")

        current = [c for c in st.session_state.selected_columns if c in all_columns]
        selected = st.multiselect(
            "Select columns to show:",
            options=all_columns,
            default=current,
            key="col_multiselect",
        )
        st.session_state.selected_columns = selected

        if st.button("↩ Reset to default", key="reset_cols", use_container_width=True):
            st.session_state.selected_columns = DEFAULT_COLUMNS.copy()
            st.rerun()

        st.divider()

        st.markdown(f"**Total records:** {len(df):,}")
        st.markdown("**Records by region:**")
        for region, count in df['REGION'].value_counts().items():
            st.caption(f"• {region}: {count:,}")

        st.divider()

        if st.button("🔄 Clear cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ── Filters ────────────────────────────────────────────────────────────────
    st.markdown("### Filters")
    st.caption(
        "**AND** = all conditions must match (narrows results).  "
        "**OR** = any condition can match (broadens results)."
    )

    with st.expander("💡 Not sure which operator to use? Click here.", expanded=False):
        st.markdown("""
#### The everyday operators

| Operator | Use it when... | Example input | What you get |
|---|---|---|---|
| **contains** | Word appears anywhere in the value | `electrician` | "ELECTRICIANS", "ELECTRICIAN FOREMAN" |
| **equals** | Exact full match only | `USA` | Rows where the column is exactly "USA" |
| **starts with** | Value begins with your text | `PLUMB` | "PLUMBERS", "PLUMBING TECH" |
| **ends with** | Value ends with your text | `FOREMAN` | "GENERAL FOREMAN", "2-FOREMAN" |
| **> < >= <=** | Numeric comparison | `100` on BILL_RATE with `>` | All rows with bill rate over 100 |
| **= (number)** | Exact numeric match | `2026` on DATE | Rows where the column equals 2026 |

---

#### The pattern operator — for multi-word searches

`%` means **"anything can go here"** — words, spaces, or nothing.

| You type | Matches |
|---|---|
| `%PIPE%` | "PLUMBERS, PIPEFITTERS, STEAMFITTERS..." |
| `%GENERAL%FOREMAN%` | "1-GENERAL FOREMAN" |
| `%SHEET%METAL%` | "SHEET METAL WORKERS" |

> **Tip:** `%word1%word2%` finds rows where word1 appears *before* word2 — useful when you know two parts of a value but not the exact wording.
        """)

    text_operators   = ['contains', 'equals', 'starts with', 'ends with', 'pattern (use % wildcard)']
    number_operators = ['>', '<', '>=', '<=', '= (number)']
    all_operators    = text_operators + number_operators

    col_opts = [''] + all_columns

    filters_to_remove = []
    for i, f in enumerate(st.session_state.filters):

        # Connector row between filters
        if i > 0:
            cc1, _, __ = st.columns([1, 1, 8])
            with cc1:
                prev_conn = st.session_state.filters[i - 1].get('connector', 'AND')
                connector_val = st.selectbox(
                    "connector",
                    options=['AND', 'OR'],
                    index=0 if prev_conn == 'AND' else 1,
                    key=f"conn_{i}",
                    label_visibility="collapsed",
                )
                st.session_state.filters[i - 1]['connector'] = connector_val

        # Read the column value from widget state so is_fixed is correct
        # in the SAME render cycle as the user's selection (not one cycle late)
        cur_col  = st.session_state.get(f"fc_{i}", f.get('column', ''))
        f['column'] = cur_col
        is_fixed = cur_col in FIXED_OPTIONS

        # Normalise stored value type and purge the stale widget key
        # so Streamlit never tries to reuse the same key for two different widget types
        if is_fixed and not isinstance(f.get('value'), list):
            f['value'] = []
            st.session_state.pop(f"fv_fix_{i}", None)   # reset stale multiselect state
        elif not is_fixed and isinstance(f.get('value'), list):
            f['value'] = ''
            st.session_state.pop(f"fv_txt_{i}", None)   # reset stale text_input state

        if is_fixed:
            # Fixed-option column: multiselect only, no operator dropdown
            c1, c2, c4 = st.columns([3, 5, 1])
            with c1:
                f['column'] = st.selectbox(
                    "Column",
                    options=col_opts,
                    index=col_opts.index(cur_col) if cur_col in col_opts else 0,
                    key=f"fc_{i}",
                    label_visibility="collapsed",
                )
            with c2:
                safe_default = [v for v in f.get('value', []) if v in FIXED_OPTIONS.get(f['column'], [])]
                f['value'] = st.multiselect(
                    "Value",
                    options=FIXED_OPTIONS.get(f['column'], []),
                    default=safe_default,
                    key=f"fv_fix_{i}",       # type-specific key — never conflicts with text_input
                    label_visibility="collapsed",
                    placeholder="Select one or more…",
                )
        else:
            # Free-text / numeric column: operator dropdown + text input
            c1, c2, c3, c4 = st.columns([3, 2, 3, 1])
            with c1:
                f['column'] = st.selectbox(
                    "Column",
                    options=col_opts,
                    index=col_opts.index(cur_col) if cur_col in col_opts else 0,
                    key=f"fc_{i}",
                    label_visibility="collapsed",
                )
            with c2:
                cur_op = f.get('operator', 'contains')
                f['operator'] = st.selectbox(
                    "Operator",
                    options=all_operators,
                    index=all_operators.index(cur_op) if cur_op in all_operators else 0,
                    key=f"fo_{i}",
                    label_visibility="collapsed",
                )
            with c3:
                cur_text = f.get('value', '') if isinstance(f.get('value'), str) else ''
                f['value'] = st.text_input(
                    "Value",
                    value=cur_text,
                    key=f"fv_txt_{i}",       # type-specific key — never conflicts with multiselect
                    label_visibility="collapsed",
                    placeholder="%pattern% or value...",
                )

        with c4:
            st.write("")
            if len(st.session_state.filters) > 1:
                if st.button("✕", key=f"rm_{i}", help="Remove this filter"):
                    filters_to_remove.append(i)

    for i in sorted(filters_to_remove, reverse=True):
        st.session_state.filters.pop(i)
    if filters_to_remove:
        st.rerun()

    ca, cb = st.columns([1, 5])
    with ca:
        if st.button("＋ Add filter"):
            st.session_state.filters.append(
                {'column': '', 'operator': 'contains', 'value': '', 'connector': 'AND'}
            )
            st.rerun()
    with cb:
        if st.button("🗑 Clear all filters"):
            st.session_state.filters = [
                {'column': 'POSITION', 'operator': 'contains', 'value': '', 'connector': 'AND'}
            ]
            st.rerun()

    st.divider()

    # ── Run / Show All / toggles ───────────────────────────────────────────────
    r1, r2, r3, r4 = st.columns([1, 1, 1.5, 1.5])
    with r1:
        run_btn = st.button("🔍 Run Query", type="primary", use_container_width=True)
    with r2:
        all_btn = st.button("Show All", use_container_width=True)
    with r3:
        show_breakdown = st.toggle(
            "📊 Rate Breakdown",
            key="breakdown_toggle",
            help="Adds BASE + all burden components to the table so you can see how the Bill Rate is built up",
        )
    with r4:
        show_charts = st.toggle(
            "🥧 Show Charts",
            key="charts_toggle",
            help="Show pie charts breaking down average Bill Rate by Building Type and Owner",
        )

    # Run query and cache the raw result
    if run_btn or all_btn:
        active_filters = st.session_state.filters if run_btn else []
        with st.spinner("Filtering data..."):
            result = filter_dataframe(df, active_filters)

        if result.empty:
            st.warning("No results found. Try adjusting your filters.")
            st.session_state.pop('_result', None)
            st.session_state.pop('_show_all', None)
        else:
            st.session_state['_result']   = result
            st.session_state['_show_all'] = bool(all_btn)

    # ── Results ────────────────────────────────────────────────────────────────
    if '_result' not in st.session_state:
        return

    result   = st.session_state['_result']
    show_all = st.session_state.get('_show_all', False)

    # Build display_df fresh every render so the toggle takes effect immediately
    if show_all:
        display_df = result.reset_index(drop=True)
    else:
        valid_cols = [c for c in st.session_state.selected_columns if c in result.columns]
        if show_breakdown:
            bd_cols  = [c for c in BREAKDOWN_COLS if c in result.columns]
            all_cols = list(dict.fromkeys(valid_cols + bd_cols))  # deduplicated, order preserved
            display_df = result[all_cols].reset_index(drop=True) if all_cols else result.reset_index(drop=True)
        else:
            display_df = result[valid_cols].reset_index(drop=True) if valid_cols else result.reset_index(drop=True)

    st.success(f"**{len(display_df):,} rows** returned")

    # ── Summary metrics ────────────────────────────────────────────────────────
    bill_rates = pd.to_numeric(
        result['BILL_RATE'] if 'BILL_RATE' in result.columns else pd.Series(dtype=float),
        errors='coerce',
    ).dropna()
    if not bill_rates.empty:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Min Bill Rate",    f"${bill_rates.min():,.2f}")
        m2.metric("Avg Bill Rate",    f"${bill_rates.mean():,.2f}")
        m3.metric("Median Bill Rate", f"${bill_rates.median():,.2f}")
        m4.metric("Max Bill Rate",    f"${bill_rates.max():,.2f}")

    st.dataframe(display_df, use_container_width=True, height=450)

    d1, d2, _ = st.columns([1, 1, 4])
    with d1:
        st.download_button(
            "📥 Download Excel",
            data=to_excel(display_df),
            file_name="labor_rates.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "📥 Download CSV",
            data=display_df.to_csv(index=False),
            file_name="labor_rates.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Charts ─────────────────────────────────────────────────────────────────
    if show_charts:
        PURPLE_PALETTE = [
            '#3b1f52', '#52297a', '#6b35a0', '#8b4fc4', '#a96fd8',
            '#c498e8', '#d9bef0', '#ede0f8', '#b07fd4', '#7a4aab',
        ]

        def make_pie(data_series, label_col, value_col, title, top_n=10):
            """Aggregate into top_n categories + 'Other', return a plotly fig."""
            grp = (
                data_series
                .dropna(subset=[label_col, value_col])
                .assign(**{value_col: lambda d: pd.to_numeric(d[value_col], errors='coerce')})
                .dropna(subset=[value_col])
                .groupby(label_col)[value_col]
                .mean()
                .reset_index()
                .sort_values(value_col, ascending=False)
            )
            if len(grp) > top_n:
                top    = grp.iloc[:top_n]
                other_val = grp.iloc[top_n:][value_col].mean()
                other_row = pd.DataFrame([{label_col: 'Other', value_col: other_val}])
                grp = pd.concat([top, other_row], ignore_index=True)

            fig = px.pie(
                grp,
                names=label_col,
                values=value_col,
                title=title,
                color_discrete_sequence=PURPLE_PALETTE,
                hole=0.35,
            )
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Avg Bill Rate: $%{value:,.2f}<extra></extra>',
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#3b1f52',
                title_font_size=13,
                title_font_color='#3b1f52',
                showlegend=True,
                legend=dict(font_size=10),
                margin=dict(t=50, b=20, l=20, r=20),
            )
            return fig

        st.divider()
        c1, c2 = st.columns(2)

        with c1:
            if 'BUILDING_TYPE' in result.columns and 'BILL_RATE' in result.columns:
                fig_bt = make_pie(
                    result, 'BUILDING_TYPE', 'BILL_RATE',
                    'Avg Bill Rate by Building Type',
                )
                st.plotly_chart(fig_bt, use_container_width=True)
            else:
                st.info("BUILDING_TYPE or BILL_RATE column not available.")

        with c2:
            if 'OWNER' in result.columns and 'BILL_RATE' in result.columns:
                owner_result = result.copy()
                owner_result['OWNER'] = owner_result['OWNER'].fillna('Unknown').replace('', 'Unknown')
                fig_own = make_pie(
                    owner_result, 'OWNER', 'BILL_RATE',
                    'Avg Bill Rate by Owner (Top 10)',
                )
                st.plotly_chart(fig_own, use_container_width=True)
            else:
                st.info("OWNER or BILL_RATE column not available.")


if __name__ == "__main__":
    main()
