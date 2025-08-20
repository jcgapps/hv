import os
import glob
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st

# ----------------- Utilidades -----------------
def ensure_dirs():
    os.makedirs("cartones", exist_ok=True)
    os.makedirs("condiciones", exist_ok=True)

def parse_card_csv(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    text = text.replace(";", ",").replace("\t", " ")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    tokens = []
    if len(lines) >= 5:
        for ln in lines:
            parts = [p.strip() for p in (ln.split(",") if "," in ln else ln.split()) if p.strip()]
            tokens.extend(parts)
    else:
        sep = "," if "," in text else " "
        tokens = [t.strip() for t in text.split(sep) if t.strip()]

    if len(tokens) != 25:
        raise ValueError(f"{os.path.basename(path)}: se esperaban 25 valores, llegaron {len(tokens)}.")

    vals = []
    for t in tokens:
        t = t.upper()
        if t in {"FREE", "X"} or t == "":
            vals.append("FREE")
            continue
        for p in ("B-", "I-", "N-", "G-", "O-"):
            t = t.replace(p, "")
        int(t)
        vals.append(t)

    arr = np.array(vals, dtype=object).reshape(5, 5)
    if arr[2, 2] in {"FREE", "0"}:
        arr[2, 2] = "FREE"
    return arr

def parse_condition_csv(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    text = text.replace(";", ",").replace("\t", " ")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    tokens = []
    if len(lines) >= 5:
        for ln in lines:
            parts = [p.strip() for p in (ln.split(",") if "," in ln else ln.split()) if p.strip()]
            tokens.extend(parts)
    else:
        sep = "," if "," in text else " "
        tokens = [t.strip() for t in text.split(sep) if t.strip()]

    if len(tokens) != 25:
        raise ValueError(f"{os.path.basename(path)}: se esperaban 25 valores (0/1), llegaron {len(tokens)}.")
    if not all(t in {"0", "1"} for t in tokens):
        raise ValueError(f"{os.path.basename(path)}: solo 0/1 permitidos.")
    arr = np.array([int(t) for t in tokens], dtype=int).reshape(5, 5)
    return arr

def normalize_single_token(tok: str) -> str | None:
    if not tok:
        return None
    s = tok.upper().strip()
    s = s.replace("B-", "").replace("I-", "").replace("N-", "").replace("G-", "").replace("O-", "")
    for L in ("B", "I", "N", "G", "O"):
        if s.startswith(L):
            s = s[len(L):]
            break
    s = s.strip().replace(",", "").replace(";", "")
    if not s.isdigit():
        return None
    n = int(s)
    if 1 <= n <= 75:
        return str(n)
    return None

def numbers_set_from_history(history: list[str]) -> set[str]:
    return set(history)

def hits_mask(card: np.ndarray, called_set: set[str]) -> np.ndarray:
    mask = np.zeros_like(card, dtype=bool)
    for r in range(5):
        for c in range(5):
            v = str(card[r, c]).upper()
            mask[r, c] = (v == "FREE") or (v in called_set)
    return mask

def matches_pattern(mask: np.ndarray, pat: np.ndarray) -> bool:
    must = pat.astype(bool)
    return np.all(mask[must])

def render_board(card: np.ndarray, mask: np.ndarray) -> Image.Image:
    W, H = 420, 460
    cell = 80
    img = Image.new("RGB", (W, H), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    for c, ch in enumerate("BINGO"):
        x0 = 10 + c * cell
        draw.rectangle([x0, 10, x0 + cell - 2, 10 + 40], fill=(225, 245, 255))
        draw.text((x0 + 30, 20), ch, fill=(20, 20, 20))
    for r in range(5):
        for c in range(5):
            x0 = 10 + c * cell
            y0 = 60 + r * cell
            bg = (210, 244, 221) if mask[r, c] else (255, 255, 255)
            draw.rectangle([x0, y0, x0 + cell - 2, y0 + cell - 2], fill=bg, outline=(180, 180, 180))
            val = "★" if str(card[r, c]).upper() == "FREE" else str(card[r, c])
            draw.text((x0 + 28, y0 + 28), val, fill=(0, 0, 0))
    return img

# ----------------- App -----------------
ensure_dirs()
st.set_page_config(page_title="Bingo – Cartones y Condiciones", layout="wide")
st.title("🎯 Bingo")

# Estado
if "called_history" not in st.session_state:
    st.session_state.called_history = []
if "input_num" not in st.session_state:
    st.session_state.input_num = ""

# Callbacks
def add_number():
    tok = normalize_single_token(st.session_state.input_num)
    if tok is None:
        st.toast("Número inválido. Usa 1..75 o B/I/N/G/O con número.")
    else:
        if tok in st.session_state.called_history:
            st.toast(f"El {tok} ya fue cantado.")
        else:
            st.session_state.called_history.append(tok)
            st.toast(f"Agregado: {tok}")
    st.session_state.input_num = ""  # limpiar input

def undo_last():
    if st.session_state.called_history:
        last = st.session_state.called_history.pop()
        st.toast(f"Deshecho: {last}")

# Cargar archivos
carton_paths = sorted(glob.glob(os.path.join("cartones", "*.csv")))
cartones, carton_errors = [], []
for p in carton_paths:
    try:
        grid = parse_card_csv(p)
        cartones.append({"name": os.path.basename(p), "grid": grid})
    except Exception as e:
        carton_errors.append(str(e))

cond_paths = sorted(glob.glob(os.path.join("condiciones", "*.csv")))
condiciones, cond_errors = [], []
for p in cond_paths:
    try:
        pat = parse_condition_csv(p)
        condiciones.append({"name": os.path.basename(p), "mask": pat})
    except Exception as e:
        cond_errors.append(str(e))

# ============ Barra lateral: activar / desactivar ============
st.sidebar.header("⚙️ Selección de juego")

# Inicializar switches persistentes (True por defecto)
for c in cartones:
    key = f"card_active::{c['name']}"
    if key not in st.session_state:
        st.session_state[key] = True
for cond in condiciones:
    key = f"cond_active::{cond['name']}"
    if key not in st.session_state:
        st.session_state[key] = True

with st.sidebar.expander("Cartones activos", expanded=True):
    for c in cartones:
        key = f"card_active::{c['name']}"
        st.session_state[key] = st.checkbox(c["name"], value=st.session_state[key], key=key)
with st.sidebar.expander("Condiciones activas", expanded=True):
    for cond in condiciones:
        key = f"cond_active::{cond['name']}"
        st.session_state[key] = st.checkbox(cond["name"], value=st.session_state[key], key=key)

# Filtrar por selección
active_cartones = [c for c in cartones if st.session_state.get(f"card_active::{c['name']}", True)]
active_condiciones = [c for c in condiciones if st.session_state.get(f"cond_active::{c['name']}", True)]

# ============ Entrada rápida ============
cols_top = st.columns([2, 1])
with cols_top[0]:
    st.subheader("Números cantados (rápido para móvil)")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.text_input(
            "Escribe un número y toca Agregar (también acepta B-7, I22, etc.)",
            key="input_num",
            label_visibility="collapsed",
            placeholder="Ej: 14 o B-14",
        )
    with c2:
        st.button("➕ Agregar", use_container_width=True, on_click=add_number)
    with c3:
        st.button("↩️ Borrar último", use_container_width=True, on_click=undo_last)

    called_sorted = sorted(numbers_set_from_history(st.session_state.called_history), key=lambda x: int(x))
    st.markdown("**Ordenados (para marcar cartones):** " + (", ".join(called_sorted) if called_sorted else "—"))
    st.markdown("**Historial (último al final):** " + (", ".join(st.session_state.called_history) if st.session_state.called_history else "—"))

with cols_top[1]:
    st.subheader("Estado de archivos")
    st.write(f"Cartones cargados: **{len(cartones)}** (activos: {len(active_cartones)})")
    for c in cartones:
        prefix = "✅" if c in active_cartones else "⛔"
        st.caption(f"{prefix} {c['name']}")
    st.write(f"Condiciones cargadas: **{len(condiciones)}** (activas: {len(active_condiciones)})")
    for c in condiciones:
        prefix = "✅" if c in active_condiciones else "⛔"
        st.caption(f"{prefix} {c['name']}")

if carton_errors:
    st.error("Errores en cartones:\n- " + "\n- ".join(carton_errors))
if cond_errors:
    st.error("Errores en condiciones:\n- " + "\n".join(cond_errors))

if not active_cartones:
    st.info("No hay cartones activos. Activa alguno en la barra lateral.")
    st.stop()
if not active_condiciones:
    st.warning("No hay condiciones activas. Activa al menos una en la barra lateral.")

called_set = numbers_set_from_history(st.session_state.called_history)

# ============ Evaluar ganadores primero ============
winners = []
for card in active_cartones:
    grid = card["grid"]
    mask = hits_mask(grid, called_set)
    for cond in active_condiciones:
        if matches_pattern(mask, cond["mask"]):
            winners.append((card["name"], cond["name"]))

# ============ Mostrar ganadores arriba ============
st.markdown("## 📣 Resultado")
if winners:
    for carton, cond in winners:
        st.success(f"🎉 **¡GANADOR!**\n\n**Cartón:** {carton}\n**Condición:** {cond}")
        try:
            st.toast(f"🎉 ¡GANADOR! Cartón: {carton} — Condición: {cond}")
        except Exception:
            pass
else:
    st.info("Todavía no hay cartones ganadores.")

st.markdown("---")

# ============ Mostrar cartones ============
for card in active_cartones:
    grid = card["grid"]
    mask = hits_mask(grid, called_set)
    board_img = render_board(grid, mask)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"### {card['name']}")
        st.image(board_img, use_container_width=True, caption="Marcado con números cantados")

    matched = [cond["name"] for cond in active_condiciones if matches_pattern(mask, cond["mask"])]
    with c2:
        if matched:
            st.success("✅ Este cartón es ganador con:\n" + "\n".join([f"• {m}" for m in matched]))
        else:
            st.info("Sin bingo aún.")

st.markdown("---")
st.subheader("📊 Resumen final")
if winners:
    df = pd.DataFrame(winners, columns=["Cartón", "Condición cumplida"])
    st.dataframe(df, use_container_width=True)
else:
    st.write("Sin ganadores por ahora.")

st.caption("Tip: desmarca condiciones o cartones en la barra lateral para seguir jugando con los demás.")
        for p in ("B-", "I-", "N-", "G-", "O-"):
            t = t.replace(p, "")
        int(t)
        vals.append(t)

    arr = np.array(vals, dtype=object).reshape(5, 5)
    if arr[2, 2] in {"FREE", "0"}:
        arr[2, 2] = "FREE"
    return arr

def parse_condition_csv(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    text = text.replace(";", ",").replace("\t", " ")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    tokens = []
    if len(lines) >= 5:
        for ln in lines:
            parts = [p.strip() for p in (ln.split(",") if "," in ln else ln.split()) if p.strip()]
            tokens.extend(parts)
    else:
        sep = "," if "," in text else " "
        tokens = [t.strip() for t in text.split(sep) if t.strip()]

    if len(tokens) != 25:
        raise ValueError(f"{os.path.basename(path)}: se esperaban 25 valores (0/1), llegaron {len(tokens)}.")
    if not all(t in {"0", "1"} for t in tokens):
        raise ValueError(f"{os.path.basename(path)}: solo 0/1 permitidos.")
    arr = np.array([int(t) for t in tokens], dtype=int).reshape(5, 5)
    return arr

def normalize_single_token(tok: str) -> str | None:
    if not tok:
        return None
    s = tok.upper().strip()
    s = s.replace("B-", "").replace("I-", "").replace("N-", "").replace("G-", "").replace("O-", "")
    for L in ("B", "I", "N", "G", "O"):
        if s.startswith(L):
            s = s[len(L):]
            break
    s = s.strip().replace(",", "").replace(";", "")
    if not s.isdigit():
        return None
    n = int(s)
    if 1 <= n <= 75:
        return str(n)
    return None

def numbers_set_from_history(history: list[str]) -> set[str]:
    return set(history)

def hits_mask(card: np.ndarray, called_set: set[str]) -> np.ndarray:
    mask = np.zeros_like(card, dtype=bool)
    for r in range(5):
        for c in range(5):
            v = str(card[r, c]).upper()
            mask[r, c] = (v == "FREE") or (v in called_set)
    return mask

def matches_pattern(mask: np.ndarray, pat: np.ndarray) -> bool:
    must = pat.astype(bool)
    return np.all(mask[must])

def render_board(card: np.ndarray, mask: np.ndarray) -> Image.Image:
    W, H = 420, 460
    cell = 80
    img = Image.new("RGB", (W, H), (248, 250, 252))
    draw = ImageDraw.Draw(img)
    for c, ch in enumerate("BINGO"):
        x0 = 10 + c * cell
        draw.rectangle([x0, 10, x0 + cell - 2, 10 + 40], fill=(225, 245, 255))
        draw.text((x0 + 30, 20), ch, fill=(20, 20, 20))
    for r in range(5):
        for c in range(5):
            x0 = 10 + c * cell
            y0 = 60 + r * cell
            bg = (210, 244, 221) if mask[r, c] else (255, 255, 255)
            draw.rectangle([x0, y0, x0 + cell - 2, y0 + cell - 2], fill=bg, outline=(180, 180, 180))
            val = "★" if str(card[r, c]).upper() == "FREE" else str(card[r, c])
            draw.text((x0 + 28, y0 + 28), val, fill=(0, 0, 0))
    return img

# ----------------- App -----------------
ensure_dirs()
st.set_page_config(page_title="Bingo – Cartones y Condiciones", layout="wide")
st.title("🎯 Bingo — Cartones desde `cartones/` y Condiciones desde `condiciones/`")

# Estado
if "called_history" not in st.session_state:
    st.session_state.called_history = []
if "input_num" not in st.session_state:
    st.session_state.input_num = ""

# Callbacks (solucionan el error al limpiar input)
def add_number():
    tok = normalize_single_token(st.session_state.input_num)
    if tok is None:
        st.toast("Número inválido. Usa 1..75 o B/I/N/G/O con número.")
    else:
        if tok in st.session_state.called_history:
            st.toast(f"El {tok} ya fue cantado.")
        else:
            st.session_state.called_history.append(tok)
            st.toast(f"Agregado: {tok}")
    st.session_state.input_num = ""  # limpiar input

def undo_last():
    if st.session_state.called_history:
        last = st.session_state.called_history.pop()
        st.toast(f"Deshecho: {last}")

# Cargar archivos
carton_paths = sorted(glob.glob(os.path.join("cartones", "*.csv")))
cartones, carton_errors = [], []
for p in carton_paths:
    try:
        grid = parse_card_csv(p)
        cartones.append({"name": os.path.basename(p), "grid": grid})
    except Exception as e:
        carton_errors.append(str(e))

cond_paths = sorted(glob.glob(os.path.join("condiciones", "*.csv")))
condiciones, cond_errors = [], []
for p in cond_paths:
    try:
        pat = parse_condition_csv(p)
        condiciones.append({"name": os.path.basename(p), "mask": pat})
    except Exception as e:
        cond_errors.append(str(e))

cols_top = st.columns([2, 1])
with cols_top[0]:
    st.subheader("Números cantados (entrada rápida para móvil)")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.text_input(
            "Escribe un número y toca Agregar (también acepta B-7, I22, etc.)",
            key="input_num",
            label_visibility="collapsed",
            placeholder="Ej: 14 o B-14",
        )
    with c2:
        st.button("➕ Agregar", use_container_width=True, on_click=add_number)
    with c3:
        st.button("↩️ Borrar último", use_container_width=True, on_click=undo_last)

    called_sorted = sorted(numbers_set_from_history(st.session_state.called_history), key=lambda x: int(x))
    st.markdown("**Ordenados (para marcar cartones):** " + (", ".join(called_sorted) if called_sorted else "—"))
    st.markdown("**Historial (último al final):** " + (", ".join(st.session_state.called_history) if st.session_state.called_history else "—"))

with cols_top[1]:
    st.subheader("Estado de archivos")
    st.write(f"Cartones cargados: **{len(cartones)}**")
    for c in cartones:
        st.caption("• " + c["name"])
    st.write(f"Condiciones cargadas: **{len(condiciones)}**")
    for c in condiciones:
        st.caption("• " + c["name"])

if carton_errors:
    st.error("Errores en cartones:\n- " + "\n- ".join(carton_errors))
if cond_errors:
    st.error("Errores en condiciones:\n- " + "\n".join(cond_errors))

if not cartones:
    st.info("No se encontraron cartones. Coloca archivos CSV 5×5 en la carpeta **cartones/**.")
    st.stop()
if not condiciones:
    st.warning("No se encontraron condiciones. Coloca archivos CSV 5×5 de 0/1 en **condiciones/**.")

called_set = numbers_set_from_history(st.session_state.called_history)

# Evaluar ganadores primero
winners = []
for card in cartones:
    grid = card["grid"]
    mask = hits_mask(grid, called_set)
    for cond in condiciones:
        if matches_pattern(mask, cond["mask"]):
            winners.append((card["name"], cond["name"]))

# Mostrar ganadores arriba
st.markdown("## 📣 Resultado")
if winners:
    for carton, cond in winners:
        st.success(f"🎉 **¡GANADOR!**\n\n**Cartón:** {carton}\n**Condición:** {cond}")
        try:
            st.toast(f"🎉 ¡GANADOR! Cartón: {carton} — Condición: {cond}")
        except Exception:
            pass
else:
    st.info("Todavía no hay cartones ganadores.")

st.markdown("---")

# Mostrar cartones
for card in cartones:
    grid = card["grid"]
    mask = hits_mask(grid, called_set)
    board_img = render_board(grid, mask)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"### {card['name']}")
        st.image(board_img, use_container_width=True, caption="Marcado con números cantados")

    matched = [cond["name"] for cond in condiciones if matches_pattern(mask, cond["mask"])]
    with c2:
        if matched:
            st.success("✅ Este cartón es ganador con:\n" + "\n".join([f"• {m}" for m in matched]))
        else:
            st.info("Sin bingo aún.")

st.markdown("---")
st.subheader("📊 Resumen final")
if winners:
    df = pd.DataFrame(winners, columns=["Cartón", "Condición cumplida"])
    st.dataframe(df, use_container_width=True)
else:
    st.write("Sin ganadores por ahora.")

st.caption("Tip: en móvil, escribe el número y toca **Agregar**. Si te equivocas, usa **Borrar último**.")
