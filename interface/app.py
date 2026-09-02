"""
Streamlit UI - Cliente de la API de inferencia (Bank Marketing).

Consume el endpoint FastAPI (`GET /health`, `POST /predict`) del modelo
Random Forest servido desde `model_artifact/`. No contiene lógica de
negocio ni de preprocesamiento: solo construye el payload según el
schema de `ClientFeatures`, llama a la API y muestra la respuesta.

Ejecutar:
    pip install streamlit requests
    streamlit run streamlit_app.py
"""

import requests
import streamlit as st

# ---------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------

st.set_page_config(
    page_title="Bank Marketing · Predicción",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------
# Estilos (minimalista, un solo acento, inputs blancos forzados)
# ---------------------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        --accent: #2563eb;
        --text-main: #111827;
        --text-muted: #6b7280;
        --border: #d1d5db;
        --bg-card: #ffffff;
    }

    .stApp { background-color: #fafafa; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { max-width: 720px; padding-top: 2.5rem; }

    h1 { font-size: 1.6rem !important; font-weight: 700 !important;
         color: var(--text-main); margin-bottom: 0.2rem; }
    .subtitle { color: var(--text-muted); font-size: 0.92rem; margin-bottom: 1.8rem; }

    .section-label {
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.04em;
        text-transform: uppercase; color: var(--text-muted);
        margin: 1.4rem 0 0.4rem 0;
    }

    div[data-testid="stForm"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.6rem 1.8rem 1.2rem 1.8rem;
        background-color: var(--bg-card);
    }

    /* Títulos de cada input, siempre visibles */
    label[data-testid="stWidgetLabel"] p {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: var(--text-main) !important;
    }

    /* Forzar fondo blanco + texto oscuro en todos los inputs */
    .stTextInput input,
    .stNumberInput input,
    div[data-baseweb="select"] > div,
    div[data-baseweb="popover"] {
        background-color: #ffffff !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }
    .stTextInput input::placeholder,
    .stNumberInput input::placeholder {
        color: #9ca3af !important;
        opacity: 1 !important;
    }
    ul[role="listbox"] { background-color: #ffffff !important; }
    li[role="option"] { color: var(--text-main) !important; }

    .stButton > button, .stFormSubmitButton > button {
        background-color: var(--accent);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 0.55rem 1.4rem; width: 100%;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background-color: #1d4ed8;
    }

    .result-card { border-radius: 12px; padding: 1.4rem 1.6rem;
        border: 1px solid var(--border); margin-top: 1.4rem; }
    .result-yes { background-color: #ecfdf5; border-color: #a7f3d0; }
    .result-no  { background-color: #fef2f2; border-color: #fecaca; }
    .result-title { font-size: 1.05rem; font-weight: 700; margin-bottom: 0.2rem; }
    .result-meta { color: var(--text-muted); font-size: 0.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Configuración de la API (URL configurable, útil entre local/Docker)
# ---------------------------------------------------------------

if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"

with st.sidebar:
    st.markdown("**Configuración**")
    st.session_state.api_url = st.text_input("URL de la API", value=st.session_state.api_url)
    if st.button("Probar conexión"):
        try:
            r = requests.get(f"{st.session_state.api_url}/health", timeout=5)
            if r.status_code == 200:
                st.success(f"OK · modelo v{r.json().get('model_version')}")
            else:
                st.error(f"La API respondió {r.status_code}: {r.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"No se pudo conectar: {e}")

# ---------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------

st.markdown("<h1>Bank Marketing · Predicción de suscripción</h1>", unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Completa los datos del cliente para estimar si suscribirá '
    'un depósito a plazo.</p>',
    unsafe_allow_html=True,
)

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]
DAYS = list(range(1, 32))

# ---------------------------------------------------------------
# Formulario
# - default / housing / loan -> selectbox fijo (Literal["yes","no"] del API)
# - day / month -> selectbox con opciones válidas del dataset
# - resto -> input libre (texto o número), sin valores precargados
# ---------------------------------------------------------------

with st.form("predict_form"):

    st.markdown('<div class="section-label">Perfil del cliente</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input(
            "Edad", min_value=18, max_value=100, value=None,
            step=1, placeholder="ej. 40",
        )
        job = st.text_input(
            "Trabajo", value="", placeholder="ej. admin., technician, blue-collar",
        )
        marital = st.text_input(
            "Estado civil", value="", placeholder="ej. married, single, divorced, unknown",
        )
    with c2:
        education = st.text_input(
            "Educación", value="", placeholder="ej. basic.6y, high.school, university.degree, unknown",
        )
        

    st.markdown('<div class="section-label">Situación financiera</div>', unsafe_allow_html=True)
    c2, c3 = st.columns(2)
    with c2:
        balance = st.number_input(
            "Balance", value=None, step=1, placeholder="ej. 1500",
        )
    with c3:
        default = st.selectbox(
            "¿Tiene incumplimiento de pago?", options=["no", "yes"], index=None,
            placeholder="Select",
        )
    c4, c5 = st.columns(2)
    with c4:
        housing = st.selectbox(
            "¿Tiene préstamo hipotecario?", options=["no", "yes"], index=None, placeholder="Select",
        )
    with c5:
        loan = st.selectbox(
            "¿Tiene préstamo personal?", options=["no", "yes"], index=None, placeholder="Select",
        )

    st.markdown('<div class="section-label">Campañas de marketing</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        contact = st.text_input(
            "Contactado mediante", value="", placeholder="ej. cellular, telephone",
        )
        day = st.selectbox("Day", options=DAYS, index=None, placeholder="Select day")
        month = st.selectbox("Month", options=MONTHS, index=None, placeholder="Select month")
    with c6:
        campaign = st.number_input(
            "Contactos realizados en esta campaña", min_value=0, value=None, step=1, placeholder="ej. 1",
        )
        pdays = st.number_input(
            "Días desde contacto de una campaña anterior", min_value=-1, value=None, step=1,
            placeholder="-1 si nunca fue contactado",
        )
        previous = st.number_input(
            "Contactos anteriores a esta campaña", min_value=0, value=None, step=1, placeholder="ej. 0",
        )

    poutcome = st.text_input(
        "Resultado de la campaña de marketing anterior", value="", placeholder="ej. failure, success, nonexistent",
    )

    submitted = st.form_submit_button("Predecir")

# ---------------------------------------------------------------
# Validación + llamada a la API + resultado
# ---------------------------------------------------------------

if submitted:
    raw_fields = {
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "balance": balance, "housing": housing, "loan": loan,
        "contact": contact, "day": day, "month": month, "campaign": campaign,
        "pdays": pdays, "previous": previous, "poutcome": poutcome,
    }

    missing = [
        name for name, value in raw_fields.items()
        if value is None or (isinstance(value, str) and value.strip() == "")
    ]

    if missing:
        st.error("Completa los siguientes campos antes de continuar: " + ", ".join(missing))
        st.stop()

    payload = {
        "age": int(age),
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": int(balance),
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": int(day),
        "month": month,
        "campaign": int(campaign),
        "pdays": int(pdays),
        "previous": int(previous),
        "poutcome": poutcome,
    }

    try:
        resp = requests.post(f"{st.session_state.api_url}/predict", json=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        st.error(f"No se pudo conectar con la API en `{st.session_state.api_url}`: {e}")
        st.stop()

    if resp.status_code == 422:
        st.error("Datos inválidos según el esquema de la API:")
        st.json(resp.json())
        st.stop()
    elif resp.status_code != 200:
        st.error(f"Error de la API ({resp.status_code}): {resp.text}")
        st.stop()

    result = resp.json()
    is_yes = result["prediction"] == 1
    proba = result.get("probability")

    css_class = "result-yes" if is_yes else "result-no"
    label = "Sí suscribirá" if is_yes else "No suscribirá"
    icon = "✅" if is_yes else "⛔"

    st.markdown(
        f"""
        <div class="result-card {css_class}">
            <div class="result-title">{icon} {label}</div>
            <div class="result-meta">Modelo v{result['model_version']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if proba is not None:
        st.markdown(
            '<div class="section-label">Probabilidad de suscripción</div>',
            unsafe_allow_html=True,
        )
        st.progress(min(max(proba, 0.0), 1.0))
        st.metric(label="probability", value=f"{proba:.2%}")
    else:
        st.caption("La API no devolvió probabilidad para esta respuesta.")
    