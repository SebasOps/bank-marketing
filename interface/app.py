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
# Estilos (minimalista, un solo acento, tipografía limpia)
# ---------------------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        --accent: #2563eb;
        --text-main: #111827;
        --text-muted: #6b7280;
        --border: #e5e7eb;
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

    div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input {
        border-radius: 8px !important;
    }
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

# ---------------------------------------------------------------
# Formulario
# Campos yes/no -> selectbox fijo (coincide con Literal["yes","no"] del API).
# Resto -> input libre (texto o número), sin restringir opciones.
# ---------------------------------------------------------------

with st.form("predict_form"):

    st.markdown('<div class="section-label">Perfil del cliente</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("age", min_value=18, max_value=100, value=40, step=1)
        job = st.text_input("job", value="admin.")
        marital = st.text_input("marital", value="married")
    with c2:
        education = st.text_input("education", value="secondary")
        default = st.selectbox("default", options=["no", "yes"])
        balance = st.number_input("balance", value=0, step=1)

    st.markdown('<div class="section-label">Situación financiera</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        housing = st.selectbox("housing", options=["no", "yes"])
    with c4:
        loan = st.selectbox("loan", options=["no", "yes"])

    st.markdown('<div class="section-label">Campaña de contacto</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        contact = st.text_input("contact", value="cellular")
        day = st.number_input("day", min_value=1, max_value=31, value=15, step=1)
        month = st.text_input("month", value="may")
    with c6:
        campaign = st.number_input("campaign", min_value=0, value=1, step=1)
        pdays = st.number_input("pdays", min_value=-1, value=-1, step=1)
        previous = st.number_input("previous", min_value=0, value=0, step=1)

    poutcome = st.text_input("poutcome", value="unknown")

    submitted = st.form_submit_button("Predecir")

# ---------------------------------------------------------------
# Llamada a la API + resultado
# ---------------------------------------------------------------

if submitted:
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
