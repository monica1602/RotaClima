from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["1. Propriedade", "2. Diagnóstico ambiental", "3. Mapa dos talhões", "4. Planejamento de rotação",
         "5. Comparação de cenários", "6. Simulador climático", "7. Relatório"]


@pytest.fixture(scope="module")
def app(demo_climate):
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_errors(app, page):
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    assert any(h.value.startswith(page[:2]) for h in app.title)


def test_simulator_drought_preset(app):
    app.sidebar.radio[0].set_value("6. Simulador climático").run()
    next(b for b in app.button if b.label == "Seca severa").click().run()
    assert not app.exception
    assert any(m.value.startswith("**Cenário:**") and "chuva -40" in m.value.replace("−", "-") for m in app.markdown), [m.value for m in app.markdown if "Cenário" in m.value]
