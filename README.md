# RotaClima — rotação de culturas orientada por dados da NASA

Protótipo para o desafio **NASA Space Apps Challenge 2026 — "Mudança no Campo: Adaptando Fazendas com Dados da NASA"**.

O RotaClima transforma dados climáticos e de satélite da NASA em **recomendações de rotação de culturas explicáveis**
para cada talhão, permitindo ao agricultor **comparar estratégias** (conservação da água, recuperação do solo,
equilíbrio econômico-ambiental) e **testar cenários climáticos hipotéticos** (seca, calor, atraso das chuvas, veranico).

> Protótipo de apoio à decisão. As recomendações **não substituem** a avaliação de um engenheiro agrônomo,
> a análise de solo e o Zoneamento Agrícola de Risco Climático (ZARC).

## Como executar

```bash
pip install -r requirements.txt
streamlit run app.py
```

O app abre em `http://localhost:8501`. As cinco propriedades demonstrativas já têm dados em cache
(`data/cache/`), então a demonstração funciona sem internet. Coordenadas novas são consultadas na hora
(NASA POWER ~10–30 s; NDVI MODIS sob demanda na tela de mapa).

Testes:

```bash
python -m pytest -q
```

## Telas

| Tela | O que mostra |
|---|---|
| 1. Propriedade | Localização, talhões (solo, irrigação, rotação atual), importação de GeoJSON, calibração de chuva por pluviômetro, premissas econômicas |
| 2. Diagnóstico ambiental | Climatologia mensal, chuva × ET0 por ano, anomalias recentes, umidade do solo (GWETROOT), resumo em linguagem simples |
| 3. Mapa dos talhões | Talhões sobre imagem de satélite, NDVI MODIS (250 m), vulnerabilidade do solo, série temporal de NDVI |
| 4. Planejamento de rotação | Top 5 rotações de 2 anos com linha do tempo, pontuação por critério, **por que** e **riscos**, comparação com a rotação atual |
| 5. Comparação de cenários | Melhor rotação para cada prioridade (A/B/C), radar de critérios, distribuição de margens, robustez a pesos |
| 6. Simulador climático | Cenários "e se" sobre os anos históricos: chuva ±%, +°C, atraso das chuvas, veranico, extremos de calor, limite de irrigação |
| 7. Relatório | Relatório Markdown e CSV com dados, estratégias, justificativas, limitações e validações necessárias |

## Dados

| Fonte | Uso | Tipo |
|---|---|---|
| [NASA POWER](https://power.larc.nasa.gov/) (API diária, comunidade AG): `T2M`, `T2M_MAX`, `T2M_MIN`, `PRECTOTCORR`, `RH2M`, `ALLSKY_SFC_SW_DWN`, `WS2M`, `GWETROOT` | Diagnóstico climático, ET0 FAO-56, balanço hídrico diário de 2001 até a data mais recente disponível | Dado observado/reanálise (≈ 0,5° × 0,625°) |
| [MODIS MOD13Q1](https://modis.ornl.gov/data/modis_webservice.html) via ORNL DAAC: `250m_16_days_NDVI` | Vigor da vegetação e solo exposto por talhão | Dado de satélite (250 m, 16 dias) |
| Usuário | Talhões, textura, pH, MO, irrigação, rotação atual, preços e custos | Informado |
| Catálogo de culturas (`rotaclima/crops.py`) | Kc (FAO-56), Ky (FAO-33), profundidade de raiz, limites térmicos, regras de sucessão | Parâmetros da literatura + heurísticas do protótipo |

Os talhões e os valores econômicos das propriedades demonstrativas são **fictícios**; clima e NDVI são **reais**.

## Como a recomendação é calculada

1. **Enumeração**: todas as rotações de 2 anos (verão + 2ª safra/inverno) com as culturas permitidas, respeitando janelas de semeadura.
2. **Simulação**: cada cultura é simulada em todos os anos históricos com balanço hídrico diário (Penman-Monteith FAO-56, Kc, água disponível por textura, estresse hídrico, calor e geada) → produtividade relativa e margem por ano.
3. **Critérios (0–100)**: água (déficit e consumo), solo (cobertura, resíduo, diversidade, leguminosas, raízes profundas), fitossanidade (regras de sucessão), risco climático (probabilidade de quebra > 30%) e economia (margem média e pior decil).
4. **Prioridade**: pesos dos cenários A/B/C ou personalizados; o ranking mostra as razões e os riscos de cada alternativa.
5. **Simulador**: os mesmos cálculos sobre o clima histórico alterado pelo cenário escolhido.

## Limitações

- Cenários do simulador são **hipotéticos** ("e se"); não são previsão do tempo nem projeção climática (para projeções, ver NASA NEX-GDDP-CMIP6).
- Precipitação em grade (≈ 55 km) suaviza a chuva local e tende a subestimar veranicos — use a calibração com pluviômetro.
- Modelo de produtividade simplificado (FAO-33); não considera pragas, doenças, nutrição ou manejo.
- Regras de sucessão e pesos do índice de solo são heurísticas e precisam de validação agronômica local.
- NDVI de 250 m mistura talhões vizinhos, estradas e matas em áreas pequenas.

## Estrutura

```
app.py                  interface Streamlit (7 telas)
rotaclima/nasa_power.py API NASA POWER, cache e ET0 FAO-56
rotaclima/modis.py      NDVI MODIS (ORNL DAAC)
rotaclima/crops.py      culturas, parâmetros e regras de sucessão
rotaclima/engine.py     cenários climáticos e balanço hídrico
rotaclima/recommender.py avaliação, ranking e explicações
rotaclima/diagnostics.py indicadores climáticos
rotaclima/presets.py    propriedades demonstrativas
tests/                  testes do motor e de renderização das telas
```
