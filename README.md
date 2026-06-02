# 🏜️ S-ARIDA  
### *Sistema de Alerta y Riesgo por Incidencia de Sequías y Desastres Ambientales*

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Hecho con ❤️ en Colombia](https://img.shields.io/badge/Hecho%20con%20❤️%20en-Colombia-yellow)]()

---

## 🌍 Descripción del proyecto

**S-ARIDA** es un sistema de análisis y alerta temprana para el monitoreo y predicción de **sequías** en **Riohacha** en el departamento de La Guajira (Colombia). Para lograrlo, hace uso de datos climáticos del reanálisis **ERA5** y modelos de **machine learning** para estimar la probabilidad de ocurrencia de sequía en los siguientes 3 meses.

Este proyecto de ciencia al servicio de la ciudadanía busca fortalecer la capacidad de **prevención y gestión del riesgo climático** en una de las regiones más afectadas por la aridez y el cambio climático en Colombia: La Guajira.

YouTube: https://youtu.be/4Pf9hkQNCAI  
Dashboard: https://s-arida.streamlit.app/


---

## 🎯 Objetivos

- Analizar tendencias históricas de sequías usando datos ERA5 y el test de Mann-Kendall.    
- Entrenar un modelo predictivo para estimar el riesgo trimensual de sequía.  
- Proveer medios de visualización de datos interactivos que permitan realizar interpretaciones útiles para la toma de decisiones preventivas, planificación y desarrollo de acciones de mitigación de las consecuencias de las sequías.
- Brindar recursos informativos para comprender mejor estas anomalías climáticas (causas y consecuencias) y pautas estratégicas para la prevención y mitigación de sus efectos en los ecosistemas y las comunidades locales.

---

## 🧩 Estructura del repositorio

S-ARIDA/  
│  
├── 📁 Análisis Histórico  
│ └── 01 Análisis_histórico_Riohacha.ipynb  
├── 📁 Modelo Predictivo  
│ └── 01 modelo_predictivo.ipynb  
├── 📁 Dashboard  
│ └── app.py  
│  
├── README.md   
└── requirements.txt  

## 🧠 Equipo

S-ARIDA fue realizado por Quark5, un equipo multidisciplinario de ingenieros en sistemas, físicos e investigadores sociales.

👩‍💻 [Alexa Serrano] — Diseño UI/UX / Dashboard  
👨‍💻 [Andre Avila] —  Divulgación Científica / Comunicación / Prompt Design / Ciencia Ciudadana   
👨‍💻 [Cristian Orduz] — Diseño UI/UX / Dashboard, Machine Learning / Modelado Predictivo 

👩‍💻 [Jhon Almanzar] — Diseño UI/UX / Dashboard, Machine Learning / Modelado Predictivo  
👩‍💻 [Mariannly Marquez] — Ciencia de Datos / Análisis Hístorico  




---

## 🚀 Features principales de S-ARIDA API

### 🔹 1. Modelo predictivo de sequías
- Desarrollado con datos abiertos del programa europeo **Copernicus** (servicio de cambio climático).
- Analiza **más de 40 años de información climática** (1985–2025) de La Guajira y Riohacha.
- Utiliza variables clave: **precipitación, evaporación, humedad del suelo, temperatura** y **radiación solar**.
- Genera mensualmente la **probabilidad de ocurrencia de una sequía** con base en patrones históricos.
- Los resultados se presentan en una escala de riesgo interpretativa:  
  `Bajo (0–0.25) | Moderado (0.25–0.5) | Alto (0.5–0.75) | Crítico (>0.75)`.
- Objetivo: **transformar décadas de datos climáticos en información predictiva para la prevención y adaptación**.

---

### 🔹 2. Visualización de tendencias climáticas (Mann–Kendall)
- Muestra la evolución histórica del **índice SPEI** (precipitación vs. evaporación) en La Guajira.
- La línea discontinua representa la **tendencia Mann–Kendall**, que detecta si hay un cambio sostenido en las condiciones de sequía.
- Una tendencia negativa indica **aumento en la aridez y pérdida progresiva de humedad**, reflejando el impacto del cambio climático.
- Permite a los usuarios **visualizar y comprender la relación entre las sequías históricas y la variabilidad climática**.

---

### 🔹 3. Panel de análisis climático integral
- Combina indicadores e índices hidrometeorológicos (SPEI, SPI, precipitación, evaporación, humedad del suelo, temperatura y radiación solar).
- Ofrece una lectura interactiva y comparativa de las variables que influyen en la aparición de sequías.
- Permite entender **cómo cambian las condiciones climáticas en el tiempo y su influencia en el riesgo actual.**

---

### 🔹 4. Línea de tiempo histórica de sequías
- Resume los principales eventos de sequía registrados en La Guajira entre 1985 y 2025.
- Combina **fuentes institucionales, prensa y análisis estadístico**.
- Permite comparar **periodos críticos históricos** con las proyecciones del modelo actual.
- Favorece la comprensión de **cómo la frecuencia e intensidad de las sequías ha variado** en los últimos 40 años.

---

### 🔹 5. Chatbot educativo impulsado por Gemini AI
- Asistente interactivo que **actúa como un experto del IDEAM**, entrenado con información científica **nacional e internacional verificada**.
- Su propósito: **hacer accesible la información técnica** a funcionarios y ciudadanos, explicando los indicadores del modelo y orientando sobre prevención.
- Responde con claridad y contexto local, evitando desinformación y priorizando la comprensión.
- Ejemplos de temas y preguntas que puede responder:
  - **Sobre indicadores climáticos:**  
    _“¿Qué significa el valor del SPEI que aparece en el panel y cómo se relaciona con el riesgo de sequía?”_
  - **Sobre prevención y gestión del riesgo:**  
    _“¿Qué acciones deberían tomar las autoridades locales si aumenta la probabilidad de sequía?”_
  - **Sobre cambio climático:**  
    _“¿Cómo influye el cambio climático en la duración y frecuencia de las sequías en regiones como La Guajira?”_ 
  - **Sobre impactos sociales y ecológicos:**  
    _“¿Qué consecuencias puede tener una sequía prolongada en los ecosistemas y comunidades rurales?”_
  - **Sobre interpretación de datos y visualizaciones:**  
    _“¿Cómo se interpreta la tendencia negativa en la gráfica Mann–Kendall?”_
  - **Líneas locales de reporte de emergencias y atención frente a desastres**
    _“¿A dónde debo llamar para reportar un incendio activo?”_

---

### 🔹 6. Buzón de reportes ciudadanos
- Espacio donde cualquier persona puede registrar observaciones locales:  
  ej. _“mi pozo se secó”_ o _“el río bajó su nivel.”_
- Cada reporte queda almacenado con **fecha y ubicación** y es **descargable en formato `.csv`**.
- Promueve la **ciencia ciudadana** y mejora la capacidad institucional para **responder a tiempo a eventos de sequía.**

---

### 🔹 7. Playground para expertos
- Entorno interactivo para investigadores, técnicos y estudiantes.
- Permite **ingresar manualmente valores de variables meteorológicas** (precipitación, temperatura, evaporación, etc.) y observar la respuesta del modelo.
- Facilita el aprendizaje sobre el comportamiento del algoritmo y la exploración de **escenarios hipotéticos de cambio climático.**

---

### 🔹 8. Recomendaciones dinámicas
- Sección que presenta **acciones prácticas antes, durante y después de una sequía**, diferenciadas por público:
  - 🏛️ Instituciones (planificación, manejo del agua, respuesta rápida)
  - 👥 Comunidad (uso racional del agua, prevención de incendios, salud)
- Información basada en lineamientos del **IDEAM**, **UNGRD** y **FAO**.
- Busca fortalecer la **gobernanza del agua** y la **resiliencia comunitaria**.

---

### 🔹 9. Integración y accesibilidad
- Toda la información se muestra a través de **visualizaciones interactivas, dashboards y endpoints de API.**
- El sistema está diseñado para que **funcionarios, técnicos y ciudadanos** puedan explorar y comprender los datos sin necesidad de conocimientos avanzados.
- Promueve la **transparencia, el acceso al conocimiento científico y la toma de decisiones informadas.**

---

## ⚙️ Requisitos

Leer requirements.txt y requirements_dashboard.txt para instalar las librerías necesarias.


## 📚 Licencia


S-ARIDA  © 2025 by Quark5 is licensed under CC BY 4.0. To view a copy of this license, visit https://creativecommons.org/licenses/by/4.0/


---

## 🌞 Cita recomendada

“ Alexa Serrano, Andre Avila, Cristian Orduz, Jhon Almanzar y Mariannly Marquez (2025). S-ARIDA: Sistema de Alerta y Riesgo por Incidencia de Sequías y Desastres Ambientales. Proyecto de análisis climático para Riohacha basado en datos ERA5 y aprendizaje automático.”
