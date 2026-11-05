# Motion Art Lab

Aplicación de escritorio que convierte el movimiento de una webcam en arte generativo. La diferencia entre fotogramas, el umbral, el filtro, la morfología y los componentes conectados están implementados con NumPy. OpenCV solo abre la cámara. La interfaz, los rastros y las partículas los dibuja pygame-ce.

El documento de diseño es `MOTION_ART_LAB_PLAN_IMPLEMENTACION.md`. Esta versión es el MVP: detector por diferencia de fotogramas, limpieza de máscara, rastros, partículas, diagnóstico y captura PNG.

## Requisitos

- Python 3.12 o superior. En el equipo de desarrollo no estaba instalado 3.12; la ejecución se comprobó con **Python 3.13.3**.
- Webcam opcional. Si no hay cámara, la app sigue abierta y puede usar la **Demo sintética**, marcada en pantalla. Esa demo no es una persona real.

## Instalación

Desde la raíz del repositorio, en PowerShell:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Si tienes Python 3.12, sustituye `py -3.13` por `py -3.12`.

## Ejecución

```powershell
python -m motion_art_lab.main
```

Sin abrir la cámara:

```powershell
python -m motion_art_lab.main --demo
```

## Controles

| Acción | Ratón | Teclado |
|---|---|---|
| Pausar o reanudar | Pausar / Reanudar | Espacio |
| Rastros de luz | Rastros | 1 |
| Partículas | Partículas | 2 |
| Limpiar el lienzo | Limpiar | C |
| Guardar PNG | Guardar PNG | S |
| Diagnóstico | Diagnóstico | D |
| Cerrar ajustes, o salir | — | Esc |

La sensibilidad es inversa al umbral: más sensibilidad, menor `T`. El valor de `T` se muestra junto al slider. Intensidad y persistencia cambian el trazo y la vida de las partículas mientras la app está abierta.

En **Ajustes avanzados** están el filtro de promedio 3×3, la apertura, el cierre, el área mínima, el espejo y el fondo de cámara. La apertura y el cierre empiezan en 1 y 0. Si un dedo desaparece, baja la apertura a 0.

Las imágenes se guardan en `outputs/gallery/`. El diagnóstico exporta la cuadrícula visible; el modo artístico exporta el lienzo.

## Pruebas

```powershell
python -m pytest
python -m ruff check src tests
```

Resultado medido en este equipo, el 4 de noviembre de 2026:

- `33 passed` con pytest 9.2.0.
- Ruff 0.17.3 sin avisos sobre `src` y `tests`.
- Autocomprobación gráfica con la demo sintética: la ventana llegó a unos **55.6 FPS**, el último fotograma tardó **4.1 ms** en procesarse, el movimiento medido fue **1.9 %** y había **3 regiones**. Esas cifras son de esa ejecución, no un objetivo garantizado.
- La cámara del equipo se abrió a **640×480** y el pipeline la redujo a **320×240**. El primer fotograma quedó en calentamiento. En una muestra de seis lecturas el tiempo de proceso estuvo entre **3 y 24 ms**. No se hizo una prueba visual de una mano frente a la cámara dentro de la ventana.

## Qué incluye y qué no

Incluye captura, demo sintética, diferencia de fotogramas, umbral, filtro de promedio, erosión, dilatación, apertura, cierre, componentes conectados, descarte por área, rastros, partículas (máximo 350), cuatro paletas, diagnóstico y PNG.

No incluye Running Average, Lucas–Kanade, silueta, modo experimental, galería completa ni exportación de video. Esos puntos siguen previstos como ampliación y no bloquean esta versión.

## Limitación del detector

Frame Differencing ve el cambio entre dos fotogramas. Una persona que se queda quieta deja de generar máscara. La máscara no es una silueta completa del cuerpo. El diagnóstico enseña el fotograma, la diferencia, la máscara cruda y la máscara limpia para poder comprobarlo.
