# Motion Art Lab — Plan maestro de implementación

> **Documento rector / Source of truth · Propuesta de proyecto final · Procesamiento de Imágenes y Video**  
> **Versión:** 1.0 · **Fecha:** 04-11-2026 · **Idioma de la aplicación:** español  
> **Estado:** especificación de funcionalidades previstas, **no** declaración de funcionalidades ya implementadas.  
> **Estructura:** dos módulos principales: **Módulo 1 — Motor matemático y transformaciones** y **Módulo 2 — Interfaz y experiencia artística**.

## 0. Propósito del proyecto (propuesta original)

**Motion Art Lab** es una aplicación de escritorio que utiliza una cámara web para detectar el movimiento de una persona y transformarlo en **arte generativo en tiempo real**. El movimiento podrá producir **rastros luminosos, partículas, siluetas abstractas y combinaciones de efectos**, con parámetros modificables durante la ejecución.

El propósito académico es demostrar el procesamiento de imágenes y video mediante algoritmos **programados manualmente en Python con apoyo de operaciones elementales de NumPy**, sin delegar la solución central a detectores, filtros morfológicos, seguimiento óptico o segmentadores ya resueltos por una biblioteca. OpenCV se reserva principalmente para la **adquisición de video** y operaciones de entrada/salida; Pygame se encarga de la **visualización, los efectos y los controles**.

El proyecto reúne dos objetivos complementarios:

1. **Técnico:** convertir fotogramas en diferencias de intensidad, máscaras, regiones de movimiento y señales temporales mediante formulaciones verificables.
2. **Artístico/experiencial:** transformar esas señales en una interacción visual atractiva, clara, fluida y fácil de explorar, sin exigir conocimientos técnicos a quien utilice la app.

### 0.1 Output esperado

**Resultado mínimo obligatorio (MVP evaluable):**

- Aplicación de escritorio ejecutable con **webcam**, al menos un detector manual (**Frame Differencing**) y limpieza de máscara hecha con operaciones propias.
- **Al menos dos efectos artísticos funcionando en tiempo real:** **Rastros de luz** y **Partículas**.
- Interfaz de estudio creativo, con vista previa, cambio de modo, sensibilidad, intensidad, paleta, limpieza del lienzo, iniciar/pausar y exportación de **captura PNG**.
- Panel de depuración que permita visualizar **video original, diferencia, máscara cruda, máscara limpia y obra final**; sin falsificar resultados.
- Código fuente organizado, pruebas automatizadas de las transformaciones, medición de FPS/tiempos y evidencia documentada de casos de prueba.
- Reporte técnico, explicación de fórmulas y una demostración grabada o en vivo.

**Alcance ampliado preferente:** segundo detector manual (**Running Average**), modos **Silueta** y **Experimental**, presets visuales, guardado de configuración y exportación de una secuencia o video si la plataforma lo permite.

**Extensión avanzada condicionada al tiempo:** flujo óptico **Lucas–Kanade** implementado manualmente, aplicado a partículas direccionales. No forma parte de la definición de terminado del MVP.

### 0.2 Lo que NO promete el proyecto

- No reconoce identidades, poses, manos ni personas mediante redes neuronales.
- Una **máscara de movimiento no equivale a una segmentación completa de la persona**. Si alguien queda quieto, Frame Differencing puede dejar de detectarlo; Running Average puede terminar incorporándolo al fondo.
- No garantiza resultados de FPS antes de medirlos sobre el equipo real.
- No requiere internet, servicios cloud, cuentas de usuario ni almacenar video por defecto.
- No utiliza `cv2.createBackgroundSubtractorMOG2`, `cv2.createBackgroundSubtractorKNN`, `cv2.calcOpticalFlowPyrLK`, `cv2.absdiff`, `cv2.threshold`, `cv2.blur`, `cv2.GaussianBlur`, `cv2.erode`, `cv2.dilate`, `cv2.morphologyEx`, `cv2.connectedComponents` o `cv2.findContours` para sustituir las implementaciones académicas.

### 0.3 Origen de los requisitos

- **Base de la propuesta inicial:** webcam, detección por diferencia, limpieza, persistencia, partículas, rastros, silueta, experimentación visual, Python/OpenCV/NumPy/Pygame, plan de cuatro semanas y mínimo dos efectos.
- **Extensión técnica discutida después:** comparación con Running Average y Lucas–Kanade, parámetros iniciales de filtrado, apertura/cierre, componentes conectados y suavizado temporal.
- **Decisión de diseño de este plan:** una sola interfaz Pygame/pygame-ce con componentes visuales personalizados, evitando integrar dos bucles gráficos distintos; esta elección no altera los objetivos científicos originales.

---

## 1. Tecnologías, dependencias y responsabilidades

| Tecnología | Uso específico | Alcance y restricciones | Prioridad |
|---|---|---|---|
| **Python 3.12** | Lenguaje, paquetes, estructuras de datos, módulos y coordinación | Versión base propuesta; compatibilidad validada con dependencias instaladas | Obligatoria |
| **NumPy** | Arreglos de píxeles, diferencias, umbrales, convoluciones propias, máscaras, álgebra lineal, métricas | Permitidas operaciones de bajo nivel y vectorización; no sustituye el diseño del algoritmo | Obligatoria |
| **OpenCV (`opencv-python`)** | `VideoCapture`, lectura de fotogramas, elección de dispositivo, metadatos, liberación de cámara; `VideoWriter` únicamente si se activa la exportación de video | Sin llamadas de alto nivel que resuelvan el procesamiento central | Obligatoria |
| **Pygame Community Edition (`pygame-ce`)** | Ventana, eventos de mouse/teclado, superficies, tipografía, dibujo, UI y composición visual | API importada como `pygame`; se emplea **pygame-ce** como única distribución de Pygame | Obligatoria |
| **pytest** | Pruebas unitarias, regresión, fixtures y casos sintéticos reproducibles | El motor matemático no requiere cámara durante los tests | Obligatoria (desarrollo) |
| **Ruff** | Formato y comprobaciones estáticas | Herramienta de calidad; no forma parte del procesamiento | Recomendable |
| **Git + GitHub** | Historial, respaldos, referencias y evidencias del proceso | Sin dependencia de GitHub durante la ejecución | Recomendable |
| **VS Code / Cursor** | Entornos de desarrollo | No forman parte de las dependencias de ejecución | Entorno |
| **Inter** | Texto, botones y datos técnicos de la interfaz | Fuente abierta bajo OFL; no adjuntar fuentes a documentos de entrega externos salvo que corresponda | Visual |
| **Cormorant Garamond** | Títulos grandes y carácter editorial | Fuente abierta bajo OFL; compatible con la estética de referencia | Visual |
| **Biblioteca estándar** | `dataclasses`, `enum`, `pathlib`, `collections`, `json`, `logging`, `time`, `math`, `typing`, `random` | Preferencia sobre dependencias adicionales | Obligatoria |

**No previstas para el MVP:** TensorFlow, PyTorch, MediaPipe, scikit-image, SciPy, Qt, Streamlit, web framework, GPU/CUDA, backend web o base de datos. La ausencia de estas dependencias mantiene el enfoque en algoritmos propios y simplifica la demostración en una laptop.

**Instalación reproducible:** `pyproject.toml` con dependencias compatibles y archivo de bloqueo resuelto en el entorno de desarrollo. Las versiones finales se fijarán tras comprobar instalación y ejecución en el sistema objetivo; no se asumen números de versión o benchmarks no verificados.

Documentación base: [Python](https://docs.python.org/3/), [NumPy](https://numpy.org/doc/stable/), [OpenCV VideoCapture](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html), [pygame-ce](https://pyga.me/docs/), [pytest](https://docs.pytest.org/).

---

## 2. Arquitectura general: dos módulos principales

```text
WEBCAM / FUENTE DE PRUEBA
        |
        v
+--------------------------------------------------------------+
| MÓDULO 1 — MOTOR MATEMÁTICO Y TRANSFORMACIONES               |
| Adquisición -> preparación -> detección -> limpieza          |
| -> componentes -> características -> señales de movimiento   |
+---------------------------+----------------------------------+
                            |
                     MotionResult
             (fotograma, máscaras, regiones,
                 movimiento, métricas, estado)
                            |
                            v
+--------------------------------------------------------------+
| MÓDULO 2 — INTERFAZ Y EXPERIENCIA ARTÍSTICA                   |
| Control de sesión -> efectos -> compositor -> lienzo         |
| -> widgets y presets -> exportaciones -> experiencia visual  |
+--------------------------------------------------------------+
                            |
                            v
          VISTA INTERACTIVA / PNG / VIDEO OPCIONAL
```

El flujo **no** se bifurca en una aplicación científica y otra artística: ambas funciones conviven en una sola ventana. La separación existe **en el código y en los contratos de datos**, no como programas independientes.

### 2.1 Contrato de entrada/salida entre módulos

```python
@dataclass(frozen=True)
class MotionRegion:
    id: int
    area_px: int
    centroid_xy: tuple[float, float]
    bbox_xywh: tuple[int, int, int, int]

@dataclass
class MotionResult:
    timestamp_s: float
    frame_rgb: np.ndarray       # uint8, shape (H, W, 3)
    difference: np.ndarray      # float32 o uint8, shape (H, W)
    raw_mask: np.ndarray        # bool, shape (H, W)
    clean_mask: np.ndarray      # bool, shape (H, W)
    regions: list[MotionRegion]
    motion_fraction: float      # fracción 0..1 de píxeles positivos
    mean_motion_energy: float   # estadístico de diferencia
    flow_vectors: np.ndarray | None  # opcional: (N, 4) [x, y, u, v]
    detector_name: str
    status: str                 # warmup, ready, camera_error, etc.
    processing_ms: float
```

**Invariantes:** máscaras con las mismas dimensiones que el fotograma procesado; `raw_mask` y `clean_mask` booleanas; coordenadas en el sistema del procesamiento (no de la pantalla); marcas temporales monótonas; errores sin regiones ficticias; diferencias e imágenes copiadas solo cuando sea necesario.

**Coordenadas:** `numpy` utiliza `(y, x)` para acceder a la imagen; la UI y las partículas usan `(x, y)`. La conversión entre espacio de cámara y espacio de lienzo se centraliza, conservando relación de aspecto y documentando el recorte o letterboxing.

### 2.2 Ciclo de aplicación

```text
poll_events -> update_settings -> acquire_frame -> process_frame
            -> update_effects(dt) -> render_scene -> present_frame
```

- Una sola hebra de UI/render para la primera versión. Una posible hebra de captura queda fuera del MVP y requeriría cola acotada y descarte de frames antiguos.
- La renderización no modifica las matrices del detector.
- Cambiar de detector, cámara o resolución reinicia sus estados temporales, evitando comparar imágenes incompatibles.
- El bucle se mantiene interactivo aunque la cámara falle, con indicación visible y posibilidad de reintentar.

---

# MÓDULO 1 — MOTOR MATEMÁTICO Y TRANSFORMACIONES

## 3. Captura y preparación de fotogramas

**Salida canónica:** RGB `uint8` de forma `(H, W, 3)`; escala de grises calculada manualmente en `float32`; resolución de procesamiento inicial **320 × 240** (ancho × alto). La cámara puede entregar 640 × 480 u otra resolución; el tamaño efectivo se confirma tras la apertura. Una resolución mayor no se interpreta automáticamente como mejor detección.

**Funciones previstas:**

| Función / clase | Entrada → salida | Responsabilidad |
|---|---|---|
| `CameraSource.open(index, width, height)` | parámetros → estado | Abrir dispositivo; solicitar resolución y consultar resolución real |
| `CameraSource.read()` | — → `(ok, frame_bgr, timestamp)` | Adquirir fotograma; identificar cortes y errores |
| `CameraSource.release()` | — → `None` | Liberación segura incluso ante excepciones |
| `SyntheticFrameSource.read()` | configuración → imagen sintética | Probar el sistema sin webcam; **modo demo claramente etiquetado** |
| `bgr_to_rgb(frame)` | BGR → RGB | Reordenación explícita de canales mediante NumPy |
| `rgb_to_grayscale(frame)` | RGB → gris `float32` | `0.299R + 0.587G + 0.114B` como aproximación de luminancia tipo Rec. 601 |
| `resize_nearest(frame, w, h)` | matriz → matriz | Redimensionado por índice/vecino más cercano, si la cámara no respeta la resolución pedida |
| `normalize_frame(frame)` | matriz → validada | Verificación de forma, canales, dtype, finitud y rango |

**Precaución numérica:** para obtener diferencias de intensidades se realiza primero una conversión a entero con signo o `float32`. Restar dos arreglos `uint8` directamente puede causar desbordamiento modular y diferencias incorrectas.

## 4. Detección de movimiento: tres alternativas

### 4.1 A — Diferencia entre fotogramas (obligatoria, línea base)

Dado un fotograma gris actual `I_t` y el previo `I_(t-1)`:

```text
D_t(x,y) = | I_t(x,y) - I_(t-1)(x,y) |
M_t(x,y) = 1 si D_t(x,y) > T; de otro modo 0
```

**Proceso:** primer frame → estado de calentamiento; siguientes frames → diferencia absoluta → eventual filtrado de la diferencia → umbral → máscara. **T inicial: 20–30** para grises de ocho bits (configurable). La comparación se aplica antes de actualizar la referencia previa.

**Ventaja:** simple, interpretable y rápida. **Límite:** detecta cambio temporal; una persona completamente quieta deja de activarlo. Los contornos de un objeto en movimiento pueden aparecer más prominentes que su interior.

**Contrato:** `FrameDiffDetector.process(gray, settings) -> DetectorOutput` con `difference` y `raw_mask`; `reset()` borra el fotograma anterior.

### 4.2 B — Promedio móvil del fondo (ampliación preferente)

```text
D_t(x,y) = | I_t(x,y) - B_(t-1)(x,y) |
M_t(x,y) = 1 si D_t(x,y) > T; de otro modo 0
B_t(x,y) = (1 - alpha) B_(t-1)(x,y) + alpha I_t(x,y)
```

`B` es un modelo de fondo **float32**. Primero se detecta con el fondo previo y **después** se actualiza; esto evita atenuar artificialmente la diferencia con el propio frame actual. Valor exploratorio de `alpha`: **0.01–0.08 por fotograma**, ajustable según FPS, estabilidad de iluminación y rapidez del fondo. Una formulación basada en tiempo puede usar `alpha = 1 - exp(-dt/tau)` para que el comportamiento sea más consistente entre FPS diferentes.

**Inicialización:** con el primer frame o un breve promedio de frames sin movimiento; calentamiento visible y reinicio del fondo desde la interfaz.

**Ventaja:** tolera cambios graduales de fondo. **Límite:** un objeto inmóvil acaba incorporándose al fondo; saltos súbitos de luz generan cambios importantes. No se promete que siempre supere a A.

**Contrato:** `RunningAverageDetector.process(gray, settings)` y `reset_background()`.

### 4.3 C — Flujo óptico Lucas–Kanade (extensión opcional)

La hipótesis de constancia aproximada de brillo y movimiento local pequeño conduce a:

```text
I_x * u + I_y * v + I_t = 0

A = [[I_x1, I_y1], ..., [I_xn, I_yn]]
b = [-I_t1, ..., -I_tn]
(u, v) = argmin ||A [u, v]^T - b||^2
```

**Propuesta concreta:** versión **dispersa** sobre puntos con gradientes informativos; gradientes por diferencias finitas propias; ventana local inicial de **5×5 o 7×7**; mínimos cuadrados con `numpy.linalg.lstsq` o solución 2×2 protegida; descarte de regiones mal condicionadas; número acotado de puntos. No se emplea `cv2.goodFeaturesToTrack` ni `cv2.calcOpticalFlowPyrLK`.

**Limitaciones:** apertura (aperture problem), regiones uniformes, movimientos grandes, oclusiones y cambios de iluminación. Una pirámide multiescala sería una mejora **posterior**, no parte de la definición del MVP. El flujo óptico estima movimiento aparente de la imagen, **no velocidad física en metros/segundo**.

**Contrato opcional:** `LucasKanadeDetector.estimate(prev_gray, curr_gray, points_xy) -> Nx4` con `(x, y, u, v)` válidos más indicador de calidad; el resultado alimenta la dirección de las partículas.

### 4.4 Criterios comparativos sin resultados inventados

| Criterio | Frame Differencing | Running Average | Lucas–Kanade disperso |
|---|---|---|---|
| Complejidad de implementación manual | Baja | Media | Alta |
| Costo cualitativo previsto | Bajo | Bajo a medio | Medio a alto |
| Señal obtenida | Cambio entre frames | Cambio respecto al fondo | Dirección y desplazamiento local |
| Principal beneficio artístico | Estelas / movimiento instantáneo | Siluetas de regiones cambiantes | Partículas orientadas por el movimiento |
| Riesgo principal | Desaparece al detenerse | Absorbe objetos quietos | Fallos en textura y grandes desplazamientos |
| Prioridad | **MVP** | **Ampliación** | **Opcional** |

El costo depende de resolución, hardware, número de puntos y calidad de implementación. El cuadro es una **estimación cualitativa**, no un benchmark.

## 5. Limpieza de máscara: operaciones propias

**Orden operativo recomendado (sin confundir suavizado previo con limpieza binaria):**

```text
Frame/Fondo -> mapa D -> [filtro espacial opcional] -> umbral T
            -> apertura / cierre según el ruido
            -> componentes conectados + descarte de áreas
            -> máscara binaria limpia
            -> [suavizado temporal opcional para intensidad visual]
```

| Operación | Especificación | Valor inicial experimental | Función prevista |
|---|---|---|---|
| Filtro promedio | Promedio espacial con kernel normalizado | 3×3; alternativamente 5×5 | `mean_filter(image, kernel_size)` |
| Umbral binario | `M = D > T` | T = 20–30, escala de 8 bits | `binary_threshold(diff, threshold)` |
| Erosión | Un píxel permanece activo si todos los vecinos de la estructura están activos | Estructura 3×3; 1 iteración | `binary_erode(mask, structure, iterations)` |
| Dilatación | Un píxel es activo si algún vecino de la estructura está activo | Estructura 3×3; 1 iteración | `binary_dilate(mask, structure, iterations)` |
| Apertura | Erosión y luego dilatación | 3×3, 1 pasada inicial | `binary_open(mask, structure)` |
| Cierre | Dilatación y luego erosión | 3×3, 1 pasada inicial | `binary_close(mask, structure)` |
| Componentes conectados | Etiquetas de regiones binarias, conectividad 4 u 8 | 8 vecinos; área mínima 20–100 px a 320×240 | `label_components(mask, connectivity)` |
| Descarte por área | Conserva componentes con área ≥ `min_area` | 50 px como preset inicial, ajustable | `remove_small_components(mask, min_area)` |
| Suavizado temporal | `S_t = beta*M_t + (1-beta)*S_(t-1)` | beta = 0.4–0.7 | `temporal_smooth(mask, previous, beta)` |

**Decisiones importantes:**

- El filtro promedio sobre una imagen binaria produciría grises: cuando se utilice, preferentemente **antes** del umbral sobre el mapa de diferencias.
- Abrir y cerrar no siempre mejora simultáneamente la máscara; su aplicación y orden serán configurables. **Demasiada erosión elimina dedos y trazos finos**.
- El tratamiento de bordes se define explícitamente (padding con fondo cero para morfología binaria y replicación/reflexión para filtro promedio), evitando cambios accidentales en las esquinas.
- Para acelerar sin perder autoría, son válidas operaciones elementales de arreglos, cortes, `numpy.any`, `numpy.all` y vistas de ventanas. No se delega la operación completa a `cv2` o `scipy.ndimage`.
- Los componentes conectados pueden partir de **BFS/flood-fill iterativo**; una implementación **dos pasadas + Union-Find** ofrece un camino de optimización y mayor valor académico.
- El suavizado temporal da valores flotantes entre 0 y 1. Sirve directamente para opacidad/persistencia o requiere una nueva umbralización si se quiere volver a una máscara estrictamente binaria.
- El mínimo de área en píxeles depende de la resolución; a otra resolución se ajusta proporcionalmente a la superficie o se calibra de nuevo.

### 5.1 Datos derivados de la máscara

Funciones previstas:

- `compute_motion_fraction(mask)` → proporción de píxeles activos.
- `compute_motion_centroid(mask)` → centro de masa de la actividad, o `None` si la máscara está vacía.
- `component_stats(labels)` → área, caja envolvente y centroide de cada región.
- `sample_emission_points(mask, regions, max_points, seed)` → muestras reproducibles para partículas sin crear una por cada píxel.
- `track_region_centroids(previous_regions, current_regions, max_distance)` → asociación temporal simple y desplazamientos aproximados; **no** identificaciones persistentes de personas.
- `compute_motion_energy(diff, clean_mask)` → intensidad media del cambio, normalizada y documentada.
- `transform_coordinates(xy, input_size, canvas_rect, mirror)` → posición real en el lienzo, sin distorsión de aspecto.

## 6. Pruebas deterministas del Módulo 1

La unidad de prueba es una matriz sintética: los resultados esperados se calculan a mano y permiten verificar la implementación sin cámara.

| Caso | Entrada preparada | Resultado esperado |
|---|---|---|
| Diferencia | Dos matrices idénticas | Diferencia cero; máscara vacía |
| Cambio puntual | Un píxel pasa de 10 a 200 | Detección en la posición correcta |
| Overflow | Dos píxeles `uint8` con 0 y 255 | Diferencia 255, no envoltura modular |
| Umbral exacto | Diferencias `T-1`, `T`, `T+1` | Solo `T+1` activo si la condición es `>` |
| Erosión | Bloque 3×3 + punto aislado | Punto aislado eliminado; bloque encogido |
| Dilatación | Un único punto blanco | Expansión según elemento estructurante |
| Apertura | Región grande + salpicaduras | Se preserva región útil; salpicaduras pequeñas desaparecen |
| Cierre | Región grande con hueco pequeño | Se reduce o elimina el hueco |
| Componentes | Dos cuadrados disjuntos | Dos etiquetas, áreas y centroides correctos |
| Conectividad | Dos píxeles diagonalmente vecinos | Separados en 4-conexión; unidos en 8-conexión |
| Running Average | Frames constantes | Fondo estable y sin detecciones persistentes |
| Suavizado temporal | Serie conocida 0–1 | Coincidencia con recurrencia calculada manualmente |
| Reconfiguración | Cambio de resolución/detector | Estado temporal reiniciado, sin excepción |
| Frames nulos | Cámara devuelve lectura fallida | Estado de error; UI sigue respondiendo |
| LK opcional | Translación sintética pequeña | Vectores cercanos al desplazamiento dentro de tolerancia explicada |

Las pruebas de integración utilizan una fuente sintética que desplaza cuadrados a diferentes velocidades, introduce ruido localizado y simula cambio gradual/repentino de iluminación.

---

# MÓDULO 2 — INTERFAZ Y EXPERIENCIA ARTÍSTICA

## 7. Concepto de diseño: un estudio creativo, no un panel industrial

La interfaz está inspirada en las referencias visuales originales de **Motion Art Lab**: turquesa suave, negro profundo, acentos coral delgados, tipografía editorial y arte basado en movimiento. La pantalla de trabajo debe sentirse **luminosa, limpia, delicada, experimental y divertida**. La imagen artística es la protagonista; los controles son comprensibles desde el primer uso.

### 7.1 Sistema visual propuesto

| Token | Valor sugerido | Función |
|---|---|---|
| `bg.canvas` | `#11191F` | Lienzo cinematográfico para luz y partículas |
| `bg.app` | `#F7F8F6` | Fondo general claro |
| `surface` | `#FFFFFF` | Paneles y tarjetas |
| `surface.aqua` | `#B9E9E6` | Tarjetas/presets suaves |
| `brand.teal` | `#3A9E9D` | Selección y enlaces |
| `brand.coral` | `#D78F7A` | Detalles artísticos, esquinas/guías |
| `text.primary` | `#15282F` | Texto principal de interfaz |
| `text.muted` | `#65767B` | Etiquetas secundarias |
| `border.subtle` | `#DCE8E6` | Separación sin marcos pesados |
| `glow.aqua` | `#7DE5DB` | Paleta para rastros |

**Tipografía:** Cormorant Garamond (títulos editoriales) + Inter (controles y explicaciones). Tamaño de controles legible, contraste alto y foco claramente visible. Los colores anteriores son **tokens iniciales de diseño**, no mediciones del material de referencia.

**Dirección artística:** grandes espacios en blanco, tarjetas de borde sutil, esquinas suavemente redondeadas (12–20 px), animaciones cortas, muy pocas sombras, líneas coral en esquinas como guiño a las diapositivas, degradados suaves solo donde benefician el arte. Nada de estética recargada tipo dashboard financiero o terminal de hackers.

### 7.2 Pantalla principal — «Estudio»

```text
+-----------------------------------------------------------------------------+
| MOTION ART LAB      Estudio   Galería   Acerca de           [Ayuda]  [● Cam] |
+-----------------------------------------------------------------------------+
|                                                                             |
|  Crea con movimiento                         ESTADO: En vivo  ·  24 FPS*    |
|  Cada gesto deja una huella.                                               |
|                                                                             |
| +------------------------------------------------+ +----------------------+ |
| |                                                | | TU LIENZO            | |
| |                                                | | Modo:  Rastros de luz| |
| |             LIENZO INTERACTIVO                 | | [Rastros][Partículas]| |
| |             video + arte en vivo              | | [Silueta][Experimento]| |
| |                                                | |                      | |
| |                                                | | Sensibilidad   ----o| |
| |                                                | | Intensidad      ---o| |
| |                                                | | Persistencia    --o | |
| |                                                | |                      | |
| +------------------------------------------------+ | Paleta   [A][B][C]  | |
|  [Iniciar/Pausar] [Limpiar] [Capturar PNG]         | | [Avanzado ▾]        | |
|                                                  +----------------------+ |
|  Vistas: [Arte final] [Original] [Diferencia] [Máscara] [Comparar]          |
+-----------------------------------------------------------------------------+
|  LOCAL · No se envía video a internet             [?] Atajos / Guía         |
+-----------------------------------------------------------------------------+
```

`24 FPS*` es un ejemplo de ubicación en wireframe, **no una cifra medida o garantizada**; la implementación debe mostrar el valor real o `—` mientras no exista medición.

**Adaptabilidad:** diseño pensado inicialmente para una ventana de **1280×800**; mínimo usable propuesto **960×640**; controles con scroll y vista previa que conserva relación de aspecto. Un modo **Presentación** oculta paneles y amplía el lienzo para proyector, con salida fácilmente reversible.

### 7.3 Pantallas y estados

1. **Bienvenida:** logo/título editorial, subtítulo «Convierte tus movimientos en arte», ilustración de efecto generativo (no video ajeno), botón «Entrar al estudio» y nota de uso local de cámara.
2. **Estudio:** cámara, lienzo, efecto seleccionado y controles inmediatos.
3. **Laboratorio / Diagnóstico:** panel técnico opcional para examen: fotograma original, diferencia, máscara bruta, máscara limpia y render final; estadísticas reales de FPS, tiempos y área activa.
4. **Galería local:** mosaico de capturas exportadas, fecha, modo, preset; archivos almacenados localmente y apertura mediante el sistema operativo si se implementa.
5. **Acerca de / Cómo funciona:** explicación sencilla del pipeline, distinción entre movimiento y silueta completa, créditos y enlaces a referencias.
6. **Estados especiales:** sin cámara/permisos, cámara desconectada, calentamiento de fondo, pausa, exportación exitosa/fallida, parámetros restaurados y modo demo sintético claramente marcado.

### 7.4 Controles visibles y experiencia de uso

| Control | Comportamiento | Estado / retorno visible |
|---|---|---|
| Selector de cámara | Cambia dispositivo; reinicia el estado del detector | Nombre/índice, conexión, error recuperable |
| Inicio/Pausa | Pausa captura/procesamiento de forma coherente; al reanudar gestiona referencia temporal | Etiqueta y estado consistentes |
| Selector de modo | Rastros, Partículas, Silueta, Experimental | Modo seleccionado con acento turquesa |
| Sensibilidad | Cambia umbral (relación inversa explícita: mayor sensibilidad → T menor) | Número T o lectura explicativa |
| Intensidad | Escala cantidad/opacidad de efecto | Actualización visual inmediata |
| Persistencia | Ajusta decaimiento del rastro | Vista con más o menos estela |
| Paleta | Cambia gradiente de efectos, no el análisis de máscara | Muestra de colores seleccionados |
| Espejo | Invierte imagen de presentación y coordenadas del arte de forma consistente | Indicador «Efecto espejo» |
| Limpiar | Vacía superficies de persistencia y partículas sin reiniciar necesariamente cámara | Lienzo limpio inmediatamente |
| Captura PNG | Exporta composición visible, con o sin paneles según opción | Confirmación + ruta real |
| Diagnóstico | Revela vistas intermedias y parámetros | Lecturas numéricas reales |
| Reiniciar fondo | Exclusivo de Running Average | Mensaje de calentamiento |
| Restaurar valores | Vuelve al preset predeterminado | Restablecimiento verificable |

**Atajos sugeridos:** `Espacio` iniciar/pausar, `1–4` cambiar modo, `C` limpiar, `S` guardar PNG, `D` diagnóstico, `F` presentación, `H` ayuda, `Esc` cerrar panel/presentación antes de salir. Los atajos aparecen en una ayuda breve y no sustituyen los botones.

### 7.5 Componentes visuales propios (Pygame)

- `Theme`: colores, estilos tipográficos, radios, espaciado, escalas y estados.
- `SceneManager`: navegación entre bienvenida, estudio, galería y ayuda.
- `Widget` base: rectángulo, estado, hit-testing, `update(dt)`, `draw(surface)` y eventos.
- `Button`, `IconButton`, `Toggle`, `Slider`, `SegmentedControl`, `Dropdown`, `ColorSwatch`, `Tooltip`, `Toast`, `Modal`, `Card`, `ScrollArea`.
- `LayoutManager`: anclajes, márgenes, columnas y reflow al redimensionar.
- `FocusManager`: teclado, foco visible, orden lógico y navegación sin mouse.
- `CameraStatus`: conexión/permisos/calentamiento con mensajes no técnicos.
- `ParameterPanel`: controles básicos y expansión «Ajustes avanzados».
- `DebugPanel`: modos de vista, histogramas/estadísticos sencillos (solo si hay datos reales).
- `GalleryView`: lista local de capturas y estado vacío atractivo.

Los widgets se dibujan con formas, tipografía y superficies de Pygame. Una interfaz personalizada requiere más trabajo que widgets predefinidos, pero mantiene una sola tecnología gráfica y permite reproducir la estética propia de Motion Art Lab.

## 8. Efectos visuales y su lógica

### 8.1 Modo Rastros de luz — obligatorio

**Concepto:** el movimiento deja pinceladas luminosas que aparecen, continúan y desaparecen suavemente.

**Señal:** centros de regiones, muestras de máscara y posiciones históricas. **Dinámica:** segmentos conectados entre posiciones sucesivas cuando el movimiento es suficientemente coherente. **Persistencia:**

```text
Trail_t = gamma(dt) * Trail_(t-1) + NewStroke_t
0 < gamma(dt) < 1
```

`gamma(dt) = exp(-lambda * dt)` evita que el desvanecimiento cambie drásticamente con el FPS. El resultado se limita al rango de color antes de dibujar. Las líneas tienen grosor, color y opacidad variable; el lienzo conserva una textura luminosa sin dejar manchas permanentes.

**Funciones previstas:** `update_trails(regions, time_s)`, `append_stroke(point, color)`, `prune_old_strokes()`, `fade_trail_surface(dt)`, `render_trails(surface)`.

### 8.2 Modo Partículas — obligatorio

**Concepto:** motas luminosas nacen cerca de regiones con movimiento y siguen reglas simples de vida, dirección, fricción y desvanecimiento.

**Estado de partícula:** `(position_xy, velocity_xy, age_s, lifetime_s, size_px, color, alpha)`. Física discreta dependiente de tiempo:

```text
position_(t+dt) = position_t + velocity_t * dt
velocity_(t+dt) = velocity_t * exp(-drag * dt) + acceleration * dt
alpha = max(0, 1 - age_s / lifetime_s) ^ fade_power
```

El número de emisiones depende de la actividad y se encuentra **acotado por presupuesto**; no se genera una partícula por píxel activo. Si Lucas–Kanade se incorpora, la dirección inicial puede tomar `(u,v)` de un flujo válido; de lo contrario se utiliza desplazamiento aproximado de regiones o un campo creativo seeded.

**Funciones:** `spawn_particles(motion_result, palette, budget)`, `update_particles(dt)`, `cull_particles()`, `render_particles(surface)`, `reset_particles()`.

### 8.3 Modo Silueta abstracta — ampliación

El color rellena **regiones activas de movimiento** o contornos derivados de la máscara. No debe etiquetarse como segmentación anatómica o silueta completa cuando el detector solo mide cambio temporal. Posibles acabados: tintas aguamarina, transparencias superpuestas, borde difuso construido con dilataciones propias.

**Funciones:** `mask_to_colored_layer(clean_mask, palette)`, `render_silhouette_layer(surface)`.

### 8.4 Modo Experimental — ampliación

Permite combinar rastros + partículas y modificar parámetros de forma controlada (densidad, paleta, jitter, mezcla, decaimiento). Los presets deben ser reproducibles mediante una semilla de aleatoriedad guardada en la configuración.

**Funciones:** `compose_effects(active_modes, motion_result, dt)`, `apply_preset(preset)`, `reset_experiment()`.

### 8.5 Compositor, cámara y superficies

Orden recomendado: fondo sólido/gradiente → video opcional con opacidad ajustable → visualización opcional de máscara → capa de persistencia → partículas → elementos de interfaz. El arte puede mostrarse **sobre el video** o sobre **fondo oscuro limpio** sin alterar el detector.

**Conversión NumPy → Pygame:** las imágenes de procesamiento usan `(H, W, 3)`; `pygame.surfarray` interpreta el primer eje espacial como `X`, de modo que el arreglo para crear una superficie usa `(W, H, 3)` (por ejemplo, transposición de los dos primeros ejes). Véase la [documentación de surfarray](https://pyga.me/docs/ref/surfarray.html).

**Funciones:** `frame_to_surface(rgb)`, `draw_video_layer()`, `draw_mask_preview()`, `composite_layers()`, `capture_canvas_png(path)`, `capture_full_scene_png(path)`.

### 8.6 Paletas artísticas iniciales

| Preset | Colores | Sensación |
|---|---|---|
| **Aurora** | Turquesa, aguamarina, blanco cálido | Fresco / etéreo |
| **Coral Bloom** | Coral, rosa pálido, crema | Orgánico / artístico |
| **Nocturno** | Azul eléctrico, violeta, cian | Futurista / inmersivo |
| **Monocromo** | Blanco suave y grises sobre negro | Editorial / minimalista |

El color es parte de la interpretación estética, no una indicación de certeza de detección ni de velocidad física.

## 9. Usabilidad, exportación y privacidad

- **Primer uso:** pantalla breve que explica permisos de cámara, qué hace el programa y dónde quedan las imágenes. Nada se guarda ni transmite sin elección explícita.
- **Modo sin cámara:** imágenes sintéticas generadas localmente y marca visible «Demo sintética». Permite enseñar interfaz/algoritmos sin afirmar que se trata de una captura real.
- **Manejo de errores:** sin cámara, permiso denegado, desconexión, imagen corrupta, FPS reducido, exportación fallida. Ninguno bloquea toda la interfaz.
- **Exportaciones:** PNG con nombre basado en fecha/hora y modo, dentro de `outputs/gallery/`; JSON opcional con parámetros de generación, detector, resolución y FPS medido. La generación de video es posterior y requiere verificar códec/contenerización.
- **Persistencia de ajustes:** `config/user_settings.json` para preferencias no sensibles, validación de rangos y valores predeterminados ante archivo corrupto.
- **Privacidad:** procesamiento local, sin reconocimiento facial, sin telemetría ni grabación automática. La ventana informa cuándo la cámara está activa.
- **Accesibilidad:** texto con contraste, botones de tamaño cómodo, estados no transmitidos solo por color, controles usables con mouse y atajos, sliders con lectura numérica.
- **Rendimiento:** control del máximo de partículas y del número de puntos de emisión; tiempos separados de captura, procesamiento y render. Objetivo de diseño **aspiracional** de fluidez en 320×240 y presentación a ~30 FPS; el logro real se valida con mediciones, no se presupone.

---

## 10. Organización prevista del repositorio

```text
motion-art-lab/
├── README.md
├── MOTION_ART_LAB_PLAN_IMPLEMENTACION.md
├── pyproject.toml
├── .gitignore
├── LICENSE                         # licencia del proyecto por decidir
├── src/
│   └── motion_art_lab/
│       ├── __init__.py
│       ├── main.py                 # punto de entrada
│       ├── config.py               # dataclasses, presets, validación
│       ├── contracts.py            # MotionResult, MotionRegion
│       ├── module1_processing/
│       │   ├── sources.py          # webcam / sintetizador
│       │   ├── preprocessing.py    # BGR/RGB/gris, resize, validación
│       │   ├── frame_difference.py
│       │   ├── running_average.py
│       │   ├── lucas_kanade.py     # extensión opcional
│       │   ├── filters.py
│       │   ├── morphology.py
│       │   ├── components.py
│       │   ├── temporal.py
│       │   ├── motion_features.py
│       │   └── pipeline.py
│       ├── module2_experience/
│       │   ├── app.py              # event loop
│       │   ├── theme.py
│       │   ├── layout.py
│       │   ├── widgets.py
│       │   ├── scenes.py
│       │   ├── studio.py
│       │   ├── debug_view.py
│       │   ├── effects/
│       │   │   ├── trails.py
│       │   │   ├── particles.py
│       │   │   ├── silhouette.py
│       │   │   └── compositor.py
│       │   ├── gallery.py
│       │   ├── input_actions.py
│       │   └── export.py
│       └── metrics.py
├── assets/
│   ├── fonts/                      # fuente y licencia según OFL
│   ├── icons/                      # originales o con licencia verificada
│   └── previews/                   # ejemplos propios, no resultados ficticios
├── config/
│   └── default_presets.json
├── outputs/
│   ├── gallery/                    # ignorado por Git
│   └── reports/                    # ignorado por Git si son personales
├── tests/
│   ├── test_frame_difference.py
│   ├── test_running_average.py
│   ├── test_filters_morphology.py
│   ├── test_components.py
│   ├── test_temporal.py
│   ├── test_motion_features.py
│   ├── test_particles.py
│   └── test_pipeline_integration.py
├── docs/
│   ├── algorithms.md
│   ├── methodology.md
│   ├── test_plan.md
│   ├── experiments.md
│   └── REFERENCES.md               # se puede extraer del apéndice inferior
└── results/
    └── evidence/                   # capturas propias y matrices de pruebas
```

Los nombres de archivos reflejan **responsabilidades**, no evidencia de que existan actualmente. Las pruebas del Módulo 1 no dependen de importar Pygame; los efectos reciben `MotionResult` sin importar directamente OpenCV.

## 11. Configuración inicial y contratos

```python
@dataclass(frozen=True)
class ProcessingSettings:
    detector: str = "frame_diff"
    width: int = 320
    height: int = 240
    threshold: int = 25
    mean_filter_size: int = 3
    morph_kernel_size: int = 3
    opening_iterations: int = 1
    closing_iterations: int = 0
    component_connectivity: int = 8
    min_component_area: int = 50
    running_alpha: float = 0.03
    temporal_beta: float = 0.55
    temporal_smoothing_enabled: bool = False

@dataclass(frozen=True)
class ArtisticSettings:
    mode: str = "trails"
    palette: str = "aurora"
    intensity: float = 0.65
    persistence: float = 0.65
    particle_limit: int = 350
    mirror_preview: bool = True
    show_camera_background: bool = False
    show_diagnostics: bool = False
```

**Nota:** son defaults experimentales **sujetos a ajuste**. En particular, `opening_iterations=1` y `closing_iterations=0` parten de una limpieza conservadora; el cierre se habilita cuando las pruebas muestren huecos. El efecto de aumentar el slider «Sensibilidad» se traduce en **reducir** `threshold` dentro de un rango seguro, no en incrementar ambos sin explicación.

## 12. Fases y cronograma del proyecto

La **propuesta original contempla cuatro semanas**. El siguiente orden preserva ese horizonte para el objetivo mínimo y sitúa las mejoras fuera de la ruta crítica.

| Fase | Duración propuesta | Desarrollo del Módulo 1 | Desarrollo del Módulo 2 | Evidencia / criterio de salida |
|---|---|---|---|---|
| **F1 — Captura y detección** | Semana 1 | Cámara, RGB/gris, Frame Differencing, umbral y pruebas de diferencia | Ventana inicial, vista original/máscara, estados de cámara | Máscara de movimiento visible con ejemplos sintéticos y webcam; pruebas básicas correctas |
| **F2 — Limpieza y rastros** | Semana 2 | Promedio espacial, morfología, componentes, estadísticas | Lienzo, modo Rastros, persistencia, controles primarios | Rastro visual responde a movimiento; se ve diferencia entre máscara cruda y limpia |
| **F3 — Partículas e interfaz** | Semana 3 | Muestreo de regiones, métricas, estabilización | Modo Partículas, paletas, panel final, atajos, PNG, errores | Dos efectos operativos; controles y exportación correctos |
| **F4 — Pruebas y presentación** | Semana 4 | Benchmark reproducible, casos de iluminación, sensibilidad, FPS, documentación | Ajustes visuales, diagnóstico, experiencia de exposición, galería básica si alcanza | Aplicación demostrable, pruebas, capturas, reporte y video de evidencia |
| **Extensión A** | Después del MVP / si existe holgura | Running Average, comparativa de sensibilidad | Selector de detector, reinicio de fondo | Comparación cualitativa documentada con mismo video |
| **Extensión B** | Opcional | Lucas–Kanade disperso | Partículas direccionales / vector field | Vectores contrastados en escenas sintéticas; sin fingir precisión |
| **Extensión C** | Opcional | Ajustes por resolución / estabilidad | Silueta, Experimental, exportación de video | Nuevos modos sin regresión de los dos obligatorios |

### 12.1 Dependencias entre fases

```text
Cámara -> Preprocesamiento -> Frame Differencing -> Limpieza -> Regiones
                                                    |            |
                                                    v            v
                                               Rastros       Partículas
                                                    \           /
                                                     -> UI / PNG

Running Average ----------> alternativa de detección  (NO bloqueante)
Lucas–Kanade  ------------> vectores direccionales   (NO bloqueante)
```

### 12.2 Definición de terminado (DoD) para el proyecto final

- [ ] Webcam real o fuente sintética **claramente identificada** disponible.
- [ ] Diferencia entre fotogramas calculada manualmente sin operaciones prefabricadas de OpenCV.
- [ ] Umbral, filtrado y al menos una operación morfológica manual validados con pruebas.
- [ ] Máscara binaria limpia y regiones coherentes; ausencia de errores de `uint8`.
- [ ] Rastros de luz funcionales con desvanecimiento controlado.
- [ ] Partículas funcionales con límites de cantidad y vida.
- [ ] Interfaz consistente y comprensible, inicio/pausa, modo, sensibilidad, intensidad, paleta y limpiar.
- [ ] Visualización de depuración original → diferencia → máscara → arte.
- [ ] Captura PNG válida y mensajes de éxito/error.
- [ ] Pruebas sintéticas automatizadas y registros de pruebas con webcam real.
- [ ] FPS y tiempos medidos sobre hardware real y descritos sin exagerar.
- [ ] Reporte metodológico, documentación de algoritmos, fuentes y demostración.

La inclusión de Running Average, Lucas–Kanade, Silueta y Experimental **no cambia** esta definición mínima salvo actualización explícita del alcance académico.

---

## 13. Plan de evaluación y metodología

### 13.1 Escenarios de prueba

1. **Sin movimiento:** sujeto/cámara quietos; observar ruido y falsos positivos.
2. **Movimiento lento:** mano que recorre una trayectoria; comprobar continuidad de rastros.
3. **Movimiento rápido:** desplazamiento brusco; registrar fragmentación/pérdidas.
4. **Iluminación estable:** prueba de referencia.
5. **Iluminación gradual:** observación de adaptación de Running Average.
6. **Cambio de luz repentino:** cuantificación de cambios espurios.
7. **Fondo con textura/movimiento:** ventilador, monitor o sombras; analizar sensibilidad.
8. **Diferentes umbrales:** comparar sensibilidad y ruido con la misma secuencia.
9. **Sin cámara/corte temporal:** verificar recuperación y mensajes de la interfaz.
10. **Repetición con mismo clip sintético:** reproducción determinista para comparar detectores y parámetros.

### 13.2 Métricas y evidencias

| Medida | Definición | Cómo se registra |
|---|---|---|
| FPS de render | Presentaciones de pantalla / segundo | Temporizador real |
| FPS de captura | Frames leídos correctamente / segundo | Timestamps de adquisición |
| Tiempo de procesamiento | Milisegundos por frame dentro del Módulo 1 | Media, mediana y percentil alto cuando haya suficientes muestras |
| Tiempo de render | Milisegundos por frame del Módulo 2 | Por modo artístico |
| Porcentaje de máscara | Píxeles activos / total | Trama temporal por escenario |
| Número de regiones | Componentes aceptados tras el filtro | Registrar por frame |
| Falsos positivos visuales | Actividad bajo condición estática | Descripción y, si se etiqueta manualmente, medida reproducible |
| Continuidad de rastros | Observación de interrupciones y parpadeo | Capturas/clip de escenario repetido |

**Sin verdad terreno etiquetada no se afirma precision, recall, F1 ni exactitud de detección.** Si se crean máscaras de referencia manuales para un pequeño conjunto, entonces sí es posible calcular `IoU = |P∩G|/|P∪G|`, indicando tamaño y condiciones del dataset. La comparación A/B utiliza **el mismo video**, misma resolución y condiciones equivalentes.

### 13.3 Entregables académicos

- Software funcional y código fuente.
- Documento de metodología con fórmulas, decisiones e implementaciones propias.
- Pruebas automatizadas con matrices pequeñas y escenarios sintéticos.
- Evidencia de webcam original, diferencia, máscara previa, máscara limpia y salida artística.
- Tabla de tiempos/FPS **medidos**, condiciones de hardware, resolución y configuración.
- Video corto o demo en vivo que muestre rastros y partículas.
- Referencias bibliográficas, créditos de tipografías y repos consultados.

---

## 14. Registro de riesgos y decisiones técnicas

| Riesgo | Consecuencia | Mitigación en el diseño |
|---|---|---|
| Luz variable o cámara con autoexposición | Falsos positivos | Umbral ajustable, filtro, comparación A/B, aviso de limitaciones |
| Fondo de alta actividad | Detección dispersa | Área mínima, muestreo y validación por escenarios |
| Persona quieta | Desaparición de máscara | Explicación honesta; alternativa B no resuelve totalmente el caso |
| Procesamiento Python lento | Interacción entrecortada | 320×240, NumPy vectorizado, número de partículas acotado y profiling |
| Sobreprocesamiento morfológico | Pérdida de detalle | Kernels 3×3, pruebas de dedos/brazos y vistas cruda/limpia |
| Desalineación cámara/lienzo | Rastros en coordenadas incorrectas | Transformación centralizada, letterbox y test de esquinas |
| Reinicio accidental de fondo | Parpadeo/artefactos | Estado de calentamiento, reinicio explícito y limpieza temporal |
| Pygame UI demasiado compleja | Retraso académico | MVP con widgets esenciales antes de galería/presets avanzados |
| Dependencia de Lucas–Kanade | Retraso en entrega | Extensión posterior y aislada mediante la misma interfaz de detector |
| Código externo sin licencia verificada | Riesgo de reutilización indebida | Solo referencia conceptual; implementación propia y trazabilidad |
| Privacidad de webcam | Registro involuntario de personas | Procesamiento local y exportación explícita, sin guardado automático |

---

## 15. Investigación de repositorios y ejemplos existentes

**Fecha de investigación:** 04-11-2026. Se revisaron descripciones y documentación pública de repositorios reales. Las referencias son **fuentes para estudio, comparación y atribución**, no requisitos de incorporar código externo. Su existencia y descripción se verificaron; **no se han ejecutado ni auditado sus tests, seguridad o exactitud**. Los vínculos y licencias pueden variar; antes de integrar código se revisan licencia exacta, autoría y compatibilidad.

### 15.1 Referencias de implementación por tema

| ID | Referencia real | Qué aporta a Motion Art Lab | Uso recomendado | Estado de licencia verificado |
|---|---|---|---|---|
| **R1** | [PavlosMak/MotionDetection](https://github.com/PavlosMak/MotionDetection) | Ejemplo básico de diferencia de frames y umbral sobre cambio | Contrastar planteamiento, no copiar llamadas de OpenCV | No confirmada en esta revisión |
| **R2** | [god233012yamil/Python-OpenCV-Examples — frame differencing](https://github.com/god233012yamil/Python-OpenCV-Examples/blob/main/motion_detection_with_frame_differencing.py) | Pipeline de diferencia → gris → blur → umbral → dilatación → regiones | **Referencia de secuencia**, no de implementación manual: usa funciones CV prefabricadas | No confirmada |
| **R3** | [IamMosiow/computer-vision-ml-internship-2023](https://github.com/IamMosiow/computer-vision-ml-internship-2023) (`background_estimation/`) | Ejemplos de estimación de fondo con promedio móvil/mediana y detección básica | Comparar planteamiento de fondo; verificar detalles del código en el subdirectorio | No confirmada |
| **R4** | [tnybny/Exploring-optical-flow](https://github.com/tnybny/Exploring-optical-flow) | Notebook didáctico de gradientes y Lucas–Kanade con mínimos cuadrados | Referencia conceptual principal para opción C | No confirmada |
| **R5** | [ZORAKEN/cv-algorithms-python](https://github.com/ZORAKEN/cv-algorithms-python) | Morfología binaria, filtros y convolución desde fundamentos | Estudio de operadores; implementación final propia | No confirmada |
| **R6** | [OmarElprolosy66/CCL_Connected-Component-Labeling](https://github.com/OmarElprolosy66/CCL_Connected-Component-Labeling) | Dos pasadas + Union-Find, conectividad 4/8 y tests | Referencia para componentes conectados | **MIT** en repositorio |
| **R7** | [seung-lab/connected-components-3d](https://github.com/seung-lab/connected-components-3d) | Explicación y variantes de etiquetado de componentes | Referencia teórica de calidad; **no** incluir biblioteca en MVP | Comprobar antes de reutilizar código |
| **R8** | [Zhangjy1101/ParticleUniverse](https://github.com/Zhangjy1101/ParticleUniverse) | Interacción creativa y partículas con Pygame | Inspiración artística, interacción y presets | No confirmada |
| **R9** | [mahoorifatemeh-byte/Pygame-Fireworks](https://github.com/mahoorifatemeh-byte/Pygame-Fireworks) | Desvanecimiento de partículas, rastros cortos y persistencia | Estudio de render/partículas, adaptación conceptual | **MIT** en repositorio |
| **R10** | [rOsHaNiC-26/Generative-Art-Engine-2.0 — `pygame_dashboard.py`](https://github.com/rOsHaNiC-26/Generative-Art-Engine-2.0/blob/main/python_prototypes/pygame_dashboard.py) | Prototipo de interfaz Pygame con zona de lienzo y panel de controles | Inspiración para layout, **no** estética a copiar | No confirmada |
| **R11** | [bwsw/rt-motion-detection-opencv-python](https://github.com/bwsw/rt-motion-detection-opencv-python) | Ejemplo de detección robusta, escalado y separación de parámetros de rendimiento | Referencia de optimización, **sin incorporar Numba** al MVP; repositorio archivado | **Apache-2.0**; archivado desde 2023 |
| **R12** | [pygame-community/pygame-ce](https://github.com/pygame-community/pygame-ce) | Referencia oficial de eventos, superficies, texto y `surfarray` | Dependencia de interfaz, no algoritmo de visión | **LGPL-2.1-or-later** documentada |
| **R13** | [rsms/inter](https://github.com/rsms/inter) y [CatharsisFonts/Cormorant](https://github.com/CatharsisFonts/Cormorant) | Tipografías libres para identidad editorial + interfaz | Recursos visuales | **OFL-1.1** en ambos repositorios |

**Precaución sobre las referencias:** los repositorios son material de consulta, no una auditoría de corrección ni de calidad. Para R3 se verificó la existencia del repositorio y del directorio `background_estimation/`, pero no se auditó toda la implementación del directorio. Running Average tiene suficiente especificación matemática en este documento para implementarse sin depender de código externo.

### 15.2 Fuente académica primaria y documentación

- **P1 — Lucas–Kanade (1981):** B. D. Lucas and T. Kanade, “An Iterative Image Registration Technique with an Application to Stereo Vision,” *Proc. 7th International Joint Conference on Artificial Intelligence*, pp. 674–678, 1981. [Publicación IJCAI](https://ijcai.org/proceedings/1981/127). Fundamenta el método; no es una biblioteca.
- **D1 — OpenCV, video:** [Getting Started with Videos](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html). Confirma la API de `VideoCapture`/`VideoWriter`, comprobación de frames y liberación de cámara.
- **D2 — NumPy:** [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) y [NumPy reference](https://numpy.org/doc/stable/reference/). Base para vectorización y tratamiento de matrices.
- **D3 — pygame-ce:** [Documentación](https://pyga.me/docs/) y [surfarray](https://pyga.me/docs/ref/surfarray.html). Diseño del loop y conversión a superficies.
- **D4 — pytest:** [Guía de fixtures](https://docs.pytest.org/en/stable/explanation/fixtures.html). Construcción de pruebas independientes de la webcam.

### 15.3 Política de uso de referencias

- **Reimplementación académica:** fórmulas, pseudocódigo, ideas matemáticas y ejemplos mínimos como material de estudio; solución y pruebas elaboradas para Motion Art Lab.
- **Código copiado o adaptado:** solo bajo licencia explícitamente identificada, respetando atribuciones, avisos y condiciones; registro del origen por archivo y línea o commit cuando corresponda.
- **Repositorios sin licencia visible o verificada:** únicamente inspiración conceptual, no incorporación literal de archivos/funciones.
- **Documentación oficial:** enlazada en el reporte y citada donde corresponda.
- **Imágenes y fuentes:** recursos con licencia comprobada o creados específicamente; ninguna descarga de fotos personales o fondos de terceros por defecto.

### 15.4 Conclusión de la investigación: ¿qué requiere código externo?

**No es necesario importar implementaciones de otros repositorios para terminar el proyecto.** Frame Differencing, Running Average, umbral, filtro promedio, erosión, dilatación, apertura, cierre, suavizado temporal, flood-fill y componentes conectados son implementables a partir de las fórmulas y contratos aquí definidos, con NumPy y pruebas sintéticas.

Las referencias externas añaden valor para **contrastar buenas prácticas y explicar el fundamento**, especialmente Union-Find, Lucas–Kanade, el fade de superficies y la organización de una interfaz creativa. La implementación más compleja y con mayor riesgo de tiempo es Lucas–Kanade; permanece desacoplada y opcional.

---

## 16. Criterios finales de coherencia del proyecto

1. **No hay arte sin señal real:** salvo modo demo explícitamente marcado, los efectos nacen de máscara/regiones/flujo calculados; un fondo animado decorativo nunca se presenta como detección.
2. **El procesamiento es explicable:** cada paso ofrece fórmulas, parámetros y resultados intermedios accesibles desde diagnóstico.
3. **La interfaz se entiende sin manual:** modos reconocibles, cambios instantáneos, errores claros y captura sencilla.
4. **El código es verificable:** las operaciones de visión clave tienen pruebas sintéticas reproducibles y no dependen de algoritmos de alto nivel prefabricados.
5. **La performance es medida:** FPS y tiempos muestran datos reales del dispositivo; presets no prometen métricas universales.
6. **El MVP tiene límites explícitos:** dos efectos, un detector manual y limpieza validada bastan para la entrega; las extensiones no comprometen su estabilidad.
7. **La intención visual permanece:** la aplicación tiene identidad turquesa/coral/editorial, pero conserva contraste y legibilidad durante la exposición.

**Definición de producto en una frase:** *Motion Art Lab convierte el cambio entre fotogramas de una webcam en un lienzo de luz y partículas, mediante algoritmos de procesamiento propios presentados en una interfaz artística, clara y comprobable.*
