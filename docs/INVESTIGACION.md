# Investigación y alcance de las comparaciones

## Qué se consiguió

- Generador y lector E35 que recuperan símbolos digitales; algunas pruebas previas con fotografías reales funcionaron.
- Cuatro localizadores distribuidos y recuperación óptica en determinadas perspectivas.
- Comparador visual frente a QR mediante el mismo mensaje y tamaño de lienzo.

## Comparación histórica, no benchmark de CI

| Datos | PixelLogic E35 | QR-H |
|---|---:|---:|
| 4 bytes, oclusión sintética | 30/30 | 18/30 |
| 18 bytes, oclusión sintética | 30/30 | 18/30 |
| 102 bytes, oclusión sintética | 11/30 | 28/30 |

Resultados de una batería anterior en condiciones específicas. El tamaño idéntico de imagen no iguala cantidad de módulos, redundancia ni capacidad. No interpretar como ventaja general. Los archivos de estos ensayos no se presentan como ejecutados nuevamente por este repositorio.

## Hipótesis descartada

Se ensayaron mapeos basados en reglas celulares reversibles que mezclan bits entre bytes. En el esquema corrector por bytes de PixelLogic, bajo ciertas oclusiones pueden aumentar el número de bytes afectados; por ello se preserva E35 sin esa integración.

## Qué sería necesario para una publicación científica

Benchmark independiente, conjunto abierto de fotografías de dispositivos y papeles distintos, control de tamaño físico y redundancia, lectores QR alternativos, análisis de incertidumbre y una contribución no cubierta por el estado del arte.
