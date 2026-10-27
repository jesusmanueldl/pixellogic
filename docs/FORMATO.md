# Formato visual experimental PixelLogic E35

## Propósito

Representar una carga binaria mediante una matriz visual propia, con localizadores en las cuatro esquinas y redundancia para recuperación en daños limitados. No es QR ni implementa ISO/IEC 18004.

## Construcción actual

1. Codificar el texto como bytes UTF-8.
2. Empaquetar metadatos de tamaño, longitud e integridad y aplicar redundancia; el núcleo incluye CRC y mecanismos algebraicos.
3. Distribuir datos en la matriz interior, escoger tamaño compatible y añadir cuatro marcadores cuadrados de esquina.
4. Renderizar módulos blancos/negros en PNG con zona blanca alrededor.
5. En la lectura, localizar el símbolo, corregir geometría/binarizar, recuperar metadatos, corregir errores dentro de su capacidad y validar integridad.

El código real, no esta descripción, es la referencia de implementación: `core/e31.py`, `core/e22.py`, `core/e23.py`, `core/e32.py`, `core/e35.py`. La versión experimental no pretende ser una especificación definitiva para implementaciones interoperables de terceros.

## ¿Es cifrado?

No. CRC/corrección de errores son mecanismos de integridad y recuperación, **no** confidencialidad ni autenticación. No insertes claves o secretos.

## Limitaciones

- Lectura robusta en todos los móviles no garantizada.
- Pueden fallar inclinación extrema, deformación no plana, manchas, oclusiones y mala impresión.
- Capacidad dependiente de los tamaños soportados por el núcleo.
- Solo el lector PixelLogic comprende su formato.
- Se requiere auditar el protocolo antes de usarlo en producción o intercambiar símbolos entre implementaciones independientes.
