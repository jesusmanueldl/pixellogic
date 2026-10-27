"""PixelLogic E34 user-facing CLI around the unmodified E32 experimental reader.
Research prototype. Camera/photo performance not yet independently validated.
"""
import argparse
import csv
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import e15
import e31
import e32
import e35


def generate(message: bytes, output: Path, pixels: int = 900):
    if pixels < 240 or pixels > 6000:
        raise ValueError('pixels debe estar entre 240 y 6000')
    symbol = e31.build_symbol(message)
    image = e15.render(symbol, pixels)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise OSError(f'No se pudo guardar {output}')
    return symbol.shape[0]


def load_image(path: Path):
    if not path.is_file():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if len(data) == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def read(path: Path):
    img = load_image(path)
    if img is None:
        return None, {'stage': 'invalid_image'}, 0.0
    begin = time.perf_counter()
    message, info = e35.decode_image(img, return_info=True)
    elapsed = time.perf_counter() - begin
    return message, info, elapsed


def main(argv=None):
    parser = argparse.ArgumentParser(description='PixelLogic E35: generador E34 con rescate proyectivo experimental.')
    sub = parser.add_subparsers(dest='command', required=True)
    g = sub.add_parser('generar', help='Crear símbolo PNG para imprimir o fotografiar')
    g.add_argument('--texto', required=True, help='Texto UTF-8 de 1 a 180 bytes, sujeto a capacidad del símbolo')
    g.add_argument('--salida', required=True, type=Path)
    g.add_argument('--pixels', type=int, default=900)
    r = sub.add_parser('leer', help='Leer PNG/JPG capturado sin suministrar geometría')
    r.add_argument('--imagen', required=True, type=Path)
    b = sub.add_parser('lote', help='Leer todas las imágenes de una carpeta y crear un CSV')
    b.add_argument('--carpeta', required=True, type=Path)
    b.add_argument('--salida', default=Path('resultados_fotos.csv'), type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'generar':
            data = args.texto.encode('utf-8')
            side = generate(data, args.salida, args.pixels)
            print(f'OK | {len(data)} bytes UTF-8 | {side}x{side} módulos | {args.salida}')
            return 0
        if args.command == 'leer':
            message, info, elapsed = read(args.imagen)
            if message is None:
                print(f'NO RECUPERADO | etapa: {info.get("stage")} | {elapsed:.3f}s')
                return 2
            print(f'RECUPERADO | {len(message)} bytes | {elapsed:.3f}s | {info}')
            print('UTF-8:', message.decode('utf-8', errors='replace'))
            print('HEX:', message.hex())
            return 0
        paths = sorted(p for p in args.carpeta.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff') and p.is_file())
        if not paths:
            raise ValueError('No se encontraron imágenes compatibles')
        rows = []
        for p in paths:
            message, info, secs = read(p)
            rows.append({'filename':p.name,'recovered':int(message is not None),
                         'utf8':message.decode('utf-8',errors='replace') if message is not None else '',
                         'hex':message.hex() if message is not None else '',
                         'stage':info.get('stage',''), 'seconds':round(secs,6)})
            print(f'{p.name}: {"RECUPERADO" if message is not None else "FALLÓ"} ({secs:.3f}s)',flush=True)
        args.salida.parent.mkdir(parents=True,exist_ok=True)
        with args.salida.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
        print('CSV:',args.salida)
        return 0
    except (ValueError, OSError, FileNotFoundError, cv2.error) as exc:
        print('ERROR:',exc,file=sys.stderr)
        return 1

if __name__=='__main__':
    sys.exit(main())
