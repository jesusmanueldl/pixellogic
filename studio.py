"""PixelLogic E35 vs QR: GUI + CLI. The E35 decoding core is preserved."""
import argparse, csv, sys, time
from pathlib import Path
import cv2, numpy as np, qrcode
from qrcode.constants import ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H
from PIL import Image,ImageTk,ImageDraw,ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from pixellogic import generate as pixel_generate, read as pixel_read, load_image

LEVELS={'M':ERROR_CORRECT_M,'Q':ERROR_CORRECT_Q,'H':ERROR_CORRECT_H}

def create_pair(message,folder,pixels=900,level='H',background='white'):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    p=folder/'pixellogic.png';q=folder/f'qr_{level}.png'
    n=pixel_generate(message.encode('utf-8'),p,pixels)
    qr=qrcode.QRCode(version=None,error_correction=LEVELS[level],box_size=10,border=4)
    qr.add_data(message.encode('utf-8'),optimize=0);qr.make(fit=True)
    qr_img=qr.make_image(fill_color='black',back_color='white').convert('RGB').resize((pixels,pixels),Image.Resampling.NEAREST)
    qr_img.save(q)
    left=Image.open(p).convert('RGB');right=qr_img
    if background!='white':
        # common visual backing is OUTSIDE the white symbol area, preserving quiet zone
        pass
    w=pixels;pad=max(28,w//22);title=max(36,w//15)
    canvas=Image.new('RGB',(w*2+pad*3,w+pad*2+title),'#e9f0f7' if background=='gray' else 'white')
    dr=ImageDraw.Draw(canvas)
    canvas.paste(left,(pad,pad+title));canvas.paste(right,(w+pad*2,pad+title))
    dr.text((pad,pad+title//4),'PixelLogic E35',fill='#101820');dr.text((w+pad*2,pad+title//4),f'QR nivel {level}',fill='#101820')
    comp=folder/'comparacion.png';canvas.save(comp)
    return {'pixel':p,'qr':q,'comparacion':comp,'pixel_modules':n,'qr_modules':len(qr.get_matrix()),'bytes':len(message.encode('utf-8'))}

def read_both(path):
    path=Path(path)
    t=time.perf_counter();pmsg,pinfo,psecs=pixel_read(path)
    img=load_image(path)
    start=time.perf_counter()
    if img is None: raise ValueError('Imagen inválida')
    qrtext,_,_=cv2.QRCodeDetector().detectAndDecode(img)
    qrsecs=time.perf_counter()-start
    return [{'sistema':'PixelLogic E35','ok':pmsg is not None,'texto':pmsg.decode('utf-8','replace') if pmsg is not None else '', 'segundos':psecs,'detalle':str(pinfo.get('stage',''))},
            {'sistema':'QR OpenCV','ok':bool(qrtext),'texto':qrtext,'segundos':qrsecs,'detalle':''}]

def benchmark(folder,out):
    paths=sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))
    rows=[]
    for p in paths:
        for r in read_both(p):rows.append({'archivo':p.name,**r})
    with open(out,'w',encoding='utf-8-sig',newline='') as f:
        ww=csv.DictWriter(f,fieldnames=['archivo','sistema','ok','texto','segundos','detalle']);ww.writeheader();ww.writerows(rows)
    return len(paths)

def gui():
    import tkinter as tk
    from tkinter import ttk,filedialog,messagebox
    root=tk.Tk();root.title('PixelLogic E35 + QR | Laboratorio visual');root.geometry('1050x800');root.configure(bg='#eef3f8')
    msg=tk.StringVar(value='Hola desde Tabasco');level=tk.StringVar(value='H');pixels=tk.IntVar(value=700);filevar=tk.StringVar(value='');status=tk.StringVar(value='Genera ambos códigos con el mismo mensaje.');imgref=[]
    outer=ttk.Frame(root,padding=16);outer.pack(fill='both',expand=True)
    ttk.Label(outer,text='PixelLogic E35 ↔ QR — Generador y lector',font=('Segoe UI',17,'bold')).pack(anchor='w',pady=(0,8))
    ttk.Label(outer,text='Mensaje UTF-8').pack(anchor='w');ttk.Entry(outer,textvariable=msg,font=('Segoe UI',12)).pack(fill='x',pady=(0,8))
    opts=ttk.Frame(outer);opts.pack(fill='x')
    ttk.Label(opts,text='Nivel QR').pack(side='left');ttk.Combobox(opts,textvariable=level,values=['M','Q','H'],width=5,state='readonly').pack(side='left',padx=8)
    ttk.Label(opts,text='Resolución').pack(side='left');ttk.Combobox(opts,textvariable=pixels,values=[500,700,900,1200],width=6,state='readonly').pack(side='left',padx=8)
    display=ttk.Label(outer,text='La comparación aparecerá aquí',anchor='center');display.pack(fill='both',expand=True,pady=12)
    log=tk.Text(outer,height=8,font=('Consolas',10));log.pack(fill='x');log.configure(state='disabled')
    def write(s):
        log.configure(state='normal');log.insert('end',s+'\n');log.see('end');log.configure(state='disabled')
    def show(path):
        im=Image.open(path).convert('RGB');im.thumbnail((970,440));tkim=ImageTk.PhotoImage(im);imgref[:]=[tkim];display.configure(image=tkim,text='')
    def make():
        folder=filedialog.askdirectory(title='Elegir carpeta de salida')
        if not folder:return
        try:
            meta=create_pair(msg.get(),folder,pixels.get(),level.get());show(meta['comparacion'])
            write(f"GENERADOS: {meta['pixel_modules']} módulos PixelLogic, {meta['qr_modules']} módulos QR (incluye margen), {meta['bytes']} bytes UTF-8")
            for name in ('pixel','qr'):
                result=read_both(meta[name]); expected='PixelLogic E35' if name=='pixel' else 'QR OpenCV'
                r=next(x for x in result if x['sistema']==expected)
                write(f"AUTOPRUEBA {expected}: {'OK' if r['ok'] and r['texto']==msg.get() else 'FALLO'} | {r['segundos']:.4f}s")
            status.set('Guardados en '+folder)
        except Exception as exc:messagebox.showerror('Error al generar',str(exc))
    def select():
        p=filedialog.askopenfilename(filetypes=[('Imágenes','*.png *.jpg *.jpeg *.bmp *.tif *.tiff')]);
        if p:filevar.set(p);show(p)
    def read():
        if not filevar.get():return messagebox.showinfo('Archivo','Selecciona una imagen')
        try:
            results=read_both(filevar.get());write('ARCHIVO: '+filevar.get())
            for r in results:write(f"{r['sistema']}: {'RECUPERADO' if r['ok'] else 'NO RECUPERADO'} | {r['segundos']:.4f}s | {r['texto']!r}")
            status.set('Lectura terminada')
        except Exception as exc:messagebox.showerror('Error al leer',str(exc))
    btn=ttk.Frame(outer);btn.pack(fill='x',pady=5)
    ttk.Button(btn,text='1. Generar PixelLogic + QR',command=make).pack(side='left',padx=4)
    ttk.Button(btn,text='2. Abrir imagen / foto',command=select).pack(side='left',padx=4)
    ttk.Button(btn,text='3. Leer con ambos',command=read).pack(side='left',padx=4)
    ttk.Label(outer,textvariable=status).pack(anchor='w')
    root.mainloop()

def main():
    ap=argparse.ArgumentParser(description='PixelLogic E35 + QR. Sin argumentos abre la interfaz visual.')
    sub=ap.add_subparsers(dest='cmd')
    g=sub.add_parser('generar');g.add_argument('--texto',required=True);g.add_argument('--salida',required=True);g.add_argument('--pixels',type=int,default=900);g.add_argument('--nivel',choices=['M','Q','H'],default='H')
    r=sub.add_parser('leer');r.add_argument('--imagen',required=True)
    b=sub.add_parser('lote');b.add_argument('--carpeta',required=True);b.add_argument('--salida',default='lecturas.csv')
    args=ap.parse_args()
    if args.cmd=='generar':
        v=create_pair(args.texto,args.salida,args.pixels,args.nivel);print('\n'.join(f'{k}: {x}' for k,x in v.items()))
    elif args.cmd=='leer':
        for x in read_both(args.imagen):print(f"{x['sistema']}: {'RECUPERADO' if x['ok'] else 'NO RECUPERADO'} ({x['segundos']:.4f}s): {x['texto']}")
    elif args.cmd=='lote':print(f"{benchmark(args.carpeta,args.salida)} imágenes, CSV {args.salida}")
    else:gui()
if __name__=='__main__':main()
