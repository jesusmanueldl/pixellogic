# Publicar PixelLogic en GitHub

## Antes de subir

1. Ejecuta `python -m unittest discover -s tests -v`.
2. Revisa autoría y licencias, y elimina credenciales, archivos personales o información sensible.
3. Inspecciona `git status` y los archivos a subir; `.gitignore` excluye entornos virtuales y resultados locales.

## Crear repositorio en GitHub

1. Inicia sesión en https://github.com/.
2. Pulsa **+ → New repository** o visita https://github.com/new.
3. Nombre: **pixellogic**. Descripción: **Experimental visual code generator and reader (E35)**.
4. Visibilidad **Public**.
5. No selecciones README, `.gitignore` ni licencia porque ya están en esta carpeta.
6. Pulsa **Create repository** y copia la URL HTTPS de tu repositorio.

## Desde PowerShell

```powershell
cd RUTA\PixelLogic_GitHub
git init
git add .
git commit -m "Initial public release: PixelLogic E35"
git branch -M main
git remote add origin https://github.com/TU-USUARIO/pixellogic.git
git push -u origin main
```

Si Git solicita autenticación, utiliza Git Credential Manager, inicio de sesión por navegador, GitHub CLI (`gh auth login`) o un token de acceso; **no** publiques contraseñas ni tokens en los archivos. GitHub ya no permite utilizar la contraseña de la cuenta como método de autenticación Git por HTTPS.

## Actualizaciones

```powershell
git add .
git commit -m "Update documentation and tests"
git push
```

## GitHub Releases (opcional)

Ve a **Releases → Draft a new release**, etiqueta `v0.1.0-experimental`, adjunta el ZIP y describe los límites.

## Antes de publicitarlo

Verifica que el README muestre el código PixelLogic real, que la instalación funcione en un equipo limpio y que el enlace a la versión publicada sea correcto. No afirmes superioridad sobre QR.
