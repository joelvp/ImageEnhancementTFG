# Image Enhancement TFG

Este proyecto proporciona herramientas y modelos para la mejora de imágenes. A continuación se detallan los pasos necesarios para clonar el repositorio, configurar el entorno y ejecutar el proyecto.

## Instalación

Primero, clona el repositorio a tu máquina local utilizando Git:

```bash
git clone https://github.com/joelvp/ImageEnhancementTFG.git
cd ImageEnhancementTFG
```

### Instalar CUDA 12.1

CUDA es necesario para aprovechar la potencia de la GPU en ciertos modelos de este proyecto. Para instalar CUDA 12.1, sigue estos pasos:

1. Descarga el instalador de CUDA 12.1 desde el [sitio web de Nvidia](https://developer.nvidia.com/cuda-12-1-0-download-archive).
2. Sigue las instrucciones de instalación proporcionadas por Nvidia para tu sistema operativo.
3. Asegúrate de que los controladores de tu GPU están actualizados y son compatibles con CUDA 12.1.

### Creación de Entorno Conda

Crea un nuevo entorno de Conda para gestionar las dependencias del proyecto:

```bash
conda create --name ImageEnhancementTFG python=3.10
conda activate ImageEnhancementTFG
```

### Instalación de Poetry y Librerias
Usaremos Poetry como gestor de dependencias:

```bash
pip install poetry
poetry install
```

### Configuración Adicional
Algunas configuraciones adicionales son necesarias para ciertos modelos. Navega al directorio NAFNet y ejecuta el siguiente comando:

```bash
cd imageenhancementtfg/models/NAFNet
python setup.py develop --no_cuda_ext
```

## Ejecutar Proyecto
Vuelve a la carpeta imageenhancementtfg y ejecuta el script principal:

```bash
cd ../..
python gui.py
```
