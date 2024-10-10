# project
Modelo de predicción modularizado y dockerizado:
Este proyecto consiste en un modelo de predicción de aprendizaje automático que se ha modularizado y dockerizado. El objetivo del modelo es predecir resultados en función de un conjunto de datos determinado y se puede ejecutar en un contenedor Docker para garantizar la coherencia y la portabilidad en diferentes entornos.


## Dependencias
- Python 3.10
- LightGBM
- Pandas
- Scikit-learn
- Docker

Todas las demás dependencias están incluidas en el archivo `requirements.txt` y se instalarán durante el proceso de compilación de Docker.


## Estructura de carpetas
/project
├── app/
│   ├── config/                # Archivos de configuracion
│   ├── helpers/               # Scripts de ayuda para carga de datos, preprocesamiento, etc.
│   ├── logs/                  # Archivos de registro generados durante la ejecución
│   ├── model/                 # Archivos de modelo y escalador entrenados
│   └── main.py                # Script principal para ejecutar la predicción
├── data/                      # Carpeta para datos de entrada y predicción
│   └── predictions/           # Directorio donde se guardan las predicciones
├── Dockerfile                 # Archivo de configuración de Docker
├── requirements.txt           # Dependencias de Python
└── README.md                  # Documentación del proyecto


## Cómo funciona el modelo
El modelo toma los datos de entrada, los procesa previamente (manejando los datos faltantes, estandarizando los campos numéricos, etc.) y luego genera predicciones utilizando un modelo LightGBM entrenado previamente. Los resultados se guardan como un archivo CSV en la carpeta especificada.

## Ejecución del contenedor y generación de predicciones
Para ejecutar el modelo dentro de un contenedor Docker y generar las predicciones, siga los pasos a continuación:

1. Clonar el repositorio:
git clone <repository_url>
cd <repository_directory>

2. Construir la imagen de Docker:
docker build -t nombre_de_la_imagen .

3. Ejecutar el contenedor y mapear la carpeta donde se guardarán las predicciones. Por ejemplo, si deseas guardar las predicciones en /home/usuario/predicciones/, debes ejecutar:
docker run -v /home/usuario/predicciones:/project/data/predictions nombre_de_la_imagen

4. Verificar las predicciones: Después de que el contenedor termine la ejecución, las predicciones estarán guardadas en el archivo predictions.csv dentro de la carpeta que hayas especificado (en este caso /home/usuario/predicciones/).
