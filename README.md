# Setup del proyecto
Para poder implmentar el IDS diseñado con Federated Learning es necesarrio seguir una serie de pasos:

### Descarga del repositorio
Lo primero de todo será descargar todas las carpetas y documentos que se encuentran en este repositorio. 
En la carpeta de Datasets se encuentran los datasets que han sido previamente tratados, ya divididos en conjuntos de entrenamiento y prueba.

### Instalación de dependencias
Para poder desplegar el entorno será necesario instalar algunas dependencias como Flower, Torch, Torchvision y Tqdm

```shell
pip install flwr~=1.4.0
pip install torch~=1.0.1
pip install torchvision~=0.15.2
pip install tqdm~=4.65.0
```

### Ejecución del código
Una vez instaladas las dependencias, ya está listo para poder poner en marcha el entorno.

Se arranca el servidor con el siguiente comando:

```shell
python server.py
```

Una vez iniciado el servidor, se pueden iniciar los clientes en otra ventana con el siguiente comando:

Si se quiere iniciar el cliente con el conjunto de datos KDDCUP99:

```shell
python clientkdd.py
```

Si se quiere iniciar el cliente con el conjunto de datos UNSW-NB15:

```shell
python clientunsw.py
```

El entorno está preparado para que se ejecuten dos clientes que utilicen el mismo conjunto de datos.