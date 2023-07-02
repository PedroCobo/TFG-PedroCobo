# Setup del proyecto
Para poder implmentar el IDS diseñado con Federated Learning es necesario seguir una serie de pasos:

### Descarga del repositorio
Lo primero de todo será descargar todas las carpetas y documentos que se encuentran en este repositorio. 

El KDDCUP99 y el UNSW-NB15  se encuentran cada uno en una carpeta zip, ya que el peso de la carpeta era elevado y no podía subirlo de ninguna otra manera. Por ello, una vez descargadas las carpetas es necesario descomprimirlas para poder utilizarse.

El CICIDS2017 se encuentra en el link: https://mega.nz/folder/rr4QXaYL#y_Qv5YvOZ072N0qz5Nd7gg, ya que pesa mucho más que los anteriores y no me permitía subirlo a GitHub directamente. Una vez se acceda al link, hay que descargar la carpeta completa y descomprimir.

Para la ejecución de las consolas, hay que tener el proyecto en la misma carpeta, teniendo cada dataset en su respectiva carpeta: DatasetKDD, DatasetUNSW y DatasetCIC.

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

Una vez iniciado el servidor, se pueden iniciar los clientes cada uno en otra ventana con el siguiente comando:

Si se quiere iniciar el cliente con el conjunto de datos KDDCUP99:

```shell
python clientkdd.py
```

Si se quiere iniciar el cliente con el conjunto de datos UNSW-NB15:

```shell
python clientunsw.py
```

Si se quiere iniciar el cliente con el conjunto de datos CICIDS2017:

```shell
python clientCIC.py
```

El entorno está preparado para que se ejecuten dos clientes que utilicen el mismo conjunto de datos.
