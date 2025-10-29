import pandas as pd
from deep_translator import GoogleTranslator

class DataPreparer: 

    """
    Clase para manejar la carga y el procesamiento de los encabezados de nuestro dataset
    patiendo del path donde se almacenan nuestros datos
    """
    
    def __init__(self, bd_connection_string, delimiter = ','):
        self.filepath = bd_connection_string
        self.delimiter = delimiter
        self.data = []

    def read_file(self):

        try:
            data = pd.read_csv(self.filepath, sep = self.delimiter)
            self.data = data
            return self.data
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo en la ruta: {self.path}")
            return None
        except Exception as e:
            print(f"Ocurrió un error al cargar los datos: {e}")
            return None
        
    def translate_header(self) -> pd.DataFrame or None:

        """
        Traduce los encabezados del dataset (de alemán a inglés) 
        y devuelve una copia del DataFrame con los nuevos encabezados.
        """
        if self.data_original is None:
            print("Error: Los datos originales no han sido cargados. Llama a 'read_file()' primero.")
            return None
        
        # Inicializar el traductor
        translator = GoogleTranslator(source='de', target='en')

        # Obtener y traducir los encabezados
        encabezados_alemanes = self.data_original.columns.tolist()
        encabezados_ingleses = [translator.translate(col) for col in encabezados_alemanes]

        # Crear una copia del DataFrame original
        data = self.data_original.copy()
        
        # Aplicar los encabezados traducidos a la copia
        data.columns = encabezados_ingleses
        return data